"""Canonical CPU packing and batched policy inference for collectors."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7.stacking import (
    encode_observation_stack,
    flatten_action_stack,
    flatten_state_stack,
)


@dataclass(frozen=True)
class PackedPolicyInput:
    """One collector policy input with explicit batch and stack axes."""

    tensors: tuple[torch.Tensor, ...]
    previous_action: torch.Tensor
    rows: int
    pack_ms: float


@dataclass(frozen=True)
class MergedPolicyInput:
    """Collector inputs concatenated along their controlled-rail axis."""

    tensors: tuple[torch.Tensor, ...]
    previous_action: torch.Tensor
    row_slices: tuple[slice, ...]


def _as_frames(observation_frames) -> tuple[object, ...]:
    frames = (
        tuple(observation_frames)
        if isinstance(observation_frames, (tuple, list))
        else (observation_frames,)
    )
    if not frames:
        raise ValueError("observation_frames must not be empty")
    return frames


def _stack_frame_values(frames, name: str, rows: int) -> torch.Tensor:
    values = []
    for frame in frames:
        value = np.asarray(getattr(frame, name))
        if value.ndim == 0 or int(value.shape[0]) != int(rows):
            raise ValueError(
                f"{name} must have leading controlled-rail axis {int(rows)}"
            )
        values.append(value)
    try:
        stacked = np.stack(values, axis=1)
    except ValueError as error:
        raise ValueError(f"{name} frame shapes differ") from error
    return torch.from_numpy(np.ascontiguousarray(stacked))


def pack_policy_input(observation_frames) -> PackedPolicyInput:
    """Pack one collector's observation history as canonical ``[B,K,...]``.

    Dynamic rail-local values are stacked on axis one.  Actor-global values
    are one vector per frame and are broadcast across every controlled rail.
    No device transfer occurs here, so listener threads can safely call this
    function without touching CUDA.
    """

    started = time.perf_counter()
    frames = _as_frames(observation_frames)
    first_center = np.asarray(getattr(frames[0], "center_local"))
    if first_center.ndim < 2:
        raise ValueError("center_local must have shape [B,...]")
    rows = int(first_center.shape[0])
    if rows <= 0:
        raise ValueError("policy input must contain at least one rail row")

    tensor_names = (
        "center_local",
        "incoming_local",
        "outgoing_local",
        "center_rail_index",
        "incoming_rail_indices",
        "outgoing_rail_indices",
        "incoming_relation",
        "outgoing_relation",
    )
    tensors = tuple(
        _stack_frame_values(frames, name, rows) for name in tensor_names
    )

    global_values = []
    for frame in frames:
        value = np.asarray(getattr(frame, "global_state"))
        if value.ndim != 1:
            raise ValueError("global_state must have shape [G] per frame")
        global_values.append(value)
    try:
        global_stack = np.stack(global_values, axis=0)
    except ValueError as error:
        raise ValueError("global_state frame shapes differ") from error
    global_batch = np.broadcast_to(
        global_stack[None, ...], (rows, *global_stack.shape)
    )
    tensors = (
        *tensors,
        torch.from_numpy(np.ascontiguousarray(global_batch)),
    )
    previous_action = _stack_frame_values(
        frames, "previous_applied_action", rows
    )
    if previous_action.ndim != 3 or previous_action.shape[-1] != 1:
        raise ValueError(
            "previous_applied_action must produce shape [B,K,1]"
        )

    return PackedPolicyInput(
        tensors=tensors,
        previous_action=previous_action,
        rows=rows,
        pack_ms=(time.perf_counter() - started) * 1000.0,
    )


def _validate_packed(item: PackedPolicyInput) -> None:
    if not isinstance(item, PackedPolicyInput):
        raise TypeError("items must contain PackedPolicyInput values")
    if len(item.tensors) != 9:
        raise ValueError("packed policy input must contain nine tensors")
    if int(item.rows) <= 0:
        raise ValueError("packed policy rows must be positive")
    if item.tensors[0].ndim < 3:
        raise ValueError("packed tensors must have explicit [B,K,...] axes")
    batch_stack = tuple(item.tensors[0].shape[:2])
    if batch_stack[0] != int(item.rows):
        raise ValueError("packed rows differ from the tensor batch axis")
    for value in (*item.tensors, item.previous_action):
        if value.device.type != "cpu":
            raise ValueError("packed policy tensors must remain on CPU")
        if tuple(value.shape[:2]) != batch_stack:
            raise ValueError("packed policy batch/stack axes differ")
    if item.previous_action.ndim != 3 or item.previous_action.shape[-1] != 1:
        raise ValueError("previous_action must have shape [B,K,1]")


def merge_policy_inputs(items) -> MergedPolicyInput:
    """Concatenate compatible collector inputs and retain split boundaries."""

    packed_items = tuple(items)
    if not packed_items:
        raise ValueError("items must not be empty")
    for item in packed_items:
        _validate_packed(item)

    reference = packed_items[0]
    for item in packed_items[1:]:
        if len(item.tensors) != len(reference.tensors):
            raise ValueError("packed policy tensor counts differ")
        for index, (current, expected) in enumerate(
            zip(item.tensors, reference.tensors)
        ):
            if (
                tuple(current.shape[1:]) != tuple(expected.shape[1:])
                or current.dtype != expected.dtype
            ):
                raise ValueError(
                    f"packed policy tensor {index} schemas differ"
                )
        if (
            tuple(item.previous_action.shape[1:])
            != tuple(reference.previous_action.shape[1:])
            or item.previous_action.dtype != reference.previous_action.dtype
        ):
            raise ValueError("packed previous_action schemas differ")

    row_slices = []
    offset = 0
    for item in packed_items:
        next_offset = offset + int(item.rows)
        row_slices.append(slice(offset, next_offset))
        offset = next_offset

    return MergedPolicyInput(
        tensors=tuple(
            torch.cat([item.tensors[index] for item in packed_items], dim=0)
            for index in range(9)
        ),
        previous_action=torch.cat(
            [item.previous_action for item in packed_items], dim=0
        ),
        row_slices=tuple(row_slices),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_policy_batch(
    merged: MergedPolicyInput,
    *,
    device,
    encoder,
    actor,
    sale_fixed,
    network_config,
    attention_diagnostics: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run one merged inference with the standalone runtime's semantics."""

    if not isinstance(merged, MergedPolicyInput):
        raise TypeError("merged must be a MergedPolicyInput")
    if len(merged.tensors) != 9:
        raise ValueError("merged policy input must contain nine tensors")
    resolved_device = torch.device(device)
    attention_diagnostics = bool(
        attention_diagnostics and network_config.use_attention
    )

    host_started = time.perf_counter()
    device_tensors = tuple(
        tensor.to(resolved_device, non_blocking=False)
        for tensor in merged.tensors
    )
    previous_action = merged.previous_action.to(
        resolved_device, non_blocking=False
    )
    _synchronize(resolved_device)
    host_to_device_ms = (time.perf_counter() - host_started) * 1000.0

    inference_started = time.perf_counter()
    with torch.inference_mode():
        encoded_stack = encode_observation_stack(
            encoder,
            device_tensors,
            return_attention=attention_diagnostics,
        )
        actor_state = flatten_state_stack(
            encoded_stack.state, network_config.num_stacks
        )
        actor_previous_action = flatten_action_stack(
            previous_action,
            num_stacks=network_config.num_stacks,
            action_dim=network_config.action_dim,
        )
        sale_state = (
            sale_fixed.state(device_tensors)
            if sale_fixed is not None
            else None
        )
        actor_output = (
            actor(
                actor_state,
                sale_state,
                previous_action=actor_previous_action,
            )
            if sale_state is not None
            else actor(
                actor_state,
                previous_action=actor_previous_action,
            )
        )
    _synchronize(resolved_device)
    encoder_actor_ms = (time.perf_counter() - inference_started) * 1000.0

    copy_started = time.perf_counter()
    controlled_action = (
        actor_output.action[:, 0].detach().to("cpu").numpy().copy()
    )
    _synchronize(resolved_device)
    device_to_host_ms = (time.perf_counter() - copy_started) * 1000.0

    diagnostics = {
        # Packing is deliberately performed by collector threads and recorded
        # on each PackedPolicyInput.  This forward-only boundary therefore has
        # no tensor-conversion work of its own, but retains the legacy key.
        "runtime/tensor_conversion_ms": 0.0,
        "runtime/host_to_device_ms": host_to_device_ms,
        "runtime/encoder_actor_ms": encoder_actor_ms,
        "runtime/device_to_host_ms": device_to_host_ms,
    }
    if attention_diagnostics:
        batch = int(device_tensors[0].shape[0])
        for direction, weights in (
            (
                "incoming",
                encoded_stack.flat_encoding.incoming_attention,
            ),
            (
                "outgoing",
                encoded_stack.flat_encoding.outgoing_attention,
            ),
        ):
            if weights is None:
                raise RuntimeError(
                    "attention diagnostics requested without attention weights"
                )
            weights = weights.reshape(
                batch, network_config.num_stacks, *weights.shape[1:]
            )[:, 0]
            probabilities = weights.clamp_min(1e-12)
            diagnostics[f"attention/{direction}_entropy"] = float(
                (-(probabilities * probabilities.log()).sum(-1))
                .mean()
                .detach()
                .cpu()
            )
            diagnostics[f"attention/{direction}_max_weight"] = float(
                probabilities.max(-1).values.mean().detach().cpu()
            )
    return controlled_action, diagnostics


__all__ = (
    "MergedPolicyInput",
    "PackedPolicyInput",
    "forward_policy_batch",
    "merge_policy_inputs",
    "pack_policy_input",
)
