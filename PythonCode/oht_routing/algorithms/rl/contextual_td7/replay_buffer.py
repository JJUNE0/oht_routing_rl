"""Memory-efficient state-ring replay for contextual per-rail transitions."""

from __future__ import annotations

import time

import numpy as np
import torch

from oht_routing.mdp.observation import (
    GLOBAL_DIM,
    LOCAL_PHYSICAL_DIM,
)
from oht_routing.mdp.reward.config import (
    REWARD_VERSION,
    canonical_reward_version,
)
from oht_routing.mdp.topology import ContextualTopology
from oht_routing.mdp.transition import CompletedContextualTransition
from oht_routing.version import (
    CONTEXTUAL_VERSION,
    is_compatible_contextual_version,
)

from .replay_types import (
    ContextualReplayBatch,
    ContextualStepSnapshot,
    ReplaySampleKey,
)
from .stacking import stack_offsets, validate_stack_config


class ContextualReplayError(RuntimeError):
    pass


REPLAY_SAMPLING_RAIL = "rail"
REPLAY_SAMPLING_SNAPSHOT = "snapshot"
REPLAY_SAMPLING_RANDOM_RAIL = "random_rail"
REPLAY_SAMPLING_MODES = (
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    REPLAY_SAMPLING_RANDOM_RAIL,
)


# Replay storage is deliberately narrower than the learner contract. Samples
# are always materialized as float32 before normalization/model use.
STATIC_PHYSICAL_FEATURE_INDICES = (0, 1, 2, 3)
UINT8_PHYSICAL_FEATURE_INDICES = (5, 8, 9, 10, 11, 12, 13, 15)
UINT16_PHYSICAL_FEATURE_INDICES = (6, 7, 14)
STATE_COUNT_UINT8_POSITIONS = (1, 2, 3, 4, 5, 6)
ACTION_FIXED_POINT_SCALE = float(np.iinfo(np.int16).max)
ACTION_FIXED_POINT_MAX_ABS_ERROR = 0.5 / ACTION_FIXED_POINT_SCALE
LAP_PRIORITY_MAX = float(np.finfo(np.float16).max)


def _copy_array(name, values, shape, dtype=np.float32):
    array = np.asarray(values, dtype=dtype)
    if array.shape != shape:
        raise ContextualReplayError(
            f"{name} shape mismatch: actual={array.shape}, expected={shape}"
        )
    if not np.isfinite(array).all():
        raise ContextualReplayError(f"{name} contains NaN or Inf")
    result = np.ascontiguousarray(array.copy())
    result.setflags(write=False)
    return result


def _quantize_action(name, values, shape):
    """Encode a bounded float action as symmetric signed 16-bit fixed point."""
    array = np.asarray(values, dtype=np.float32)
    if array.shape != shape:
        raise ContextualReplayError(
            f"{name} shape mismatch: actual={array.shape}, expected={shape}"
        )
    if not np.isfinite(array).all():
        raise ContextualReplayError(f"{name} contains NaN or Inf")
    if (np.abs(array) > 1.0 + 1e-6).any():
        raise ContextualReplayError(f"{name} must be in [-1, 1]")
    clipped = np.clip(array, -1.0, 1.0)
    return np.ascontiguousarray(
        np.rint(clipped * ACTION_FIXED_POINT_SCALE).astype(np.int16)
    )


def _decode_action(values):
    """Decode fixed-point replay actions to the learner's float32 contract."""
    return np.ascontiguousarray(
        np.asarray(values, dtype=np.float32)
        / np.float32(ACTION_FIXED_POINT_SCALE)
    )


def _encode_integral_columns(name, physical, indices, dtype):
    values = np.asarray(physical, dtype=np.float32)[:, indices]
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ContextualReplayError(
            f"{name} replay features must be exactly integral"
        )
    limit = np.iinfo(dtype)
    if (rounded < limit.min).any() or (rounded > limit.max).any():
        actual_min = float(rounded.min(initial=0.0))
        actual_max = float(rounded.max(initial=0.0))
        raise ContextualReplayError(
            f"{name} replay feature overflow for {np.dtype(dtype).name}: "
            f"range=[{actual_min}, {actual_max}], "
            f"allowed=[{limit.min}, {limit.max}]"
        )
    return np.ascontiguousarray(rounded.astype(dtype))


def snapshot_from_transition(
    transition: CompletedContextualTransition,
    *,
    reward_version: str = REWARD_VERSION,
) -> ContextualStepSnapshot:
    """Copy the raw state and controlled vectors from a completed transition."""
    state_raw = transition.state.physical_local_raw
    state_global = transition.state.global_raw
    next_raw = transition.next_state.physical_local_raw
    next_global = transition.next_state.global_raw
    if any(value is None for value in (state_raw, state_global, next_raw, next_global)):
        raise ContextualReplayError(
            "transition observations do not contain raw snapshot exports"
        )
    physical_count = int(np.asarray(state_raw).shape[0])
    controlled_count = int(transition.action.shape[0])
    return ContextualStepSnapshot(
        physical_local_state=_copy_array(
            "physical_local_state",
            state_raw,
            (physical_count, LOCAL_PHYSICAL_DIM),
        ),
        global_state=_copy_array("global_state", state_global, (GLOBAL_DIM,)),
        previous_applied_action=_copy_array(
            "previous_applied_action",
            transition.state.previous_applied_action,
            (controlled_count, 1),
        ),
        policy_action=_copy_array(
            "policy_action", transition.action, (controlled_count, 1)
        ),
        applied_action=_copy_array(
            "applied_action", transition.applied_action, (controlled_count, 1)
        ),
        reward=_copy_array(
            "reward", transition.reward.total, (controlled_count,)
        ),
        next_physical_local_state=_copy_array(
            "next_physical_local_state",
            next_raw,
            (physical_count, LOCAL_PHYSICAL_DIM),
        ),
        next_global_state=_copy_array(
            "next_global_state", next_global, (GLOBAL_DIM,)
        ),
        next_previous_applied_action=_copy_array(
            "next_previous_applied_action",
            transition.next_state.previous_applied_action,
            (controlled_count, 1),
        ),
        done=bool(transition.done),
        env_step=int(transition.env_step),
        next_env_step=int(transition.next_env_step),
        episode_id=int(transition.episode_id),
        topology_hash=transition.topology_hash,
        mapping_hash=transition.mapping_hash,
        version=CONTEXTUAL_VERSION,
        reward_version=str(reward_version),
    )


class ContextualStepReplayBuffer:
    """Fixed-capacity CPU ring indexed logically by (step slot, rail row)."""

    def __init__(
        self,
        topology: ContextualTopology,
        observation_builder,
        *,
        capacity_env_steps: int = 1_000,
        seed: int = 0,
        lap_enabled: bool = False,
        lap_alpha: float = 0.4,
        lap_min_priority: float = 1.0,
        reward_version: str = REWARD_VERSION,
        sampling_mode: str = REPLAY_SAMPLING_RAIL,
        num_stacks: int = 1,
        stack_interval: int = 1,
    ):
        if int(capacity_env_steps) <= 0:
            raise ValueError("capacity_env_steps must be positive")
        if sampling_mode not in REPLAY_SAMPLING_MODES:
            raise ValueError(
                f"sampling_mode must be one of {REPLAY_SAMPLING_MODES}"
            )
        if (
            sampling_mode
            in {REPLAY_SAMPLING_SNAPSHOT, REPLAY_SAMPLING_RANDOM_RAIL}
            and lap_enabled
        ):
            raise ValueError(
                f"{sampling_mode} replay sampling requires LAP disabled"
            )
        self.topology = topology
        self.observation_builder = observation_builder
        self.capacity = int(capacity_env_steps)
        # Worst case is one completed transition per episode: no adjacent
        # state can be shared, so C valid transitions require 2C state slots.
        self.state_capacity = 2 * self.capacity
        self.controlled_count = len(topology.controlled_rail_ids)
        self.physical_count = len(topology.all_rail_ids)
        self.rng = np.random.default_rng(seed)
        self.lap_enabled = bool(lap_enabled)
        self.lap_alpha = float(lap_alpha)
        self.lap_min_priority = float(lap_min_priority)
        self.reward_version = canonical_reward_version(reward_version)
        self.sampling_mode = str(sampling_mode)
        self.num_stacks, self.stack_interval = validate_stack_config(
            num_stacks, stack_interval
        )
        self._stack_offsets = stack_offsets(
            self.num_stacks, self.stack_interval
        )
        if (
            not np.isfinite(self.lap_alpha)
            or self.lap_alpha <= 0.0
            or not np.isfinite(self.lap_min_priority)
            or self.lap_min_priority <= 0.0
        ):
            raise ValueError("LAP alpha/min priority must be finite and positive")
        if self.lap_enabled and self.lap_min_priority > LAP_PRIORITY_MAX:
            raise ValueError(
                "lap_min_priority exceeds float16 replay storage range: "
                f"{self.lap_min_priority} > {LAP_PRIORITY_MAX}"
            )
        if self.lap_enabled:
            stored_min_priority = np.float16(self.lap_min_priority)
            if (
                not np.isfinite(stored_min_priority)
                or stored_min_priority <= 0.0
            ):
                raise ValueError(
                    "lap_min_priority underflows finite positive float16 "
                    "replay storage"
                )
        else:
            stored_min_priority = self.lap_min_priority
        self._priority = (
            np.zeros((self.capacity, self.controlled_count), np.float16)
            if self.lap_enabled else None
        )
        self._priority_sum = (
            np.zeros(self.capacity, np.float64)
            if self.lap_enabled else None
        )
        self._priority_sq_sum = (
            np.zeros(self.capacity, np.float64)
            if self.lap_enabled else None
        )
        self._priority_min = (
            np.zeros(self.capacity, np.float16)
            if self.lap_enabled else None
        )
        self._priority_max = (
            np.zeros(self.capacity, np.float16)
            if self.lap_enabled else None
        )
        self.max_priority = float(stored_min_priority)
        self.priority_update_count = 0

        # Adjacent transitions share states. The 2C allocation also preserves
        # full transition capacity under pathological reset frequency.
        self._physical_static = np.empty(
            (self.physical_count, len(STATIC_PHYSICAL_FEATURE_INDICES)),
            np.float32,
        )
        self._physical_distance_mm = np.empty(
            self.physical_count, np.float64
        )
        self._physical_static_initialized = False
        self._physical_uint8 = np.empty(
            (
                self.state_capacity,
                self.physical_count,
                len(UINT8_PHYSICAL_FEATURE_INDICES),
            ),
            np.uint8,
        )
        self._physical_uint16 = np.empty(
            (
                self.state_capacity,
                self.physical_count,
                len(UINT16_PHYSICAL_FEATURE_INDICES),
            ),
            np.uint16,
        )
        self._global = np.empty(
            (self.state_capacity, GLOBAL_DIM), np.float32
        )
        self._state_generation = np.zeros(self.state_capacity, np.int64)
        self._state_valid = np.zeros(self.state_capacity, bool)
        self._state_keys: list[tuple[int, int] | None] = [
            None
        ] * self.state_capacity
        self._key_to_state: dict[tuple[int, int], tuple[int, int]] = {}
        self._state_cursor = 0

        shape = (self.capacity, self.controlled_count)
        self._previous_applied_action = np.empty(shape, np.int16)
        self._policy_action = np.empty(shape, np.int16)
        self._applied_action = np.empty(shape, np.int16)
        self._reward = np.empty(shape, np.float32)
        self._done = np.empty(self.capacity, np.float32)
        self._env_step = np.empty(self.capacity, np.int64)
        self._episode_id = np.empty(self.capacity, np.int64)
        self._state_slot = np.empty(self.capacity, np.int64)
        self._state_gen_ref = np.empty(self.capacity, np.int64)
        self._next_state_slot = np.empty(self.capacity, np.int64)
        self._next_state_gen_ref = np.empty(self.capacity, np.int64)
        self._transition_generation = np.zeros(self.capacity, np.int64)
        self._transition_valid = np.zeros(self.capacity, bool)
        self._transition_keys: list[tuple[int, int] | None] = [
            None
        ] * self.capacity
        self._key_to_transition: dict[tuple[int, int], tuple[int, int]] = {}
        self._episode_first_step: dict[int, int] = {}
        self._episode_transition_counts: dict[int, int] = {}
        self._transition_cursor = 0

        id_to_physical = {
            int(rail_id): row
            for row, rail_id in enumerate(topology.all_rail_ids)
        }
        self._center_rows = np.ascontiguousarray(
            topology.controlled_row_to_physical_index.astype(np.int64)
        )
        self._incoming_rows = np.ascontiguousarray(np.asarray([
            [id_to_physical[int(rail_id)] for rail_id in row]
            for row in topology.incoming_neighbor_ids
        ], dtype=np.int64))
        self._outgoing_rows = np.ascontiguousarray(np.asarray([
            [id_to_physical[int(rail_id)] for rail_id in row]
            for row in topology.outgoing_neighbor_ids
        ], dtype=np.int64))
        if self._incoming_rows.shape != self._outgoing_rows.shape:
            raise ContextualReplayError(
                "incoming/outgoing topology mapping shapes differ"
            )
        self.neighbor_count = int(self._incoming_rows.shape[1])

        self.push_count = 0
        self.sample_count = 0
        self.overwrite_count = 0
        self.stale_key_reject_count = 0
        self.hash_mismatch_count = 0
        self.episode_crossing_count = 0
        self._last_sample_diag: dict[str, float] = {}

    def _validate_snapshot(self, snapshot: ContextualStepSnapshot) -> None:
        if not is_compatible_contextual_version(snapshot.version):
            raise ContextualReplayError(
                "runtime version mismatch: "
                f"{snapshot.version} != compatible {CONTEXTUAL_VERSION}"
            )
        if snapshot.topology_hash != self.topology.topology_hash:
            self.hash_mismatch_count += 1
            raise ContextualReplayError("topology hash mismatch")
        if snapshot.mapping_hash != self.topology.mapping_hash:
            self.hash_mismatch_count += 1
            raise ContextualReplayError("mapping hash mismatch")
        if snapshot.reward_version != self.reward_version:
            raise ContextualReplayError(
                "reward version mismatch: "
                f"{snapshot.reward_version} != {self.reward_version}"
            )
        if snapshot.next_env_step != snapshot.env_step + 1:
            raise ContextualReplayError("next_env_step must equal env_step + 1")
        _copy_array("physical_local_state", snapshot.physical_local_state,
                    (self.physical_count, LOCAL_PHYSICAL_DIM))
        _copy_array("global_state", snapshot.global_state, (GLOBAL_DIM,))
        _quantize_action(
            "previous_applied_action",
            snapshot.previous_applied_action,
            (self.controlled_count, 1),
        )
        _quantize_action(
            "policy_action",
            snapshot.policy_action,
            (self.controlled_count, 1),
        )
        _quantize_action(
            "applied_action",
            snapshot.applied_action,
            (self.controlled_count, 1),
        )
        _copy_array("reward", snapshot.reward, (self.controlled_count,))
        _copy_array("next_physical_local_state",
                    snapshot.next_physical_local_state,
                    (self.physical_count, LOCAL_PHYSICAL_DIM))
        _copy_array("next_global_state", snapshot.next_global_state,
                    (GLOBAL_DIM,))
        _copy_array("next_previous_applied_action",
                    snapshot.next_previous_applied_action,
                    (self.controlled_count, 1))
        if not np.array_equal(
            snapshot.next_previous_applied_action,
            snapshot.applied_action,
        ):
            raise ContextualReplayError(
                "next_previous_applied_action must equal applied_action"
            )

    def _pack_physical(self, physical):
        array = np.asarray(physical, dtype=np.float32)
        static = np.ascontiguousarray(
            array[:, STATIC_PHYSICAL_FEATURE_INDICES]
        )
        if not self._physical_static_initialized:
            distance = np.asarray(
                self.observation_builder.physical_distance_mm,
                dtype=np.float64,
            )
            if distance.shape != (self.physical_count,):
                raise ContextualReplayError(
                    "physical_distance_mm shape mismatch: "
                    f"actual={distance.shape}, "
                    f"expected=({self.physical_count},)"
                )
            if not np.isfinite(distance).all() or (distance <= 0.0).any():
                raise ContextualReplayError(
                    "physical_distance_mm must be finite and positive"
                )
        else:
            distance = self._physical_distance_mm
            if not np.array_equal(self._physical_static, static):
                raise ContextualReplayError(
                    "static physical replay features changed after initialization"
                )

        uint8_values = _encode_integral_columns(
            "uint8 physical",
            array,
            UINT8_PHYSICAL_FEATURE_INDICES,
            np.uint8,
        )
        uint16_values = _encode_integral_columns(
            "uint16 physical",
            array,
            UINT16_PHYSICAL_FEATURE_INDICES,
            np.uint16,
        )
        occupancy = uint8_values[
            :, STATE_COUNT_UINT8_POSITIONS
        ].sum(axis=1, dtype=np.uint16)
        if (occupancy > np.iinfo(np.uint8).max).any():
            raise ContextualReplayError(
                "sum of per-state OHT counts exceeds the one-byte rail "
                "OhtList protocol range"
            )
        stopped_count = uint8_values[:, -1].astype(np.uint16)
        if (stopped_count > occupancy).any():
            raise ContextualReplayError(
                "stopped_oht_count exceeds total per-state OHT count"
            )
        stop_time_sum = uint16_values[:, -1].astype(np.uint32)
        if (stop_time_sum > occupancy.astype(np.uint32) * 255).any():
            raise ContextualReplayError(
                "stop_time_sum exceeds the one-byte StopTime protocol bound"
            )
        density = (
            occupancy.astype(np.float64)
            / (distance / 1_000.0)
        ).astype(np.float32)
        if not np.array_equal(density, array[:, 4]):
            error = float(np.max(np.abs(density - array[:, 4]), initial=0.0))
            raise ContextualReplayError(
                "oht_density cannot be reconstructed bit-exactly from OHT "
                f"state counts and rail Distance: max_abs_error={error}"
            )
        if not self._physical_static_initialized:
            self._physical_static[:] = static
            self._physical_distance_mm[:] = distance
            self._physical_static_initialized = True
        return uint8_values, uint16_values

    def _materialize_physical(self, state_slots, physical_rows):
        uint8_values = self._physical_uint8[state_slots, physical_rows]
        uint16_values = self._physical_uint16[state_slots, physical_rows]
        result = np.empty(
            uint8_values.shape[:-1] + (LOCAL_PHYSICAL_DIM,),
            dtype=np.float32,
        )
        result[..., STATIC_PHYSICAL_FEATURE_INDICES] = (
            self._physical_static[physical_rows]
        )
        result[..., UINT8_PHYSICAL_FEATURE_INDICES] = uint8_values
        result[..., UINT16_PHYSICAL_FEATURE_INDICES] = uint16_values
        occupancy = uint8_values[
            ..., STATE_COUNT_UINT8_POSITIONS
        ].sum(axis=-1, dtype=np.uint16)
        distance = self._physical_distance_mm[physical_rows]
        result[..., 4] = (
            occupancy.astype(np.float64) / (distance / 1_000.0)
        ).astype(np.float32)
        return result

    def _store_state(
        self, key, physical, global_state
    ) -> tuple[int, int]:
        physical_uint8, physical_uint16 = self._pack_physical(physical)
        existing = self._key_to_state.get(key)
        if existing is not None:
            slot, generation = existing
            if (
                self._state_valid[slot]
                and self._state_generation[slot] == generation
            ):
                if not (
                    np.array_equal(
                        self._physical_uint8[slot], physical_uint8
                    )
                    and np.array_equal(
                        self._physical_uint16[slot], physical_uint16
                    )
                    and np.array_equal(self._global[slot], global_state)
                ):
                    raise ContextualReplayError(
                        "same episode/env step has different raw state"
                    )
                return slot, generation

        slot = self._state_cursor
        self._state_cursor = (slot + 1) % self.state_capacity
        old_key = self._state_keys[slot]
        if old_key is not None and self._key_to_state.get(old_key, (None,))[0] == slot:
            self._key_to_state.pop(old_key, None)
        generation = int(self._state_generation[slot]) + 1
        self._physical_uint8[slot] = physical_uint8
        self._physical_uint16[slot] = physical_uint16
        self._global[slot] = global_state
        self._state_generation[slot] = generation
        self._state_valid[slot] = True
        self._state_keys[slot] = key
        self._key_to_state[key] = (slot, generation)
        return slot, generation

    def _state_reference(self, episode: int, step: int) -> tuple[int, int] | None:
        origin = self._episode_first_step.get(int(episode))
        if origin is None:
            return None
        requested = max(int(step), int(origin))
        reference = self._key_to_state.get((int(episode), requested))
        if reference is None:
            return None
        slot, generation = reference
        if not (
            self._state_valid[slot]
            and self._state_generation[slot] == generation
        ):
            return None
        return int(slot), int(generation)

    def _transition_reference(
        self, episode: int, step: int, *, pad_episode_start: bool = False
    ) -> tuple[int, int] | None:
        origin = self._episode_first_step.get(int(episode))
        if origin is None:
            return None
        requested = int(step)
        if requested < int(origin):
            if not pad_episode_start:
                return None
            requested = int(origin)
        reference = self._key_to_transition.get((int(episode), requested))
        if reference is None:
            return None
        slot, generation = reference
        if not (
            self._transition_valid[slot]
            and self._transition_generation[slot] == generation
        ):
            return None
        return int(slot), int(generation)

    def push(self, snapshot: ContextualStepSnapshot) -> ReplaySampleKey:
        self._validate_snapshot(snapshot)
        state_key = (int(snapshot.episode_id), int(snapshot.env_step))
        next_key = (int(snapshot.episode_id), int(snapshot.next_env_step))
        if state_key[0] != next_key[0]:
            self.episode_crossing_count += 1
            raise ContextualReplayError("episode-crossing snapshot")
        existing = self._key_to_transition.get(state_key)
        if existing is not None:
            existing_slot, existing_generation = existing
            if (
                self._transition_valid[existing_slot]
                and self._transition_generation[existing_slot]
                == existing_generation
            ):
                raise ContextualReplayError(
                    "duplicate live transition key: "
                    f"episode_id={state_key[0]}, env_step={state_key[1]}"
                )
            self._key_to_transition.pop(state_key, None)
        previous_action_stored = _quantize_action(
            "previous_applied_action",
            snapshot.previous_applied_action,
            (self.controlled_count, 1),
        )[:, 0]
        policy_action_stored = _quantize_action(
            "policy_action",
            snapshot.policy_action,
            (self.controlled_count, 1),
        )[:, 0]
        applied_action_stored = _quantize_action(
            "applied_action",
            snapshot.applied_action,
            (self.controlled_count, 1),
        )[:, 0]
        prior_transition = self._transition_reference(
            state_key[0], state_key[1] - 1
        )
        if prior_transition is not None and not np.array_equal(
            previous_action_stored,
            self._applied_action[prior_transition[0]],
        ):
            raise ContextualReplayError(
                "contiguous transition previous_applied_action differs from "
                "the prior transition applied_action after fixed-point encoding"
            )
        state_slot, state_gen = self._store_state(
            state_key,
            snapshot.physical_local_state,
            snapshot.global_state,
        )
        next_slot, next_gen = self._store_state(
            next_key, snapshot.next_physical_local_state,
            snapshot.next_global_state,
        )

        slot = self._transition_cursor
        self._transition_cursor = (slot + 1) % self.capacity
        if self._transition_valid[slot]:
            self.overwrite_count += 1
            old_key = self._transition_keys[slot]
            if old_key is not None:
                current = self._key_to_transition.get(old_key)
                if current is not None and current[0] == slot:
                    self._key_to_transition.pop(old_key, None)
                old_episode = int(old_key[0])
                remaining = (
                    self._episode_transition_counts.get(old_episode, 1) - 1
                )
                if remaining > 0:
                    self._episode_transition_counts[old_episode] = remaining
                else:
                    self._episode_transition_counts.pop(old_episode, None)
                    if old_episode != state_key[0]:
                        self._episode_first_step.pop(old_episode, None)
        generation = int(self._transition_generation[slot]) + 1
        self._episode_first_step.setdefault(state_key[0], state_key[1])
        self._previous_applied_action[slot] = previous_action_stored
        self._policy_action[slot] = policy_action_stored
        self._applied_action[slot] = applied_action_stored
        self._reward[slot] = snapshot.reward
        self._done[slot] = float(snapshot.done)
        self._env_step[slot] = snapshot.env_step
        self._episode_id[slot] = snapshot.episode_id
        self._state_slot[slot], self._state_gen_ref[slot] = state_slot, state_gen
        self._next_state_slot[slot] = next_slot
        self._next_state_gen_ref[slot] = next_gen
        self._transition_generation[slot] = generation
        self._transition_valid[slot] = True
        self._transition_keys[slot] = state_key
        self._key_to_transition[state_key] = (slot, generation)
        self._episode_transition_counts[state_key[0]] = (
            self._episode_transition_counts.get(state_key[0], 0) + 1
        )
        if self.lap_enabled:
            stored_priority = np.float16(self.max_priority)
            self._priority[slot].fill(stored_priority)
            stored_priority_float = float(stored_priority)
            self._priority_sum[slot] = (
                stored_priority_float * self.controlled_count
            )
            self._priority_sq_sum[slot] = (
                stored_priority_float
                * stored_priority_float
                * self.controlled_count
            )
            self._priority_min[slot] = stored_priority
            self._priority_max[slot] = stored_priority
        self.push_count += 1
        return ReplaySampleKey(slot, generation, 0)

    def push_transition(
        self, transition: CompletedContextualTransition
    ) -> ReplaySampleKey:
        return self.push(
            snapshot_from_transition(
                transition,
                reward_version=self.reward_version,
            )
        )

    def _valid_transition_slots(self) -> np.ndarray:
        candidates = np.flatnonzero(self._transition_valid)
        if candidates.size == 0:
            return candidates
        state_ok = (
            self._state_valid[self._state_slot[candidates]]
            & (self._state_generation[self._state_slot[candidates]]
               == self._state_gen_ref[candidates])
        )
        next_ok = (
            self._state_valid[self._next_state_slot[candidates]]
            & (self._state_generation[self._next_state_slot[candidates]]
               == self._next_state_gen_ref[candidates])
        )
        candidates = candidates[state_ok & next_ok]
        if self.num_stacks == 1 or candidates.size == 0:
            return candidates

        # A retained episode origin may be duplicated for its initial stack.
        # If that origin has already fallen out of the circular buffer, only
        # sample steps at least one complete history horizon after the oldest
        # retained transition for that episode.
        eligible = np.zeros(candidates.size, dtype=bool)
        episodes = self._episode_id[candidates]
        steps = self._env_step[candidates]
        history_horizon = self._stack_offsets[-1]
        for episode in np.unique(episodes):
            positions = np.flatnonzero(episodes == episode)
            episode_steps = steps[positions]
            origin = self._episode_first_step.get(int(episode))
            origin_retained = bool(
                origin is not None and np.any(episode_steps == int(origin))
            )
            if origin_retained:
                eligible[positions] = True
            else:
                oldest_retained = int(episode_steps.min())
                eligible[positions] = (
                    episode_steps >= oldest_retained + history_horizon
                )
        return candidates[eligible]

    def validate_sample_keys(self, keys) -> None:
        keys = tuple(keys)
        if not keys:
            return
        slots = np.fromiter(
            (key.step_slot for key in keys), dtype=np.int64, count=len(keys)
        )
        generations = np.fromiter(
            (key.generation for key in keys), dtype=np.int64, count=len(keys)
        )
        rows = np.fromiter(
            (key.controlled_row for key in keys),
            dtype=np.int64,
            count=len(keys),
        )
        in_bounds = (
            (slots >= 0) & (slots < self.capacity)
            & (rows >= 0) & (rows < self.controlled_count)
        )
        valid = in_bounds.copy()
        bounded = np.flatnonzero(valid)
        if bounded.size:
            checked_slots = slots[bounded]
            valid[bounded] &= (
                self._transition_valid[checked_slots]
                & (
                    self._transition_generation[checked_slots]
                    == generations[bounded]
                )
            )
        bounded = np.flatnonzero(valid)
        if bounded.size and self.num_stacks > 1:
            valid_slots = self._valid_transition_slots()
            valid[bounded] &= np.isin(slots[bounded], valid_slots)
        bounded = np.flatnonzero(valid)
        if bounded.size:
            checked_slots = slots[bounded]
            state_slots = self._state_slot[checked_slots]
            next_slots = self._next_state_slot[checked_slots]
            valid[bounded] &= (
                self._state_valid[state_slots]
                & (
                    self._state_generation[state_slots]
                    == self._state_gen_ref[checked_slots]
                )
                & self._state_valid[next_slots]
                & (
                    self._state_generation[next_slots]
                    == self._next_state_gen_ref[checked_slots]
                )
            )
        if not valid.all():
            self.stale_key_reject_count += 1
            bad = int(np.flatnonzero(~valid)[0])
            raise ContextualReplayError(f"stale replay key: {keys[bad]}")

    def update_priorities(self, sample_keys, priorities) -> None:
        self.validate_sample_keys(sample_keys)
        if not self.lap_enabled:
            raise NotImplementedError(
                "priority update is unsupported when LAP is disabled"
            )
        values = np.asarray(priorities, dtype=np.float64).reshape(-1)
        if values.shape != (len(sample_keys),) or not np.isfinite(values).all():
            raise ContextualReplayError("priorities must be finite [sample keys]")
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.maximum(
                np.power(np.maximum(values, 0.0), self.lap_alpha),
                self.lap_min_priority,
            )
        if (
            not np.isfinite(values).all()
            or (values > LAP_PRIORITY_MAX).any()
        ):
            actual_max = float(values.max(initial=0.0))
            raise ContextualReplayError(
                "LAP priority exceeds finite float16 replay storage: "
                f"max={actual_max}, allowed_max={LAP_PRIORITY_MAX}"
            )
        stored_values = values.astype(np.float16)
        slots = np.fromiter(
            (key.step_slot for key in sample_keys),
            dtype=np.int64,
            count=len(sample_keys),
        )
        rows = np.fromiter(
            (key.controlled_row for key in sample_keys),
            dtype=np.int64,
            count=len(sample_keys),
        )
        # A sampled pair may occur more than once. Preserve the former
        # sequential-update contract by retaining the last supplied value.
        flat = slots * self.controlled_count + rows
        _, reverse_positions = np.unique(flat[::-1], return_index=True)
        positions = len(flat) - 1 - reverse_positions
        slots = slots[positions]
        rows = rows[positions]
        stored_values = stored_values[positions]
        old = self._priority[slots, rows].astype(np.float64)
        values64 = stored_values.astype(np.float64)
        self._priority[slots, rows] = stored_values
        np.add.at(self._priority_sum, slots, values64 - old)
        np.add.at(
            self._priority_sq_sum,
            slots,
            np.square(values64) - np.square(old),
        )
        affected_slots = np.unique(slots)
        affected = self._priority[affected_slots]
        self._priority_min[affected_slots] = affected.min(axis=1)
        self._priority_max[affected_slots] = affected.max(axis=1)
        valid_slots = self._valid_transition_slots()
        self.max_priority = (
            max(
                self.lap_min_priority,
                float(self._priority_max[valid_slots].max(initial=0.0)),
            )
            if valid_slots.size else self.lap_min_priority
        )
        self.priority_update_count += 1

    @staticmethod
    def _readonly_normalize(normalizer, values, name):
        # normalize() is read-only; update() is intentionally never called.
        return normalizer.normalize(values, name=name)

    def _history_slot_arrays(self, transition_slots: np.ndarray):
        unique_slots, inverse = np.unique(
            np.asarray(transition_slots, dtype=np.int64), return_inverse=True
        )
        shape = (unique_slots.size, self.num_stacks)
        state_history = np.empty(shape, np.int64)
        next_state_history = np.empty(shape, np.int64)
        action_history = np.full(shape, -1, np.int64)
        next_action_history = np.full(shape, -1, np.int64)
        for row, transition_slot in enumerate(unique_slots):
            episode = int(self._episode_id[transition_slot])
            step = int(self._env_step[transition_slot])
            origin = self._episode_first_step.get(episode)
            if origin is None:
                raise ContextualReplayError("sampled episode has no stack origin")
            for index, offset in enumerate(self._stack_offsets):
                state_reference = self._state_reference(
                    episode, step - offset
                )
                next_state_reference = self._state_reference(
                    episode, step + 1 - offset
                )
                if state_reference is None or next_state_reference is None:
                    raise ContextualReplayError(
                        "sampled transition lost required state history"
                    )
                state_history[row, index] = state_reference[0]
                next_state_history[row, index] = next_state_reference[0]

                action_step = step - offset
                action_reference = self._transition_reference(
                    episode, action_step, pad_episode_start=True
                )
                if action_reference is None:
                    raise ContextualReplayError(
                        "sampled transition lost required action history"
                    )
                action_history[row, index] = action_reference[0]

                if index > 0:
                    next_action_step = step + 1 - offset
                    next_action_reference = self._transition_reference(
                        episode,
                        next_action_step,
                        pad_episode_start=True,
                    )
                    if next_action_reference is None:
                        raise ContextualReplayError(
                            "sampled transition lost required next-action history"
                        )
                    next_action_history[row, index] = next_action_reference[0]
        return (
            state_history[inverse],
            next_state_history[inverse],
            action_history[inverse],
            next_action_history[inverse],
        )

    def sample(
        self, batch_size: int, *, device: str | torch.device = "cpu"
    ) -> ContextualReplayBatch:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        valid_slots = self._valid_transition_slots()
        if valid_slots.size == 0:
            raise ContextualReplayError("cannot sample an empty replay")
        started = time.perf_counter()
        if self.sampling_mode == REPLAY_SAMPLING_RANDOM_RAIL:
            logical_count = int(valid_slots.size * self.controlled_count)
            if int(batch_size) > logical_count:
                raise ContextualReplayError(
                    "random-rail batch size exceeds the number of valid "
                    f"logical transitions: batch={int(batch_size)}, "
                    f"valid={logical_count}"
                )
            # Treat the complete (valid step, controlled rail) Cartesian
            # product as one flat replay pool. Sampling without replacement
            # guarantees that a logical rail transition appears at most once
            # in an optimizer batch, while allowing any number of different
            # rails to come from the same environment step.
            logical_indices = self.rng.choice(
                logical_count,
                size=int(batch_size),
                replace=False,
                shuffle=False,
            )
            transition_slots = valid_slots[
                logical_indices // self.controlled_count
            ]
            controlled_rows = (
                logical_indices % self.controlled_count
            ).astype(np.int64, copy=False)
            sample_probabilities = np.full(
                int(batch_size),
                1.0 / logical_count,
                np.float64,
            )
        elif self.sampling_mode == REPLAY_SAMPLING_SNAPSHOT:
            if int(batch_size) > valid_slots.size:
                raise ContextualReplayError(
                    "snapshot batch size exceeds the number of valid "
                    f"environment steps: batch={int(batch_size)}, "
                    f"valid={int(valid_slots.size)}"
                )
            # A snapshot batch means distinct environment steps, with every
            # controlled rail from each selected step included exactly once.
            snapshot_slots = self.rng.choice(
                valid_slots, size=int(batch_size), replace=False
            )
            transition_slots = np.repeat(
                snapshot_slots, self.controlled_count
            )
            controlled_rows = np.tile(
                np.arange(self.controlled_count, dtype=np.int64),
                int(batch_size),
            )
            sample_probabilities = np.full(
                transition_slots.size,
                1.0 / len(valid_slots),
                np.float64,
            )
        # Rail sampling is the original behavior. With replacement is
        # explicit: any positive batch is allowed once one snapshot exists.
        elif self.lap_enabled:
            step_sums = self._priority_sum[valid_slots]
            total_priority = float(step_sums.sum())
            if not np.isfinite(total_priority) or total_priority <= 0:
                raise ContextualReplayError("invalid LAP total priority")
            transition_slots = self.rng.choice(
                valid_slots, size=int(batch_size), replace=True,
                p=step_sums / total_priority,
            )
            controlled_rows = np.empty(int(batch_size), np.int64)
            unique_slots, inverse = np.unique(
                transition_slots, return_inverse=True
            )
            # Build every selected step CDF in one NumPy kernel.  The former
            # implementation scanned ``inverse`` and launched a separate
            # cumsum/search for almost every batch row because duplicate
            # environment slots are rare at the production replay size.
            cdfs = np.cumsum(
                self._priority[unique_slots], axis=1, dtype=np.float64
            )

            # Preserve the exact RNG assignment of the former grouped loop:
            # groups follow sorted ``unique_slots`` and rows within a group
            # retain their original batch order.  One vector draw consumes
            # the same generator sequence as the former consecutive draws.
            grouped_positions = np.argsort(inverse, kind="stable")
            grouped_random = self.rng.random(int(batch_size))
            row_random = np.empty(int(batch_size), dtype=np.float64)
            row_random[grouped_positions] = grouped_random
            thresholds = row_random * cdfs[inverse, -1]

            # NumPy has no row-wise searchsorted, so perform all row searches
            # together with a logarithmic batched binary search.  The
            # ``<=`` branch is exactly searchsorted(..., side="right").
            lower = np.zeros(int(batch_size), dtype=np.int64)
            upper = np.full(
                int(batch_size), self.controlled_count, dtype=np.int64
            )
            while np.any(lower < upper):
                active = lower < upper
                middle = (lower + upper) // 2
                # A completed row may equal controlled_count while another
                # row still searches. Clamp only the unused probe for it.
                probe = np.minimum(middle, self.controlled_count - 1)
                move_right = active & (
                    cdfs[inverse, probe] <= thresholds
                )
                lower = np.where(move_right, middle + 1, lower)
                upper = np.where(active & ~move_right, middle, upper)
            controlled_rows[:] = lower
            sample_probabilities = (
                self._priority[transition_slots, controlled_rows].astype(
                    np.float64
                )
                / total_priority
            )
        else:
            transition_slots = self.rng.choice(
                valid_slots, size=int(batch_size), replace=True
            )
            controlled_rows = self.rng.integers(
                0, self.controlled_count, size=int(batch_size), dtype=np.int64
            )
            sample_probabilities = np.full(
                int(batch_size),
                1.0 / (len(valid_slots) * self.controlled_count),
                np.float64,
            )
        (
            state_slots,
            next_slots,
            action_slots,
            next_action_slots,
        ) = self._history_slot_arrays(transition_slots)
        batch_count = int(transition_slots.size)
        state_global_raw = self._global[state_slots]
        next_global_raw = self._global[next_slots]
        action_rows = controlled_rows[:, None]
        previous_applied_action = _decode_action(
            self._previous_applied_action[action_slots, action_rows]
        )
        next_previous_applied_action = np.empty_like(
            previous_applied_action
        )
        # The immediate next state's previous action is this transition's
        # applied action. Older stacked next states have their own transition-
        # aligned previous-action record (including episode-start padding).
        next_previous_applied_action[:, 0] = _decode_action(
            self._applied_action[action_slots[:, 0], controlled_rows]
        )
        if self.num_stacks > 1:
            next_previous_applied_action[:, 1:] = _decode_action(
                self._previous_applied_action[
                    next_action_slots[:, 1:], action_rows
                ]
            )
        global_norm = self._readonly_normalize(
            self.observation_builder.global_normalizer,
            state_global_raw.reshape(-1, GLOBAL_DIM), "replay_state_global"
        ).reshape(batch_count, self.num_stacks, GLOBAL_DIM)
        next_global_norm = self._readonly_normalize(
            self.observation_builder.global_normalizer,
            next_global_raw.reshape(-1, GLOBAL_DIM), "replay_next_global"
        ).reshape(batch_count, self.num_stacks, GLOBAL_DIM)
        center_rows = self._center_rows[controlled_rows]
        incoming_rows = self._incoming_rows[controlled_rows]
        outgoing_rows = self._outgoing_rows[controlled_rows]
        center_rail_index = np.broadcast_to(
            center_rows[:, None], (batch_count, self.num_stacks)
        )
        incoming_rail_indices = np.broadcast_to(
            incoming_rows[:, None, :],
            (batch_count, self.num_stacks, incoming_rows.shape[1]),
        )
        outgoing_rail_indices = np.broadcast_to(
            outgoing_rows[:, None, :],
            (batch_count, self.num_stacks, outgoing_rows.shape[1]),
        )
        current_center_raw = self._materialize_physical(
            state_slots, center_rows[:, None]
        )
        current_incoming_raw = self._materialize_physical(
            state_slots[:, :, None], incoming_rows[:, None, :]
        )
        current_outgoing_raw = self._materialize_physical(
            state_slots[:, :, None], outgoing_rows[:, None, :]
        )
        next_center_raw = self._materialize_physical(
            next_slots, center_rows[:, None]
        )
        next_incoming_raw = self._materialize_physical(
            next_slots[:, :, None], incoming_rows[:, None, :]
        )
        next_outgoing_raw = self._materialize_physical(
            next_slots[:, :, None], outgoing_rows[:, None, :]
        )
        local_normalizer = self.observation_builder.local_normalizer
        center = self._readonly_normalize(
            local_normalizer,
            current_center_raw.reshape(-1, LOCAL_PHYSICAL_DIM),
            "replay_center_local",
        ).reshape(batch_count, self.num_stacks, LOCAL_PHYSICAL_DIM)
        incoming = self._readonly_normalize(
            local_normalizer,
            current_incoming_raw.reshape(-1, LOCAL_PHYSICAL_DIM),
            "replay_incoming_local",
        ).reshape(current_incoming_raw.shape)
        outgoing = self._readonly_normalize(
            local_normalizer,
            current_outgoing_raw.reshape(-1, LOCAL_PHYSICAL_DIM),
            "replay_outgoing_local",
        ).reshape(current_outgoing_raw.shape)
        next_center = self._readonly_normalize(
            local_normalizer,
            next_center_raw.reshape(-1, LOCAL_PHYSICAL_DIM),
            "replay_next_center_local",
        ).reshape(batch_count, self.num_stacks, LOCAL_PHYSICAL_DIM)
        next_incoming = self._readonly_normalize(
            local_normalizer,
            next_incoming_raw.reshape(-1, LOCAL_PHYSICAL_DIM),
            "replay_next_incoming_local",
        ).reshape(next_incoming_raw.shape)
        next_outgoing = self._readonly_normalize(
            local_normalizer,
            next_outgoing_raw.reshape(-1, LOCAL_PHYSICAL_DIM),
            "replay_next_outgoing_local",
        ).reshape(next_outgoing_raw.shape)

        relation_shape = (
            batch_count,
            self.num_stacks,
            self._incoming_rows.shape[1],
            self.observation_builder._incoming_relation.shape[-1],
        )
        incoming_relation = np.broadcast_to(
            self.observation_builder._incoming_relation[controlled_rows, None],
            relation_shape,
        )
        outgoing_relation = np.broadcast_to(
            self.observation_builder._outgoing_relation[controlled_rows, None],
            relation_shape,
        )

        safe_action_slots = np.maximum(action_slots, 0)
        safe_next_action_slots = np.maximum(next_action_slots, 0)
        policy_action = _decode_action(
            self._policy_action[safe_action_slots, action_rows]
        )
        applied_action = _decode_action(
            self._applied_action[safe_action_slots, action_rows]
        )
        next_applied_action = _decode_action(
            self._applied_action[safe_next_action_slots, action_rows]
        )
        policy_action[action_slots < 0] = 0.0
        applied_action[action_slots < 0] = 0.0
        next_applied_action[next_action_slots < 0] = 0.0

        if self.num_stacks == 1:
            center = center[:, 0]
            incoming = incoming[:, 0]
            outgoing = outgoing[:, 0]
            center_rail_index = center_rail_index[:, 0]
            incoming_rail_indices = incoming_rail_indices[:, 0]
            outgoing_rail_indices = outgoing_rail_indices[:, 0]
            incoming_relation = incoming_relation[:, 0]
            outgoing_relation = outgoing_relation[:, 0]
            global_norm = global_norm[:, 0]
            previous_applied_action = previous_applied_action[:, 0, None]
            next_center = next_center[:, 0]
            next_incoming = next_incoming[:, 0]
            next_outgoing = next_outgoing[:, 0]
            next_global_norm = next_global_norm[:, 0]
            next_previous_applied_action = (
                next_previous_applied_action[:, 0, None]
            )
            policy_action = policy_action[:, 0, None]
            applied_action = applied_action[:, 0, None]
            next_applied_action = next_applied_action[:, 0, None]
        else:
            previous_applied_action = previous_applied_action[:, :, None]
            policy_action = policy_action[:, :, None]
            applied_action = applied_action[:, :, None]
            next_previous_applied_action = (
                next_previous_applied_action[:, :, None]
            )
            next_applied_action = next_applied_action[:, :, None]

        arrays = {
            "center_local": center,
            "incoming_local": incoming,
            "outgoing_local": outgoing,
            "center_rail_index": center_rail_index,
            "incoming_rail_indices": incoming_rail_indices,
            "outgoing_rail_indices": outgoing_rail_indices,
            "incoming_relation": incoming_relation,
            "outgoing_relation": outgoing_relation,
            "global_state": global_norm,
            "previous_applied_action": previous_applied_action,
            "policy_action": policy_action,
            "applied_action": applied_action,
            "reward": self._reward[transition_slots, controlled_rows][:, None],
            "next_center_local": next_center,
            "next_incoming_local": next_incoming,
            "next_outgoing_local": next_outgoing,
            "next_incoming_relation": incoming_relation,
            "next_outgoing_relation": outgoing_relation,
            "next_global_state": next_global_norm,
            "next_previous_applied_action": next_previous_applied_action,
            "next_applied_action": next_applied_action,
            "done": self._done[transition_slots][:, None],
            "controlled_rail_id": self.topology.controlled_rail_ids[
                controlled_rows
            ],
            "env_step": self._env_step[transition_slots],
            "episode_id": self._episode_id[transition_slots],
        }
        materialize_ms = (time.perf_counter() - started) * 1000
        host_started = time.perf_counter()
        target = torch.device(device)
        tensors = {}
        for name, values in arrays.items():
            if name in {
                "center_rail_index",
                "incoming_rail_indices",
                "outgoing_rail_indices",
                "controlled_rail_id",
                "env_step",
                "episode_id",
            }:
                array = np.array(
                    values, dtype=np.int64, order="C", copy=True
                )
            else:
                array = np.array(
                    values, dtype=np.float32, order="C", copy=True
                )
            tensors[name] = torch.from_numpy(array).to(target)
        host_to_device_ms = (time.perf_counter() - host_started) * 1000
        keys = tuple(
            ReplaySampleKey(
                int(slot), int(self._transition_generation[slot]), int(row)
            )
            for slot, row in zip(transition_slots, controlled_rows)
        )
        self.sample_count += 1
        self._last_sample_diag = {
            "replay/sample_reward_mean": float(arrays["reward"].mean()),
            "replay/sample_reward_std": float(arrays["reward"].std()),
            "replay/sample_done_ratio": float(arrays["done"].mean()),
            "replay/sample_policy_action_std": float(
                arrays["policy_action"].std()
            ),
            "replay/sample_applied_action_std": float(
                arrays["applied_action"].std()
            ),
            "stack/num_stacks": float(self.num_stacks),
            "stack/interval": float(self.stack_interval),
            "replay/sample_materialize_ms": materialize_ms,
            "replay/host_to_device_ms": host_to_device_ms,
            "lap/sample_probability_max": float(
                sample_probabilities.max(initial=0)
            ),
            "lap/effective_sample_size": float(
                1.0 / np.square(
                    sample_probabilities / sample_probabilities.sum()
                ).sum()
            ),
        }
        return ContextualReplayBatch(**tensors, sample_keys=keys)

    @property
    def size_env_steps(self) -> int:
        return int(self._valid_transition_slots().size)

    @staticmethod
    def estimate_capacity_bytes(
        capacity_env_steps: int,
        physical_count: int = 4_999,
        controlled_count: int = 4_996,
        neighbor_count: int = 15,
        lap_enabled: bool = False,
    ) -> int:
        c = int(capacity_env_steps)
        state_capacity = 2 * c
        packed_physical_bytes = (
            len(UINT8_PHYSICAL_FEATURE_INDICES)
            + 2 * len(UINT16_PHYSICAL_FEATURE_INDICES)
        )
        static_physical = physical_count * (
            len(STATIC_PHYSICAL_FEATURE_INDICES) * 4 + 8
        )
        state = state_capacity * (
            physical_count * packed_physical_bytes + GLOBAL_DIM * 4
        ) + static_physical
        # Previous, deterministic policy, and applied actions use signed
        # int16 fixed point. Reward remains float32.
        vectors = c * controlled_count * (3 * 2 + 4)
        metadata = c * (7 * 8 + 4 + 1) + (2 * c) * (8 + 1)
        static_mapping = controlled_count * (
            1 + 2 * int(neighbor_count)
        ) * 8
        priority = (
            c * controlled_count * 2 + c * (2 * 8 + 2 * 2)
            if lap_enabled else 0
        )
        return int(state + vectors + metadata + static_mapping + priority)

    @property
    def storage_bytes(self) -> int:
        total = 0
        for value in self.__dict__.values():
            if isinstance(value, np.ndarray):
                total += value.nbytes
        return int(total)

    def diagnostics(self) -> dict[str, float]:
        result = {
            "replay/capacity_env_steps": float(self.capacity),
            "replay/capacity_logical_transitions": float(
                self.capacity * self.controlled_count
            ),
            "replay/size_env_steps": float(self.size_env_steps),
            "replay/size_logical_transitions": float(
                self.size_env_steps * self.controlled_count
            ),
            "replay/push_count": float(self.push_count),
            "replay/sample_count": float(self.sample_count),
            "replay/overwrite_count": float(self.overwrite_count),
            "replay/stale_key_reject_count": float(
                self.stale_key_reject_count
            ),
            "replay/boundary_transition_count": 0.0,
            "replay/episode_crossing_count": float(self.episode_crossing_count),
            "replay/hash_mismatch_count": float(self.hash_mismatch_count),
            "replay/stack_boundary_excluded_env_steps": float(
                np.count_nonzero(self._transition_valid)
                - self.size_env_steps
            ),
            "replay/storage_bytes": float(self.storage_bytes),
            "replay/estimated_capacity_bytes": float(
                self.estimate_capacity_bytes(
                    self.capacity,
                    self.physical_count,
                    self.controlled_count,
                    neighbor_count=self.neighbor_count,
                    lap_enabled=self.lap_enabled,
                )
            ),
            "replay/storage_physical_bytes_per_rail_state": float(
                len(UINT8_PHYSICAL_FEATURE_INDICES)
                + 2 * len(UINT16_PHYSICAL_FEATURE_INDICES)
            ),
            "replay/storage_action_fixed_point_scale": float(
                ACTION_FIXED_POINT_SCALE
            ),
            "replay/storage_action_max_abs_error": float(
                ACTION_FIXED_POINT_MAX_ABS_ERROR
            ),
            "replay/storage_lap_priority_bytes": float(
                np.dtype(np.float16).itemsize if self.lap_enabled else 0
            ),
            "lap/enabled": float(self.lap_enabled),
            "lap/stale_update_reject_count": float(
                self.stale_key_reject_count
            ),
            "lap/new_transition_max_priority": float(self.max_priority),
            "stack/num_stacks": float(self.num_stacks),
            "stack/interval": float(self.stack_interval),
            "stack/history_horizon": float(
                (self.num_stacks - 1) * self.stack_interval
            ),
        }
        if self.lap_enabled:
            slots = self._valid_transition_slots()
            count = slots.size * self.controlled_count
            total = float(self._priority_sum[slots].sum())
            total_sq = float(self._priority_sq_sum[slots].sum())
            mean = total / count if count else 0.0
            variance = (
                max(0.0, total_sq / count - mean * mean)
                if count else 0.0
            )
            result.update({
                "lap/priority_mean": mean,
                "lap/priority_std": float(np.sqrt(variance)),
                "lap/priority_min": (
                    float(self._priority_min[slots].min())
                    if slots.size else 0.0
                ),
                "lap/priority_max": (
                    float(self._priority_max[slots].max())
                    if slots.size else 0.0
                ),
            })
        result.update(self._last_sample_diag)
        return result
