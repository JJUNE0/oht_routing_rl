"""Controlled-center action mappings in the action actually seen by the critic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oht_routing.mdp.topology import ContextualTopology


MAX_SIMULATOR_COST = 16_777_215.99
REGION_B_RL = "region_b_rl"
FREE_FLOW_RESIDUAL = "free_flow_residual"
EXP_RESIDUAL = "exp_residual"
ACTION_MODES = (FREE_FLOW_RESIDUAL, REGION_B_RL, EXP_RESIDUAL)


class ContextualActionError(FloatingPointError):
    """Raised when action/cost mapping violates the runtime contract."""


@dataclass(frozen=True)
class ContextualActionResult:
    full_action: np.ndarray
    applied_controlled_action: np.ndarray
    final_cost: np.ndarray
    diagnostics: dict[str, float]


def _finite_vector(name: str, values, expected_length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_length,):
        raise ContextualActionError(
            f"{name} shape mismatch: actual={array.shape}, "
            f"expected=({expected_length},)"
        )
    if not np.isfinite(array).all():
        raise ContextualActionError(f"{name} contains NaN or Inf")
    return array


def validate_topology_partition(topology: ContextualTopology) -> None:
    physical_count = len(topology.all_rail_ids)
    controlled_count = len(topology.controlled_rail_ids)
    boundary_count = len(topology.boundary_rail_ids)
    if controlled_count + boundary_count != physical_count:
        raise ContextualActionError(
            "controlled/boundary counts do not partition physical rails"
        )
    controlled = set(map(int, topology.controlled_rail_ids))
    boundary = set(map(int, topology.boundary_rail_ids))
    physical = set(map(int, topology.all_rail_ids))
    if controlled & boundary:
        raise ContextualActionError("controlled and boundary rail IDs overlap")
    if controlled | boundary != physical:
        raise ContextualActionError(
            "controlled and boundary rail IDs do not cover physical topology"
        )
    physical_indices = np.asarray(
        topology.controlled_row_to_physical_index, dtype=np.int64
    )
    if physical_indices.shape != (controlled_count,):
        raise ContextualActionError(
            "controlled_row_to_physical_index shape mismatch"
        )
    if len(np.unique(physical_indices)) != controlled_count:
        raise ContextualActionError("controlled physical indices contain duplicates")
    if (physical_indices < 0).any() or (physical_indices >= physical_count).any():
        raise ContextualActionError("controlled physical index is out of range")
    aligned_ids = topology.all_rail_ids[physical_indices]
    if not np.array_equal(aligned_ids, topology.controlled_rail_ids):
        raise ContextualActionError(
            "controlled row, rail ID, and physical index are misaligned"
        )


def assemble_full_action(
    topology: ContextualTopology,
    controlled_action,
) -> np.ndarray:
    validate_topology_partition(topology)
    controlled = _finite_vector(
        "controlled_action",
        np.asarray(controlled_action).reshape(-1),
        len(topology.controlled_rail_ids),
    )
    full_action = np.zeros(len(topology.all_rail_ids), dtype=np.float32)
    full_action[topology.controlled_row_to_physical_index] = controlled.astype(
        np.float32
    )
    boundary_rows = np.flatnonzero(
        np.asarray(topology.physical_index_to_controlled_row) == -1
    )
    full_action[boundary_rows] = 0.0
    return np.ascontiguousarray(full_action)


def apply_controlled_action(
    baseline_cost,
    controlled_action,
    topology: ContextualTopology,
    *,
    action_enabled: bool,
    action_scale: float,
    action_mode: str = FREE_FLOW_RESIDUAL,
    base_cost=None,
    congestion_cost=None,
    rl_cost_lambda: float = 0.5,
    max_cost: float = MAX_SIMULATOR_COST,
) -> ContextualActionResult:
    """Map normalized actor actions to controlled rail costs."""
    validate_topology_partition(topology)
    physical_count = len(topology.all_rail_ids)
    baseline = _finite_vector("baseline_cost", baseline_cost, physical_count)
    if (baseline < 0.0).any() or (baseline > float(max_cost)).any():
        raise ContextualActionError(
            f"baseline_cost outside simulator range [0, {float(max_cost)}]"
        )
    scale = float(action_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ContextualActionError("action_scale must be finite and non-negative")
    if action_mode not in ACTION_MODES:
        raise ContextualActionError(
            f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}"
        )
    residual_lambda = float(rl_cost_lambda)
    if not np.isfinite(residual_lambda) or not 0.0 <= residual_lambda <= 1.0:
        raise ContextualActionError("rl_cost_lambda must be finite and in [0, 1]")
    controlled = _finite_vector(
        "controlled_action",
        np.asarray(controlled_action).reshape(-1),
        len(topology.controlled_rail_ids),
    )
    if (np.abs(controlled) > 1.0 + 1e-6).any():
        raise ContextualActionError("controlled_action must be in [-1, 1]")
    if action_mode == FREE_FLOW_RESIDUAL:
        # The cost-driving action is the actor's normalized action itself.
        # action_scale and the legacy b_rl representation are intentionally
        # absent from this mode.
        applied = (
            controlled.copy()
            if action_enabled else np.zeros_like(controlled)
        )
        effective_scale = 1.0
    else:
        applied = (
            np.clip(scale * controlled, -1.0, 1.0)
            if action_enabled and scale > 0.0
            else np.zeros_like(controlled)
        )
        effective_scale = scale
    full_action = assemble_full_action(topology, applied)
    final_cost = baseline.copy()
    controlled_rows = topology.controlled_row_to_physical_index
    boundary_rows = np.flatnonzero(
        np.asarray(topology.physical_index_to_controlled_row) == -1
    )
    b_rl = None
    rail_cost_diagnostics = None
    if action_mode in {FREE_FLOW_RESIDUAL, REGION_B_RL}:
        base = _finite_vector("base_cost", base_cost, physical_count)
        congestion = _finite_vector(
            "congestion_cost", congestion_cost, physical_count
        )
        neutral = base + 0.5 * congestion
        if not np.allclose(neutral, baseline, rtol=1e-10, atol=1e-10):
            raise ContextualActionError(
                "neutral rail-cost components do not match baseline"
            )
    if action_mode == FREE_FLOW_RESIDUAL:
        residual = residual_lambda * base[controlled_rows] * applied
        final_cost[controlled_rows] = baseline[controlled_rows] + residual
        rail_cost_diagnostics = {
            "rl/action_mean": float(applied.mean()),
            "rl/action_std": float(applied.std()),
            "rl/action_abs_mean": float(np.abs(applied).mean()),
            "rail_cost/t_ff_mean": float(base[controlled_rows].mean()),
            "rail_cost/congestion_mean": float(
                (0.5 * congestion[controlled_rows]).mean()
            ),
            "rail_cost/residual_mean": float(residual.mean()),
            "rail_cost/residual_abs_mean": float(np.abs(residual).mean()),
            "rail_cost/final_cost_mean": float(
                final_cost[controlled_rows].mean()
            ),
        }
    elif action_mode == REGION_B_RL:
        b_rl = 0.5 + 0.5 * applied
        final_cost[controlled_rows] = (
            base[controlled_rows]
            + congestion[controlled_rows] * b_rl
        )
    else:
        exponent = applied
        if not np.isfinite(exponent).all():
            raise ContextualActionError("action exponent contains NaN or Inf")
        controlled_cost = baseline[controlled_rows] * np.exp(exponent)
        if not np.isfinite(controlled_cost).all():
            raise ContextualActionError("controlled cost overflowed")
        final_cost[controlled_rows] = controlled_cost

    # Final safety assignment does not depend on the action representation.
    final_cost[boundary_rows] = baseline[boundary_rows]
    if (final_cost < 0.0).any() or (final_cost > float(max_cost)).any():
        raise ContextualActionError(
            f"final_cost outside simulator range [0, {float(max_cost)}]"
        )
    if not np.array_equal(final_cost[boundary_rows], baseline[boundary_rows]):
        raise ContextualActionError("boundary baseline-cost invariant failed")

    ratios = final_cost[controlled_rows] / np.maximum(
        baseline[controlled_rows], np.finfo(np.float64).tiny
    )
    diagnostics = {
        "action/controlled_mean": float(controlled.mean()),
        "action/controlled_std": float(controlled.std()),
        "action/controlled_min": float(controlled.min()),
        "action/controlled_max": float(controlled.max()),
        "action/saturation_ratio": float((np.abs(controlled) >= 0.999).mean()),
        "action/applied_controlled_mean": float(applied.mean()),
        "action/applied_controlled_std": float(applied.std()),
        "action/applied_controlled_min": float(applied.min()),
        "action/applied_controlled_max": float(applied.max()),
        "action/applied_scale": effective_scale,
        "boundary/action_abs_max": float(
            np.abs(full_action[boundary_rows]).max(initial=0.0)
        ),
        "boundary/cost_baseline_abs_error_max": float(
            np.abs(final_cost[boundary_rows] - baseline[boundary_rows]).max(
                initial=0.0
            )
        ),
        "cost/baseline_mean": float(baseline.mean()),
        "cost/final_mean": float(final_cost.mean()),
        "cost/all_baseline_abs_error_max": float(
            np.abs(final_cost - baseline).max(initial=0.0)
        ),
        "cost/controlled_ratio_mean": float(ratios.mean()),
        "cost/controlled_ratio_std": float(ratios.std()),
        "action/mode_region_b_rl": float(action_mode == REGION_B_RL),
        "action/mode_free_flow_residual": float(
            action_mode == FREE_FLOW_RESIDUAL
        ),
    }
    if b_rl is not None:
        diagnostics.update({
            "b_rl/mean": float(b_rl.mean()),
            "b_rl/std": float(b_rl.std()),
            "b_rl/min": float(b_rl.min()),
            "b_rl/max": float(b_rl.max()),
        })
    if rail_cost_diagnostics is not None:
        diagnostics.update(rail_cost_diagnostics)
    return ContextualActionResult(
        full_action=full_action,
        applied_controlled_action=np.ascontiguousarray(
            applied.astype(np.float32)
        ),
        final_cost=np.ascontiguousarray(final_cost.astype(np.float64)),
        diagnostics=diagnostics,
    )
