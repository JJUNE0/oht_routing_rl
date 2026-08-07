"""Controlled-center action mappings in the action actually seen by the critic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contextual_topology import ContextualTopology


MAX_SIMULATOR_COST = 16_777_215.99
REGION_B_RL = "region_b_rl"
EXP_RESIDUAL = "exp_residual"
ACTION_MODES = (REGION_B_RL, EXP_RESIDUAL)
ACTION_VERSIONS = {
    REGION_B_RL: "contextual_region_b_rl_v2",
    EXP_RESIDUAL: "contextual_exp_residual_v2",
}

# region_b_rl maps the applied action onto the congestion multiplier b of
#     final_cost = base_cost + congestion_cost * b
# The neutral action (applied == 0) must reproduce the baseline cost exactly, so
# B_RL_NEUTRAL is simultaneously the baseline's congestion share and the b the
# policy starts from, since a freshly initialized actor outputs approximately 0
# on every rail.
#
# v1 used neutral 0.5 with span 0.5, i.e. b in [0.0, 1.0].
#
# The fixed-b sweep over b in {0, 0.25, 0.5, 0.75, 1, 1.5, 2} measured TAT as a
# U-shaped curve with its minimum near b = 0.75 (~171-173); b = 0.25 reached
# ~185 and b = 0 drove the queue to ~500 and hit queue termination. That sweep
# holds b *uniform across all 4,996 rails*, so it constrains exactly one thing
# here: the neutral point, which is the only b the initial policy applies
# everywhere at once. Neutral 0.75 therefore starts training from ~171-173
# instead of ~174. It says nothing about the span, because once the policy
# differentiates, the sweep's curve no longer applies - measured on run
# xk7dxp8g over the 58,485 full-action-scale steps:
#
#   - b_rl/min was below 0.05 on 93.8% of steps and below 0.40 on 99.4%, i.e.
#     the policy continuously drove individual rails to b ~ 0 to make them
#     attractive, and env/queued still never exceeded 101 (p50 = 41, never
#     above 300). Per-rail b ~ 0 does not reproduce the uniform b = 0 collapse.
#   - The cross-rail mean drifts slowly (lag-1 autocorrelation 0.990) and spans
#     0.421-0.810 even in 5,000-step rolling windows, while TAT over the same
#     windows only spans 170.4-174.4, with corr(mean b, TAT) = -0.157. Under
#     uniform b that same level range would span roughly 174-185.
#   - corr(b_rl/std, TAT) = -0.176: more per-rail spread goes with slightly
#     *lower* TAT, consistent with TD7 reaching 3.1% versus 1.95% for the best
#     constant b.
#
# v2 therefore keeps neutral at the measured uniform optimum and widens the span
# rather than narrowing it, so b = 0 stays reachable as the traffic-attraction
# action: b in [0.00, 1.50]. The upper end is not a cliff either - the sweep
# reached only ~180 at b = 1.5.
B_RL_NEUTRAL = 0.75
B_RL_SPAN = 0.75
EXPLORATION_SCHEDULE_VERSION = "contextual_exploration_linear_anneal_v1"
# Default contract for standalone replay/test construction.
ACTION_VERSION = ACTION_VERSIONS[REGION_B_RL]


class ContextualActionError(FloatingPointError):
    """Raised when action/cost mapping violates the runtime contract."""


@dataclass(frozen=True)
class ContextualActionResult:
    full_action: np.ndarray
    applied_controlled_action: np.ndarray
    final_cost: np.ndarray
    diagnostics: dict[str, float]


def action_version(action_mode: str) -> str:
    try:
        return ACTION_VERSIONS[str(action_mode)]
    except KeyError as error:
        raise ContextualActionError(
            f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}"
        ) from error


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
    action_mode: str = REGION_B_RL,
    base_cost=None,
    congestion_cost=None,
    max_cost: float = MAX_SIMULATOR_COST,
) -> ContextualActionResult:
    """Scale raw action into applied space and map it to controlled rail costs."""
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
    version = action_version(action_mode)
    controlled = _finite_vector(
        "controlled_action",
        np.asarray(controlled_action).reshape(-1),
        len(topology.controlled_rail_ids),
    )
    if (np.abs(controlled) > 1.0 + 1e-6).any():
        raise ContextualActionError("controlled_action must be in [-1, 1]")
    applied = (
        np.clip(scale * controlled, -1.0, 1.0)
        if action_enabled and scale > 0.0
        else np.zeros_like(controlled)
    )
    full_action = assemble_full_action(topology, applied)
    final_cost = baseline.copy()
    controlled_rows = topology.controlled_row_to_physical_index
    boundary_rows = np.flatnonzero(
        np.asarray(topology.physical_index_to_controlled_row) == -1
    )
    b_rl = None
    if action_mode == REGION_B_RL:
        base = _finite_vector("base_cost", base_cost, physical_count)
        congestion = _finite_vector(
            "congestion_cost", congestion_cost, physical_count
        )
        neutral = base + B_RL_NEUTRAL * congestion
        if not np.allclose(neutral, baseline, rtol=1e-10, atol=1e-10):
            raise ContextualActionError(
                "region_b_rl neutral cost does not match baseline"
            )
        b_rl = B_RL_NEUTRAL + B_RL_SPAN * applied
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
        "action/applied_scale": scale,
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
        "action/version_region_b_rl_v1": float(
            version == ACTION_VERSIONS[REGION_B_RL]
        ),
    }
    if b_rl is not None:
        diagnostics.update({
            "b_rl/mean": float(b_rl.mean()),
            "b_rl/std": float(b_rl.std()),
            "b_rl/min": float(b_rl.min()),
            "b_rl/max": float(b_rl.max()),
        })
    return ContextualActionResult(
        full_action=full_action,
        applied_controlled_action=np.ascontiguousarray(
            applied.astype(np.float32)
        ),
        final_cost=np.ascontiguousarray(final_cost.astype(np.float64)),
        diagnostics=diagnostics,
    )
