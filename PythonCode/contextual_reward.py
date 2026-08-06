"""Controlled-center reward construction for contextual rail control."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from Oht import OHTState
from contextual_action import ACTION_MODES, EXP_RESIDUAL, REGION_B_RL
from contextual_observation import RunningFeatureNormalizer
from contextual_reward_diagnostic import RewardDiagnosticWriter
from contextual_topology import ContextualTopology

REWARD_VERSION = (
    "contextual_controlled_reward_v13_balanced_freeflow_neutral2"
)

RAIL_REWARD_FIXED_TAT_REFERENCE = "fixed_tat_reference"
RAIL_REWARD_BASELINE_RATIO = "baseline_ratio"
RAIL_REWARD_FREE_FLOW_NEUTRAL_2 = "free_flow_neutral_2"
RAIL_REWARD_MODES = (
    RAIL_REWARD_FIXED_TAT_REFERENCE,
    RAIL_REWARD_BASELINE_RATIO,
    RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
)
# Deprecated source-level aliases for compatibility-only callers.
RAIL_TAT_FIXED_REFERENCE = RAIL_REWARD_FIXED_TAT_REFERENCE
RAIL_TAT_FREE_FLOW_RATIO = RAIL_REWARD_BASELINE_RATIO
RAIL_TAT_MODES = RAIL_REWARD_MODES

ACTIVE_OHT_CYCLE_STATES = frozenset({
    int(OHTState.MOVE_TO_LOAD),
    int(OHTState.LOADING),
    int(OHTState.MOVE_TO_UNLOAD),
    int(OHTState.UNLOADING),
})


class ContextualRewardError(RuntimeError):
    """Raised when reward inputs violate the controlled-center contract."""


@dataclass(frozen=True)
class ContextualRewardConfig:
    global_alpha: float = 0.5
    local_alpha: float = 0.5
    rail_tat_weight: float = 50.0
    rail_reward_mode: str = RAIL_REWARD_FREE_FLOW_NEUTRAL_2
    rail_free_flow_neutral_ratio: float = 2.0
    rail_baseline_ratio_reference: float | None = None
    action_mode: str = REGION_B_RL
    smooth_b_rl_weight: float = 0.25
    smooth_exp_residual_weight: float = 0.5
    tat_weight: float = 18.4
    op_weight: float = 5.0
    backlog_weight: float = 0.0005
    local_predicted_oht_weight: float = 0.10
    use_tat: bool = True
    use_op: bool = True
    use_backlog: bool = True
    tat_reference: float = 2.90706 * 60.0
    op_reference: float = 0.80
    tat_confidence_n0: float = 50.0
    tat_confidence_ramp: bool = False
    freeze_after_env_steps: int = 30_000
    local_reward_scale: float = 2.0
    tat_raw_clip: float | None = 1.0
    rail_tat_clip: float | None = 1.0
    normalizer_epsilon: float = 1e-6
    global_clip: float | None = 5.0
    local_clip: float | None = None

    def __post_init__(self):
        numeric = (
            self.global_alpha, self.local_alpha, self.rail_tat_weight,
            self.smooth_b_rl_weight, self.smooth_exp_residual_weight,
            self.tat_weight, self.op_weight,
            self.backlog_weight, self.tat_reference, self.op_reference,
            self.tat_confidence_n0, self.normalizer_epsilon,
            self.local_reward_scale, self.local_predicted_oht_weight,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("reward config contains NaN or Inf")
        if self.tat_reference <= 0 or self.normalizer_epsilon <= 0:
            raise ValueError("tat_reference and normalizer_epsilon must be positive")
        if self.local_reward_scale <= 0:
            raise ValueError("local_reward_scale must be positive")
        if self.backlog_weight < 0:
            raise ValueError("backlog_weight must be non-negative")
        if self.local_predicted_oht_weight < 0:
            raise ValueError(
                "local_predicted_oht_weight must be non-negative"
            )
        if self.rail_tat_weight < 0:
            raise ValueError("rail_tat_weight must be non-negative")
        if self.rail_reward_mode not in RAIL_REWARD_MODES:
            raise ValueError(
                f"rail_reward_mode must be one of {RAIL_REWARD_MODES}"
            )
        if (
            not np.isfinite(self.rail_free_flow_neutral_ratio)
            or self.rail_free_flow_neutral_ratio <= 0
        ):
            raise ValueError(
                "rail_free_flow_neutral_ratio must be finite and positive"
            )
        if self.rail_reward_mode == RAIL_REWARD_BASELINE_RATIO and (
            self.rail_baseline_ratio_reference is None
            or not np.isfinite(self.rail_baseline_ratio_reference)
            or self.rail_baseline_ratio_reference <= 0
        ):
            raise ValueError(
                "rail_baseline_ratio_reference must be finite and positive "
                "in baseline_ratio compatibility mode"
            )
        if self.tat_raw_clip is not None and (
            not np.isfinite(self.tat_raw_clip) or self.tat_raw_clip <= 0
        ):
            raise ValueError("tat_raw_clip must be positive or None")
        if self.rail_tat_clip is not None and (
            not np.isfinite(self.rail_tat_clip) or self.rail_tat_clip <= 0
        ):
            raise ValueError("rail_tat_clip must be positive or None")
        if not 0.0 <= self.op_reference <= 1.0:
            raise ValueError("op_reference must be in [0, 1]")
        if self.tat_confidence_n0 <= 0:
            raise ValueError("tat_confidence_n0 must be finite and positive")
        if self.freeze_after_env_steps < 0:
            raise ValueError("freeze_after_env_steps must be non-negative")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        if (
            self.smooth_b_rl_weight < 0
            or self.smooth_exp_residual_weight < 0
        ):
            raise ValueError("smooth weights must be non-negative")


@dataclass(frozen=True)
class ControlledRewardBatch:
    total: np.ndarray
    global_raw: float
    global_component: float
    local_raw: np.ndarray
    local_scaled: np.ndarray
    local_component: np.ndarray
    rail_reward_raw: np.ndarray
    rail_reward_weighted_preclip: np.ndarray
    rail_reward_postclip: np.ndarray
    rail_tat_event_count: int
    smooth_control_delta: np.ndarray
    smooth_penalty: np.ndarray
    controlled_rail_ids: np.ndarray
    total_tat_level: float
    tat_signal_available: float
    tat_error: float
    tat_raw_preclip: float
    tat_raw_postclip: float
    tat_confidence: float
    tat_raw_ramped: float
    completed_episode: float
    completed_delta: float
    op_rate: float
    op_reference: float
    op_error: float
    op_delta: float
    op_raw: float
    backlog: float
    backlog_raw: float
    local_oht_raw: np.ndarray
    local_predicted_raw: np.ndarray
    local_stop_raw: np.ndarray
    local_idle_raw: np.ndarray
    local_capacity_raw: np.ndarray
    waiting: float
    queued: float
    idle_oht_observation: np.ndarray
    smooth_weight_effective: float
    env_step: int
    episode_id: int

    @property
    def global_normalized(self):
        """Deprecated compatibility alias; Reward J is never normalized."""
        return self.global_raw

    @property
    def local_normalized(self):
        """Deprecated compatibility alias for the fixed local scale."""
        return self.local_scaled

    @property
    def tat_raw_unramped(self):
        """Deprecated compatibility alias for the post-clip TAT term."""
        return self.tat_raw_postclip

    @property
    def rail_tat_raw(self):
        """Deprecated penalty-direction alias."""
        return -self.rail_reward_raw

    @property
    def rail_tat_weighted_preclip(self):
        """Deprecated penalty-direction alias."""
        return -self.rail_reward_weighted_preclip

    @property
    def rail_tat_penalty(self):
        """Deprecated penalty-direction alias."""
        return -self.rail_reward_postclip


@dataclass
class OHTCycleTracker:
    oht_id: int
    cycle_active: bool = False
    bootstrapped: bool = False
    cycle_start_step: int | None = None
    cycle_start_global_step: int | None = None
    previous_state: int | None = None
    current_state: int | None = None
    latest_job_id: int = 0
    previous_job_ids: set[int] = field(default_factory=set)
    closed_segment_oht_tat_sum: float = 0.0
    current_segment_last_valid_oht_tat: float | None = None
    current_segment_last_state5_oht_tat: float | None = None
    current_segment_job_id: int = 0
    tat_segment_count: int = 0
    cycle_tat_valid: bool = True
    last_valid_cmd_tat: float | None = None
    last_tat_skip_reason: str | None = None
    rail_time_by_id: dict[int, float] = field(default_factory=dict)
    route_count: int = 0
    route_time: float = 0.0
    pass_delta_count: int = 0
    boundary_carry_count: int = 0
    boundary_carry_time: float = 0.0
    boundary_time_subtracted: float = 0.0
    ambiguous_pass_count: int = 0
    invalid_pass_delta_count: int = 0
    rail_occurrences: list = field(default_factory=list)
    route_free_flow_time: float = 0.0
    completed_rail_occurrence_count: int = 0
    free_flow_missing_count: int = 0
    free_flow_invalid_count: int = 0

    def clear_cycle(self) -> None:
        self.cycle_active = False
        self.bootstrapped = False
        self.cycle_start_step = None
        self.cycle_start_global_step = None
        self.closed_segment_oht_tat_sum = 0.0
        self.current_segment_last_valid_oht_tat = None
        self.current_segment_last_state5_oht_tat = None
        self.current_segment_job_id = 0
        self.tat_segment_count = 0
        self.cycle_tat_valid = True
        self.last_valid_cmd_tat = None
        self.last_tat_skip_reason = None
        self.rail_time_by_id.clear()
        self.route_count = 0
        self.route_time = 0.0
        self.pass_delta_count = 0
        self.boundary_carry_count = 0
        self.boundary_carry_time = 0.0
        self.boundary_time_subtracted = 0.0
        self.ambiguous_pass_count = 0
        self.invalid_pass_delta_count = 0
        self.rail_occurrences.clear()
        self.route_free_flow_time = 0.0
        self.completed_rail_occurrence_count = 0
        self.free_flow_missing_count = 0
        self.free_flow_invalid_count = 0

    @property
    def last_valid_oht_tat(self) -> float | None:
        """Compatibility alias used by diagnostics and older tests."""
        return self.current_segment_last_valid_oht_tat

    @property
    def last_unloading_oht_tat(self) -> float | None:
        """Compatibility alias used by diagnostics and older tests."""
        return self.current_segment_last_state5_oht_tat


@dataclass(frozen=True)
class RailOccurrenceDiagnostic:
    occurrence_index: int
    rail_id: int
    state: int
    actual_elapsed: float
    distance_per_velocity: float | None
    controlled: bool


@dataclass
class RailPassTemporalTracker:
    """Packet-to-packet state for one OHT's current rail occurrence."""

    inflight_rail_id: int | None = None
    inflight_state: int | None = None
    inflight_elapsed: float | None = None
    inflight_attributed_elapsed: float = 0.0

    def clear(self) -> None:
        self.inflight_rail_id = None
        self.inflight_state = None
        self.inflight_elapsed = None
        self.inflight_attributed_elapsed = 0.0


@dataclass(frozen=True)
class CycleRewardOutcome:
    reward_applied: bool
    skip_reason: str | None
    effective_oht_tat: float | None = None
    signed_excess: float | None = None
    cycle_rail_reward_raw: float | None = None
    route_free_flow_ratio: float | None = None
    cycle_reward_unclipped: float = 0.0
    cycle_reward_clipped: float = 0.0
    reward_rail_count: int = 0
    controlled_reward_count: int = 0
    uncontrolled_reward_count: int = 0
    reward_sum: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0


class ContextualRewardBuilder:
    """Build one reward row per controlled rail from the post-action snapshot."""

    def __init__(
        self,
        topology: ContextualTopology,
        config: ContextualRewardConfig | None = None,
        *,
        completion_diagnostic_path: str | Path | None = None,
        global_step_provider: Callable[[], int] | None = None,
        completion_diagnostic_max_global_step: int | None = None,
        reward_diagnostic_writer: RewardDiagnosticWriter | None = None,
    ):
        self.topology = topology
        self.config = config or ContextualRewardConfig()
        self._controlled_rail_id_set = frozenset(
            int(value) for value in self.topology.controlled_rail_ids
        )
        self.completion_diagnostic_path = (
            Path(completion_diagnostic_path)
            if completion_diagnostic_path is not None
            else None
        )
        self._global_step_provider = global_step_provider
        self.reward_diagnostic_writer = reward_diagnostic_writer
        self.completion_diagnostic_max_global_step = (
            int(completion_diagnostic_max_global_step)
            if completion_diagnostic_max_global_step is not None
            else None
        )
        if (
            self.completion_diagnostic_max_global_step is not None
            and self.completion_diagnostic_max_global_step < 0
        ):
            raise ValueError(
                "completion_diagnostic_max_global_step must be non-negative"
            )
        self.local_normalizer = RunningFeatureNormalizer(
            1, epsilon=self.config.normalizer_epsilon, clip=self.config.local_clip
        )
        self.global_normalizer = RunningFeatureNormalizer(
            1, epsilon=self.config.normalizer_epsilon, clip=self.config.global_clip
        )
        self.reward_steps = 0
        self._oht_cycle_trackers: dict[int, OHTCycleTracker] = {}
        self._rail_pass_trackers: dict[int, RailPassTemporalTracker] = {}
        self._reset_temporal()

    def _reset_temporal(self) -> None:
        self._total_completed_jobs = 0.0
        self._prev_op_rate: float | None = None
        self._last_global_terms: dict[str, float] = {}
        self._last_local_terms: dict[str, np.ndarray] = {}
        self._last_rail_tat_event_count = 0
        self._last_rail_tat_cycle_count = 0
        self._last_rail_tat_controlled_assignment_count = 0
        self._last_rail_tat_uncontrolled_assignment_count = 0
        self._oht_cycle_trackers.clear()
        self._rail_pass_trackers.clear()

    def reset_episode(self) -> None:
        """Clear only episode-temporal state; training statistics persist."""
        self._reset_temporal()

    def _current_global_step(self) -> int:
        return (
            int(self._global_step_provider())
            if self._global_step_provider is not None else self.reward_steps
        )

    def _cycle_diagnostic_active(self) -> bool:
        return bool(
            self.reward_diagnostic_writer is not None
            and self.reward_diagnostic_writer.window_name(
                self._current_global_step()
            ) is not None
        )

    @staticmethod
    def _job_backlog(pclient) -> tuple[float, float]:
        waiting = float(getattr(pclient, "WaitingCommandCount", 0) or 0)
        queued = float(getattr(pclient, "QueuedCommandCount", 0) or 0)
        return waiting, queued

    def _global_raw(self, pclient) -> float:
        cfg = self.config
        cur_tat = float(getattr(pclient, "TotalTat"))
        cur_op = float(getattr(pclient, "TotalOhtOperationRate"))
        completed_delta = float(
            getattr(pclient, "CompletedCommandCount", 0) or 0
        )
        if not np.isfinite((cur_tat, cur_op, completed_delta)).all():
            raise ContextualRewardError("global reward input contains NaN or Inf")
        self._total_completed_jobs += completed_delta
        completed = self._total_completed_jobs
        waiting, queued = self._job_backlog(pclient)

        # TotalTat is already the simulator's cumulative mean TAT level.
        # Never difference it to recover a marginal value: its 0.1-second
        # transport quantization is amplified by the cumulative job count.
        tat_signal_available = cur_tat > 0.0
        tat_error = (
            (cfg.tat_reference - cur_tat) / cfg.tat_reference
            if tat_signal_available else 0.0
        )
        tat_raw_preclip = (
            cfg.tat_weight * tat_error if cfg.use_tat else 0.0
        )
        tat_raw_postclip = (
            float(np.clip(
                tat_raw_preclip, -cfg.tat_raw_clip, cfg.tat_raw_clip
            ))
            if cfg.tat_raw_clip is not None else float(tat_raw_preclip)
        )
        # Reward J never gates global TAT by completed-command count. Keep the
        # compatibility diagnostic fields, but they are identity-valued.
        tat_confidence = float(tat_signal_available)
        tat_raw_ramped = tat_raw_postclip
        op_delta = (
            float(self._prev_op_rate) - cur_op
            if self._prev_op_rate is not None else 0.0
        )
        op_error = cfg.op_reference - cur_op
        op_raw = cfg.op_weight * op_error if cfg.use_op else 0.0
        backlog = waiting + queued
        backlog_raw = -cfg.backlog_weight * backlog if cfg.use_backlog else 0.0
        raw = tat_raw_ramped + op_raw + backlog_raw
        self._last_global_terms = {
            "tat_error": tat_error,
            "tat_raw_preclip": tat_raw_preclip,
            "tat_raw_postclip": tat_raw_postclip,
            "tat_confidence": tat_confidence,
            "tat_raw_ramped": tat_raw_ramped,
            "total_tat": cur_tat,
            "completed_episode": completed,
            "completed_delta": completed_delta,
            "op_rate": cur_op,
            "op_reference": cfg.op_reference,
            "op_error": op_error,
            "op_delta": op_delta,
            "op_raw": op_raw,
            "backlog": backlog,
            "backlog_raw": backlog_raw,
            "waiting": waiting,
            "queued": queued,
        }
        self._prev_op_rate = cur_op
        return float(raw)

    def _local_raw(self, pclient) -> np.ndarray:
        result = np.empty(len(self.topology.controlled_rail_ids), dtype=np.float64)
        local_values = {
            name: np.empty(len(self.topology.controlled_rail_ids), dtype=np.float64)
            for name in ("oht", "predicted", "stop", "idle", "capacity")
        }
        idle_observation = np.empty(
            len(self.topology.controlled_rail_ids), dtype=np.float64
        )
        rails = getattr(pclient, "RAILLINE_DIC", {})
        ohts = getattr(pclient, "OHT_DIC", {})
        for row, rail_id_value in enumerate(self.topology.controlled_rail_ids):
            rail_id = int(rail_id_value)
            if rail_id not in rails:
                raise ContextualRewardError(f"missing controlled rail {rail_id}")
            rail = rails[rail_id]
            rail_ohts = [
                ohts[oht_id] for oht_id in getattr(rail, "OhtList")
                if oht_id in ohts
            ]
            avg_stop = (
                float(np.mean([float(getattr(oht, "StopTime")) for oht in rail_ohts]))
                if rail_ohts else 0.0
            )
            oht_count = len(getattr(rail, "OhtList"))
            capacity = oht_count / max(1, int(getattr(rail, "PortCount")) + 1)
            result[row] = 0.0
            local_values["oht"][row] = -0.3 * oht_count
            local_values["predicted"][row] = (
                -self.config.local_predicted_oht_weight
                * float(getattr(rail, "PredictedOHTCount"))
            )
            local_values["stop"][row] = -0.3 * avg_stop
            idle_observation[row] = float(getattr(rail, "IdleOHTCount"))
            local_values["idle"][row] = 0.0
            local_values["capacity"][row] = -0.1 * capacity
            result[row] = sum(values[row] for values in local_values.values())
        if not np.isfinite(result).all():
            raise ContextualRewardError("local reward input contains NaN or Inf")
        self._last_local_terms = local_values
        self._last_idle_oht_observation = idle_observation
        return result

    def _rail_reward_raw(
        self,
        pclient,
        *,
        env_step: int,
        episode_id: int = 0,
    ) -> np.ndarray:
        credit: dict[int, float] = {}
        cycle_count = 0
        controlled_assignment_count = 0
        uncontrolled_assignment_count = 0
        diagnostic_active = self._cycle_diagnostic_active()
        free_flow_tracking_active = (
            self.config.rail_reward_mode
            in {
                RAIL_REWARD_BASELINE_RATIO,
                RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
            }
            or diagnostic_active
        )
        self._diagnostic_rails = (
            getattr(pclient, "RAILLINE_DIC", {})
            if free_flow_tracking_active else None
        )
        self._store_occurrence_details = diagnostic_active
        self._diagnostic_sim_time = (
            float(getattr(pclient, "SimTime", 0.0)) if diagnostic_active else None
        )
        for oht_id, oht in getattr(pclient, "OHT_DIC", {}).items():
            oht_id = int(oht_id)
            current_state = int(getattr(oht, "State", OHTState.NULL))
            current_job_id = int(getattr(oht, "JobID", 0) or 0)
            pass_times = list(getattr(oht, "PassTimes", None) or [])
            not_pass_times = list(getattr(oht, "NotPassTimes", None) or [])
            entries = self._oht_tat_entries(oht)
            selected, selection_reason = self._select_oht_tat_entry(oht, entries)
            current_oht_tat = self._finite_positive_attr(selected, "OHTTat")
            current_cmd_tat = self._finite_positive_attr(selected, "CmdTat")
            tracker = self._oht_cycle_trackers.get(oht_id)

            if tracker is None:
                tracker = OHTCycleTracker(
                    oht_id=oht_id,
                    previous_state=current_state,
                    current_state=current_state,
                    latest_job_id=current_job_id,
                )
                self._oht_cycle_trackers[oht_id] = tracker
                self._rail_pass_trackers[oht_id] = RailPassTemporalTracker()
                if current_state in ACTIVE_OHT_CYCLE_STATES:
                    self._start_oht_cycle(
                        tracker,
                        env_step=env_step,
                        state=current_state,
                        job_id=current_job_id,
                        bootstrapped=True,
                    )
                    self._consume_rail_snapshot(
                        oht_id,
                        pass_times,
                        not_pass_times,
                        tracker,
                        boundary=True,
                        env_step=env_step,
                        episode_id=episode_id,
                    )
                    self._update_cycle_tat(
                        tracker,
                        current_state,
                        current_oht_tat,
                        current_cmd_tat,
                        selection_reason,
                    )
                    self._write_cycle_diagnostic(
                        "oht_cycle_bootstrapped",
                        env_step=env_step,
                        episode_id=episode_id,
                        tracker=tracker,
                        previous_state=current_state,
                        current_state=current_state,
                        previous_job_id=current_job_id,
                        current_job_id=current_job_id,
                        current_oht_tat=current_oht_tat,
                        current_cmd_tat=current_cmd_tat,
                        entries=entries,
                        reward_applied=False,
                        skip_reason=selection_reason,
                    )
                else:
                    self._consume_rail_snapshot(
                        oht_id,
                        pass_times,
                        not_pass_times,
                        None,
                        boundary=False,
                        env_step=env_step,
                        episode_id=episode_id,
                    )
                continue

            previous_state = int(tracker.current_state)
            previous_job_id = int(tracker.latest_job_id)
            tracker.previous_state = previous_state
            tracker.current_state = current_state

            # A state-5 boundary owns the previous cycle. Current state-2/3
            # values must not overwrite its final TAT or ledger.
            if (
                previous_state == int(OHTState.UNLOADING)
                and current_state
                in {
                    int(OHTState.IDLE),
                    int(OHTState.MOVE_TO_LOAD),
                    int(OHTState.LOADING),
                }
            ):
                restarting = current_state in {
                    int(OHTState.MOVE_TO_LOAD),
                    int(OHTState.LOADING),
                }
                old_passes, new_passes, ambiguous = (
                    self._split_boundary_passes(
                        pass_times,
                        previous_state=previous_state,
                        current_state=current_state,
                        restarting=restarting,
                    )
                )
                tracker.ambiguous_pass_count += ambiguous
                if ambiguous:
                    self._write_cycle_diagnostic(
                        "rail_boundary_ambiguous",
                        env_step=env_step,
                        episode_id=episode_id,
                        tracker=tracker,
                        previous_state=previous_state,
                        current_state=current_state,
                        previous_job_id=previous_job_id,
                        current_job_id=current_job_id,
                        current_oht_tat=current_oht_tat,
                        current_cmd_tat=current_cmd_tat,
                        entries=entries,
                        reward_applied=False,
                        skip_reason="unattributable_boundary_pass",
                    )
                self._consume_rail_snapshot(
                    oht_id,
                    old_passes,
                    [],
                    tracker,
                    boundary=False,
                    env_step=env_step,
                    episode_id=episode_id,
                )
                outcome = self._finalize_oht_cycle(tracker, credit)
                cycle_count += 1
                if outcome.reward_applied:
                    controlled_assignment_count += outcome.controlled_reward_count
                    uncontrolled_assignment_count += outcome.uncontrolled_reward_count
                event = (
                    "oht_cycle_restarted"
                    if restarting else "oht_cycle_completed"
                )
                self._write_cycle_diagnostic(
                    event,
                    env_step=env_step,
                    episode_id=episode_id,
                    tracker=tracker,
                    previous_state=previous_state,
                    current_state=current_state,
                    previous_job_id=previous_job_id,
                    current_job_id=current_job_id,
                    current_oht_tat=current_oht_tat,
                    current_cmd_tat=current_cmd_tat,
                    entries=entries,
                    reward_applied=outcome.reward_applied,
                    skip_reason=outcome.skip_reason,
                    outcome=outcome,
                )
                if not outcome.reward_applied:
                    self._write_cycle_diagnostic(
                        "oht_cycle_skipped",
                        env_step=env_step,
                        episode_id=episode_id,
                        tracker=tracker,
                        previous_state=previous_state,
                        current_state=current_state,
                        previous_job_id=previous_job_id,
                        current_job_id=current_job_id,
                        current_oht_tat=current_oht_tat,
                        current_cmd_tat=current_cmd_tat,
                        entries=entries,
                        reward_applied=False,
                        skip_reason=outcome.skip_reason,
                        outcome=outcome,
                    )
                tracker.clear_cycle()
                tracker.previous_state = previous_state
                tracker.current_state = current_state
                tracker.latest_job_id = current_job_id
                if restarting:
                    self._start_oht_cycle(
                        tracker,
                        env_step=env_step,
                        state=current_state,
                        job_id=current_job_id,
                        bootstrapped=False,
                    )
                    self._consume_rail_snapshot(
                        oht_id,
                        new_passes,
                        not_pass_times,
                        tracker,
                        boundary=True,
                        env_step=env_step,
                        episode_id=episode_id,
                    )
                    self._update_cycle_tat(
                        tracker,
                        current_state,
                        current_oht_tat,
                        current_cmd_tat,
                        selection_reason,
                    )
                else:
                    self._consume_rail_snapshot(
                        oht_id,
                        [],
                        not_pass_times,
                        None,
                        boundary=True,
                        env_step=env_step,
                        episode_id=episode_id,
                    )
                continue

            if tracker.cycle_active:
                if current_state in ACTIVE_OHT_CYCLE_STATES:
                    self._consume_rail_snapshot(
                        oht_id,
                        pass_times,
                        not_pass_times,
                        tracker,
                        boundary=False,
                        env_step=env_step,
                        episode_id=episode_id,
                    )
                    if (
                        current_job_id != tracker.current_segment_job_id
                        and previous_state != int(OHTState.UNLOADING)
                    ):
                        old_segment_job_id = tracker.current_segment_job_id
                        self._close_tat_segment(tracker)
                        self._write_cycle_diagnostic(
                            "oht_tat_segment_closed",
                            env_step=env_step,
                            episode_id=episode_id,
                            tracker=tracker,
                            previous_state=previous_state,
                            current_state=current_state,
                            previous_job_id=old_segment_job_id,
                            current_job_id=current_job_id,
                            current_oht_tat=current_oht_tat,
                            current_cmd_tat=current_cmd_tat,
                            entries=entries,
                            reward_applied=False,
                            skip_reason=tracker.last_tat_skip_reason,
                        )
                        self._start_tat_segment(
                            tracker, current_job_id, increment=True
                        )
                        self._write_cycle_diagnostic(
                            "oht_tat_segment_started",
                            env_step=env_step,
                            episode_id=episode_id,
                            tracker=tracker,
                            previous_state=previous_state,
                            current_state=current_state,
                            previous_job_id=old_segment_job_id,
                            current_job_id=current_job_id,
                            current_oht_tat=current_oht_tat,
                            current_cmd_tat=current_cmd_tat,
                            entries=entries,
                            reward_applied=False,
                            skip_reason=selection_reason,
                        )
                    unexplained_reset = self._update_cycle_tat(
                        tracker,
                        current_state,
                        current_oht_tat,
                        current_cmd_tat,
                        selection_reason,
                    )
                    if unexplained_reset:
                        self._write_cycle_diagnostic(
                            "oht_tat_reset_unexplained",
                            env_step=env_step,
                            episode_id=episode_id,
                            tracker=tracker,
                            previous_state=previous_state,
                            current_state=current_state,
                            previous_job_id=previous_job_id,
                            current_job_id=current_job_id,
                            current_oht_tat=current_oht_tat,
                            current_cmd_tat=current_cmd_tat,
                            entries=entries,
                            reward_applied=False,
                            skip_reason="unexplained_oht_tat_reset",
                        )
                    if current_job_id != previous_job_id:
                        if previous_job_id:
                            tracker.previous_job_ids.add(previous_job_id)
                        tracker.latest_job_id = current_job_id
                        self._write_cycle_diagnostic(
                            "oht_job_changed",
                            env_step=env_step,
                            episode_id=episode_id,
                            tracker=tracker,
                            previous_state=previous_state,
                            current_state=current_state,
                            previous_job_id=previous_job_id,
                            current_job_id=current_job_id,
                            current_oht_tat=current_oht_tat,
                            current_cmd_tat=current_cmd_tat,
                            entries=entries,
                            reward_applied=False,
                            skip_reason=selection_reason,
                        )
                    if (
                        previous_state != int(OHTState.UNLOADING)
                        and current_state == int(OHTState.UNLOADING)
                    ):
                        self._write_cycle_diagnostic(
                            "oht_unloading",
                            env_step=env_step,
                            episode_id=episode_id,
                            tracker=tracker,
                            previous_state=previous_state,
                            current_state=current_state,
                            previous_job_id=previous_job_id,
                            current_job_id=current_job_id,
                            current_oht_tat=current_oht_tat,
                            current_cmd_tat=current_cmd_tat,
                            entries=entries,
                            reward_applied=False,
                            skip_reason=selection_reason,
                        )
                else:
                    self._write_cycle_diagnostic(
                        "oht_cycle_aborted",
                        env_step=env_step,
                        episode_id=episode_id,
                        tracker=tracker,
                        previous_state=previous_state,
                        current_state=current_state,
                        previous_job_id=previous_job_id,
                        current_job_id=current_job_id,
                        current_oht_tat=current_oht_tat,
                        current_cmd_tat=current_cmd_tat,
                        entries=entries,
                        reward_applied=False,
                        skip_reason="active_cycle_left_without_state_5_completion",
                    )
                    tracker.clear_cycle()
                    tracker.latest_job_id = current_job_id
                    self._consume_rail_snapshot(
                        oht_id,
                        pass_times,
                        not_pass_times,
                        None,
                        boundary=True,
                        env_step=env_step,
                        episode_id=episode_id,
                    )
            elif (
                previous_state == int(OHTState.IDLE)
                and current_state == int(OHTState.MOVE_TO_LOAD)
            ):
                self._start_oht_cycle(
                    tracker,
                    env_step=env_step,
                    state=current_state,
                    job_id=current_job_id,
                    bootstrapped=False,
                )
                _, start_passes, ambiguous = self._split_boundary_passes(
                    pass_times,
                    previous_state=previous_state,
                    current_state=current_state,
                    restarting=True,
                )
                tracker.ambiguous_pass_count += ambiguous
                if ambiguous:
                    self._write_cycle_diagnostic(
                        "rail_boundary_ambiguous",
                        env_step=env_step,
                        episode_id=episode_id,
                        tracker=tracker,
                        previous_state=previous_state,
                        current_state=current_state,
                        previous_job_id=previous_job_id,
                        current_job_id=current_job_id,
                        current_oht_tat=current_oht_tat,
                        current_cmd_tat=current_cmd_tat,
                        entries=entries,
                        reward_applied=False,
                        skip_reason="unattributable_cycle_start_pass",
                    )
                self._consume_rail_snapshot(
                    oht_id,
                    start_passes,
                    not_pass_times,
                    tracker,
                    boundary=True,
                    env_step=env_step,
                    episode_id=episode_id,
                )
                self._update_cycle_tat(
                    tracker,
                    current_state,
                    current_oht_tat,
                    current_cmd_tat,
                    selection_reason,
                )
                self._write_cycle_diagnostic(
                    "oht_cycle_start",
                    env_step=env_step,
                    episode_id=episode_id,
                    tracker=tracker,
                    previous_state=previous_state,
                    current_state=current_state,
                    previous_job_id=previous_job_id,
                    current_job_id=current_job_id,
                    current_oht_tat=current_oht_tat,
                    current_cmd_tat=current_cmd_tat,
                    entries=entries,
                    reward_applied=False,
                    skip_reason=selection_reason,
                )
            elif current_state in ACTIVE_OHT_CYCLE_STATES:
                self._start_oht_cycle(
                    tracker,
                    env_step=env_step,
                    state=current_state,
                    job_id=current_job_id,
                    bootstrapped=True,
                )
                self._consume_rail_snapshot(
                    oht_id,
                    pass_times,
                    not_pass_times,
                    tracker,
                    boundary=True,
                    env_step=env_step,
                    episode_id=episode_id,
                )
                self._update_cycle_tat(
                    tracker,
                    current_state,
                    current_oht_tat,
                    current_cmd_tat,
                    selection_reason,
                )
                self._write_cycle_diagnostic(
                    "oht_cycle_bootstrapped",
                    env_step=env_step,
                    episode_id=episode_id,
                    tracker=tracker,
                    previous_state=previous_state,
                    current_state=current_state,
                    previous_job_id=previous_job_id,
                    current_job_id=current_job_id,
                    current_oht_tat=current_oht_tat,
                    current_cmd_tat=current_cmd_tat,
                    entries=entries,
                    reward_applied=False,
                    skip_reason="active_without_idle_to_move_to_load",
                )
            else:
                self._consume_rail_snapshot(
                    oht_id,
                    pass_times,
                    not_pass_times,
                    None,
                    boundary=False,
                    env_step=env_step,
                    episode_id=episode_id,
                )

            tracker.latest_job_id = current_job_id
        result = np.asarray(
            [credit.get(int(rail_id), 0.0)
             for rail_id in self.topology.controlled_rail_ids],
            dtype=np.float64,
        )
        self._last_rail_tat_cycle_count = int(cycle_count)
        self._last_rail_tat_controlled_assignment_count = int(
            controlled_assignment_count
        )
        self._last_rail_tat_uncontrolled_assignment_count = int(
            uncontrolled_assignment_count
        )
        # Compatibility alias: historically this counted controlled rail
        # assignments, not completed OHT cycles.
        self._last_rail_tat_event_count = int(controlled_assignment_count)
        self._diagnostic_rails = None
        self._diagnostic_sim_time = None
        self._store_occurrence_details = False
        return result

    def _rail_tat(self, pclient, *, env_step: int, episode_id: int = 0):
        """Deprecated compatibility alias returning unweighted raw penalty."""
        return -self._rail_reward_raw(
            pclient, env_step=env_step, episode_id=episode_id
        )

    def _rail_tat_raw(self, pclient, *, env_step: int, episode_id: int = 0):
        """Deprecated compatibility alias returning penalty direction."""
        return self._rail_tat(
            pclient, env_step=env_step, episode_id=episode_id
        )

    def _scale_rail_reward(self, rail_reward_raw):
        raw = np.asarray(rail_reward_raw, dtype=np.float64)
        weighted_preclip = self.config.rail_tat_weight * raw
        postclip = (
            np.clip(
                weighted_preclip,
                -self.config.rail_tat_clip,
                self.config.rail_tat_clip,
            )
            if self.config.rail_tat_clip is not None else weighted_preclip
        )
        return weighted_preclip, postclip

    def _scale_rail_tat(self, rail_tat_raw):
        """Deprecated penalty-direction scaling alias."""
        weighted, postclip = self._scale_rail_reward(-np.asarray(rail_tat_raw))
        return -weighted, -postclip

    @staticmethod
    def _oht_tat_entries(oht) -> dict[int, object]:
        values = getattr(oht, "CmdCompleteTat", None)
        if hasattr(values, "items"):
            return {int(key): value for key, value in values.items()}
        return {
            int(getattr(value, "CmdID", 0) or 0): value
            for value in (values or [])
        }

    @staticmethod
    def _select_oht_tat_entry(oht, entries):
        if not entries:
            return None, "missing_tat_entry"
        candidates = {
            int(candidate)
            for candidate in (
                getattr(oht, "JobID", 0),
                getattr(oht, "DispatchedCommand", 0),
            )
            if int(candidate or 0) in entries
        }
        if len(candidates) == 1:
            return entries[next(iter(candidates))], None
        if len(candidates) > 1:
            return None, "job_and_dispatched_command_conflict"
        if len(entries) == 1:
            return next(iter(entries.values())), None
        return None, "ambiguous_tat_entries"

    @staticmethod
    def _finite_positive_attr(value, name) -> float | None:
        if value is None:
            return None
        result = float(getattr(value, name, 0) or 0)
        return result if np.isfinite(result) and result > 0 else None

    def _start_oht_cycle(
        self,
        tracker,
        *,
        env_step,
        state,
        job_id,
        bootstrapped,
    ) -> None:
        tracker.clear_cycle()
        tracker.cycle_active = True
        tracker.bootstrapped = bool(bootstrapped)
        tracker.cycle_start_step = int(env_step)
        tracker.cycle_start_global_step = self._current_global_step()
        tracker.current_state = int(state)
        tracker.latest_job_id = int(job_id)
        tracker.previous_job_ids.clear()
        ContextualRewardBuilder._start_tat_segment(
            tracker, int(job_id), increment=True
        )

    @staticmethod
    def _start_tat_segment(tracker, job_id, *, increment) -> None:
        tracker.current_segment_job_id = int(job_id)
        tracker.current_segment_last_valid_oht_tat = None
        tracker.current_segment_last_state5_oht_tat = None
        if increment:
            tracker.tat_segment_count += 1

    @staticmethod
    def _close_tat_segment(tracker) -> None:
        value = tracker.current_segment_last_valid_oht_tat
        if value is None:
            tracker.cycle_tat_valid = False
            tracker.last_tat_skip_reason = "missing_closed_segment_oht_tat"
            return
        tracker.closed_segment_oht_tat_sum += float(value)

    @staticmethod
    def _split_boundary_passes(
        pass_times,
        *,
        previous_state,
        current_state,
        restarting,
    ):
        if not restarting:
            old = [
                item for item in pass_times
                if int(getattr(item, "State")) in ACTIVE_OHT_CYCLE_STATES
            ]
            ambiguous = len(pass_times) - len(old)
            return old, [], ambiguous
        new_states = {int(current_state)}
        if int(current_state) == int(OHTState.LOADING):
            new_states.add(int(OHTState.MOVE_TO_LOAD))
        first_new = next(
            (
                index for index, item in enumerate(pass_times)
                if int(getattr(item, "State")) in new_states
            ),
            None,
        )
        if first_new is None:
            old_states = (
                {int(OHTState.MOVE_TO_UNLOAD), int(OHTState.UNLOADING)}
                if int(previous_state) == int(OHTState.UNLOADING)
                else set()
            )
            old = [
                item for item in pass_times
                if int(getattr(item, "State")) in old_states
            ]
            return old, [], len(pass_times) - len(old)
        old_candidates = pass_times[:first_new]
        new_candidates = pass_times[first_new:]
        old_states = (
            {int(OHTState.MOVE_TO_UNLOAD), int(OHTState.UNLOADING)}
            if int(previous_state) == int(OHTState.UNLOADING)
            else set()
        )
        old = [
            item for item in old_candidates
            if int(getattr(item, "State")) in old_states
        ]
        new = [
            item for item in new_candidates
            if int(getattr(item, "State")) in new_states
        ]
        ambiguous = (
            len(old_candidates) - len(old)
            + len(new_candidates) - len(new)
        )
        return old, new, ambiguous

    def _consume_rail_snapshot(
        self,
        oht_id,
        completed,
        current_inflight,
        ledger,
        *,
        boundary,
        env_step,
        episode_id,
    ) -> None:
        temporal = self._rail_pass_trackers.setdefault(
            int(oht_id), RailPassTemporalTracker()
        )
        previous_id = temporal.inflight_rail_id
        previous_elapsed = temporal.inflight_elapsed
        previous_attributed = temporal.inflight_attributed_elapsed
        previous_consumed = False

        def invalid_delta(item, reason):
            if ledger is not None:
                ledger.invalid_pass_delta_count += 1
                self._write_cycle_diagnostic(
                    "rail_pass_invalid_delta",
                    env_step=env_step,
                    episode_id=episode_id,
                    tracker=ledger,
                    previous_state=ledger.previous_state
                    if ledger.previous_state is not None else ledger.current_state,
                    current_state=ledger.current_state,
                    previous_job_id=ledger.latest_job_id,
                    current_job_id=ledger.latest_job_id,
                    current_oht_tat=None,
                    current_cmd_tat=None,
                    entries={},
                    reward_applied=False,
                    skip_reason=reason,
                )

        def add_delta(item, delta):
            if not np.isfinite(delta) or delta < 0:
                invalid_delta(item, "negative_or_nonfinite_rail_delta")
                return
            if delta == 0 or ledger is None:
                return
            state = int(getattr(item, "State"))
            if state not in ACTIVE_OHT_CYCLE_STATES:
                return
            rail_id = int(getattr(item, "ID"))
            ledger.rail_time_by_id[rail_id] = (
                ledger.rail_time_by_id.get(rail_id, 0.0) + float(delta)
            )
            ledger.route_count += 1
            ledger.route_time += float(delta)
            ledger.pass_delta_count += 1

        def record_completed_occurrence(item, actual_elapsed):
            if ledger is None or self._diagnostic_rails is None:
                return
            rail_id = int(getattr(item, "ID"))
            rail = self._diagnostic_rails.get(rail_id)
            free_flow = (
                getattr(rail, "DistancePerVelocity", None)
                if rail is not None else None
            )
            if free_flow is None:
                ledger.free_flow_missing_count += 1
                free_flow_value = None
            else:
                free_flow_value = float(free_flow)
                if not np.isfinite(free_flow_value) or free_flow_value <= 0:
                    ledger.free_flow_invalid_count += 1
                    free_flow_value = None
                else:
                    ledger.route_free_flow_time += free_flow_value
            controlled = rail_id in self._controlled_rail_id_set
            if self._store_occurrence_details:
                ledger.rail_occurrences.append(RailOccurrenceDiagnostic(
                    occurrence_index=len(ledger.rail_occurrences),
                    rail_id=rail_id,
                    state=int(getattr(item, "State")),
                    actual_elapsed=float(actual_elapsed),
                    distance_per_velocity=free_flow_value,
                    controlled=controlled,
                ))
            ledger.completed_rail_occurrence_count += 1

        for index, item in enumerate(completed):
            rail_id = int(getattr(item, "ID"))
            total = float(getattr(item, "PassTime"))
            if not np.isfinite(total) or total < 0:
                invalid_delta(item, "invalid_completed_pass_time")
                continue
            if index == 0 and previous_id is not None:
                if rail_id == previous_id and previous_elapsed is not None:
                    delta = total - previous_elapsed
                    previous_consumed = True
                    if ledger is not None and boundary:
                        ledger.boundary_time_subtracted += float(previous_elapsed)
                else:
                    if ledger is not None:
                        ledger.ambiguous_pass_count += 1
                        self._write_cycle_diagnostic(
                            "rail_boundary_ambiguous",
                            env_step=env_step,
                            episode_id=episode_id,
                            tracker=ledger,
                            previous_state=ledger.previous_state
                            if ledger.previous_state is not None
                            else ledger.current_state,
                            current_state=ledger.current_state,
                            previous_job_id=ledger.latest_job_id,
                            current_job_id=ledger.latest_job_id,
                            current_oht_tat=None,
                            current_cmd_tat=None,
                            entries={},
                            reward_applied=False,
                            skip_reason="completed_pass_did_not_match_inflight",
                        )
                    continue
            else:
                delta = total
            add_delta(item, delta)
            occurrence_elapsed = (
                previous_attributed + delta
                if index == 0 and previous_consumed else delta
            )
            if (
                np.isfinite(delta)
                and delta >= 0
                and int(getattr(item, "State")) in ACTIVE_OHT_CYCLE_STATES
            ):
                record_completed_occurrence(item, occurrence_elapsed)

        inflight_items = list(current_inflight or [])
        if len(inflight_items) > 1:
            if ledger is not None:
                ledger.ambiguous_pass_count += len(inflight_items) - 1
            inflight_items = inflight_items[:1]
        if not inflight_items:
            if completed or previous_consumed or boundary:
                temporal.clear()
            return

        item = inflight_items[0]
        rail_id = int(getattr(item, "ID"))
        elapsed = float(getattr(item, "PassTime"))
        if not np.isfinite(elapsed) or elapsed < 0:
            invalid_delta(item, "invalid_inflight_pass_time")
            temporal.clear()
            return
        same_occurrence = (
            not previous_consumed
            and previous_id == rail_id
            and previous_elapsed is not None
        )
        if same_occurrence:
            delta = elapsed - previous_elapsed
            if delta < 0:
                invalid_delta(item, "nonmonotonic_inflight_pass_time")
                return
            if ledger is not None and boundary:
                ledger.boundary_carry_count += 1
                ledger.boundary_carry_time += float(previous_elapsed)
                ledger.boundary_time_subtracted += float(previous_elapsed)
                self._write_cycle_diagnostic(
                    "rail_boundary_baseline",
                    env_step=env_step,
                    episode_id=episode_id,
                    tracker=ledger,
                    previous_state=ledger.previous_state
                    if ledger.previous_state is not None else ledger.current_state,
                    current_state=ledger.current_state,
                    previous_job_id=ledger.latest_job_id,
                    current_job_id=ledger.latest_job_id,
                    current_oht_tat=None,
                    current_cmd_tat=None,
                    entries={},
                    reward_applied=False,
                    skip_reason=None,
                )
        elif boundary:
            delta = 0.0
            if ledger is not None:
                ledger.boundary_carry_count += 1
                ledger.boundary_carry_time += elapsed
                ledger.boundary_time_subtracted += elapsed
                self._write_cycle_diagnostic(
                    "rail_boundary_baseline",
                    env_step=env_step,
                    episode_id=episode_id,
                    tracker=ledger,
                    previous_state=ledger.previous_state
                    if ledger.previous_state is not None else ledger.current_state,
                    current_state=ledger.current_state,
                    previous_job_id=ledger.latest_job_id,
                    current_job_id=ledger.latest_job_id,
                    current_oht_tat=None,
                    current_cmd_tat=None,
                    entries={},
                    reward_applied=False,
                    skip_reason="new_inflight_occurrence_at_boundary",
                )
        else:
            delta = elapsed
        add_delta(item, delta)
        temporal.inflight_rail_id = rail_id
        temporal.inflight_state = int(getattr(item, "State"))
        temporal.inflight_elapsed = elapsed
        temporal.inflight_attributed_elapsed = (
            previous_attributed + delta if same_occurrence else delta
        )

    @staticmethod
    def _update_cycle_tat(
        tracker,
        state,
        oht_tat,
        cmd_tat,
        selection_reason,
    ) -> bool:
        unexplained_reset = False
        if oht_tat is not None:
            previous = tracker.current_segment_last_valid_oht_tat
            if previous is not None and float(oht_tat) + 1e-9 < previous:
                tracker.cycle_tat_valid = False
                tracker.last_tat_skip_reason = "unexplained_oht_tat_reset"
                unexplained_reset = True
            tracker.current_segment_last_valid_oht_tat = float(oht_tat)
            if not unexplained_reset:
                tracker.last_tat_skip_reason = None
            if int(state) == int(OHTState.UNLOADING):
                tracker.current_segment_last_state5_oht_tat = float(oht_tat)
        elif selection_reason is not None:
            tracker.last_tat_skip_reason = str(selection_reason)
        if cmd_tat is not None:
            tracker.last_valid_cmd_tat = float(cmd_tat)
        return unexplained_reset

    def _finalize_oht_cycle(self, tracker, credit) -> CycleRewardOutcome:
        if tracker.bootstrapped:
            return CycleRewardOutcome(False, "bootstrapped_incomplete_cycle")
        free_flow_mode = self.config.rail_reward_mode in {
            RAIL_REWARD_BASELINE_RATIO,
            RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
        }
        if (
            not tracker.rail_time_by_id
            or not np.isfinite(tracker.route_time)
            or tracker.route_time <= 0
        ):
            return CycleRewardOutcome(
                False,
                "invalid_free_flow_cycle"
                if free_flow_mode else "missing_valid_cycle_route",
            )

        effective_oht_tat = None
        signed_excess = None
        if (
            tracker.cycle_tat_valid
            and tracker.current_segment_last_state5_oht_tat is not None
        ):
            effective_oht_tat = float(
                tracker.closed_segment_oht_tat_sum
                + tracker.current_segment_last_state5_oht_tat
            )
            signed_excess = float(
                (effective_oht_tat - self.config.tat_reference)
                / self.config.tat_reference
            )

        route_free_flow_ratio = None
        if free_flow_mode:
            if (
                not np.isfinite(tracker.route_free_flow_time)
                or tracker.route_free_flow_time <= 0
                or tracker.completed_rail_occurrence_count <= 0
                or tracker.free_flow_missing_count > 0
                or tracker.free_flow_invalid_count > 0
            ):
                return CycleRewardOutcome(
                    False,
                    "invalid_free_flow_cycle",
                    effective_oht_tat=effective_oht_tat,
                    signed_excess=signed_excess,
                )
            route_free_flow_ratio = float(
                tracker.route_time / tracker.route_free_flow_time
            )
            if not np.isfinite(route_free_flow_ratio):
                return CycleRewardOutcome(
                    False,
                    "invalid_free_flow_cycle",
                    effective_oht_tat=effective_oht_tat,
                    signed_excess=signed_excess,
                )
            if (
                self.config.rail_reward_mode
                == RAIL_REWARD_FREE_FLOW_NEUTRAL_2
            ):
                cycle_rail_reward_raw = float(
                    self.config.rail_free_flow_neutral_ratio
                    - route_free_flow_ratio
                )
            else:
                cycle_rail_reward_raw = float(
                    (
                        self.config.rail_baseline_ratio_reference
                        - route_free_flow_ratio
                    )
                    / self.config.rail_baseline_ratio_reference
                )
        else:
            if not tracker.cycle_tat_valid:
                return CycleRewardOutcome(
                    False,
                    tracker.last_tat_skip_reason or "invalid_cycle_tat",
                )
            if effective_oht_tat is None:
                return CycleRewardOutcome(False, (
                    tracker.last_tat_skip_reason
                    or "missing_final_state_5_oht_tat"
                ))
            cycle_rail_reward_raw = -float(signed_excess)

        actual_controlled_rewards = []
        for rail_id, elapsed in tracker.rail_time_by_id.items():
            reward = cycle_rail_reward_raw * elapsed / tracker.route_time
            credit[rail_id] = credit.get(rail_id, 0.0) + reward
            if rail_id in self._controlled_rail_id_set:
                weighted = reward * self.config.rail_tat_weight
                postclip = (
                    np.clip(
                        weighted,
                        -self.config.rail_tat_clip,
                        self.config.rail_tat_clip,
                    )
                    if self.config.rail_tat_clip is not None else weighted
                )
                actual_controlled_rewards.append(float(postclip))
        controlled_count = sum(
            rail_id in self._controlled_rail_id_set
            for rail_id in tracker.rail_time_by_id
        )
        values = np.asarray(actual_controlled_rewards, dtype=np.float64)
        weighted_cycle_reward = (
            cycle_rail_reward_raw * self.config.rail_tat_weight
        )
        clipped_cycle_reward = (
            float(np.clip(
                weighted_cycle_reward,
                -self.config.rail_tat_clip,
                self.config.rail_tat_clip,
            ))
            if self.config.rail_tat_clip is not None
            else float(weighted_cycle_reward)
        )
        return CycleRewardOutcome(
            True,
            None,
            effective_oht_tat=effective_oht_tat,
            signed_excess=signed_excess,
            cycle_rail_reward_raw=cycle_rail_reward_raw,
            route_free_flow_ratio=route_free_flow_ratio,
            cycle_reward_unclipped=float(weighted_cycle_reward),
            cycle_reward_clipped=float(clipped_cycle_reward),
            reward_rail_count=len(tracker.rail_time_by_id),
            controlled_reward_count=int(controlled_count),
            uncontrolled_reward_count=(
                len(tracker.rail_time_by_id) - int(controlled_count)
            ),
            reward_sum=float(values.sum()) if values.size else 0.0,
            reward_min=float(values.min()) if values.size else 0.0,
            reward_max=float(values.max()) if values.size else 0.0,
        )

    def _write_cycle_diagnostic(
        self,
        event,
        *,
        env_step,
        episode_id,
        tracker,
        previous_state,
        current_state,
        previous_job_id,
        current_job_id,
        current_oht_tat,
        current_cmd_tat,
        entries,
        reward_applied,
        skip_reason,
        outcome=None,
    ) -> None:
        outcome = outcome or CycleRewardOutcome(
            bool(reward_applied), skip_reason
        )
        if event in {"oht_cycle_completed", "oht_cycle_restarted"}:
            self._write_reward_cycle_diagnostic(
                env_step=env_step,
                episode_id=episode_id,
                tracker=tracker,
                previous_state=previous_state,
                current_state=current_state,
                previous_job_id=previous_job_id,
                current_job_id=current_job_id,
                outcome=outcome,
            )
        if self.completion_diagnostic_path is None:
            return
        global_step = (
            int(self._global_step_provider())
            if self._global_step_provider is not None
            else None
        )
        if (
            global_step is not None
            and self.completion_diagnostic_max_global_step is not None
            and global_step >= self.completion_diagnostic_max_global_step
        ):
            return
        effective_oht_tat = outcome.effective_oht_tat
        cycle_elapsed = (
            int(env_step) - int(tracker.cycle_start_step)
            if tracker.cycle_start_step is not None else None
        )
        record = {
            "event": str(event),
            "global_step": global_step,
            "env_step": int(env_step),
            "episode_step": int(env_step),
            "episode_id": int(episode_id),
            "oht_id": int(tracker.oht_id),
            "previous_state": int(previous_state),
            "current_state": int(current_state),
            "previous_job_id": int(previous_job_id),
            "current_job_id": int(current_job_id),
            "last_oht_tat": tracker.last_unloading_oht_tat
            if tracker.last_unloading_oht_tat is not None
            else tracker.last_valid_oht_tat,
            "current_oht_tat": current_oht_tat,
            "cmd_tat": current_cmd_tat,
            "tat_entry_ids": sorted(entries),
            "tat_segment_count": int(tracker.tat_segment_count),
            "closed_segment_oht_tat_sum": float(
                tracker.closed_segment_oht_tat_sum
            ),
            "final_segment_state5_oht_tat": (
                tracker.current_segment_last_state5_oht_tat
            ),
            "effective_cycle_oht_tat": effective_oht_tat,
            "reward_oht_tat_used": effective_oht_tat
            if outcome.reward_applied else None,
            "cycle_elapsed_time": cycle_elapsed,
            "oht_tat_elapsed_error": (
                float(effective_oht_tat - cycle_elapsed)
                if effective_oht_tat is not None
                and cycle_elapsed is not None else None
            ),
            "tat_reference": float(self.config.tat_reference),
            "signed_excess": outcome.signed_excess,
            "cycle_reward_unclipped": outcome.cycle_reward_unclipped,
            "cycle_reward_clipped": outcome.cycle_reward_clipped,
            "route_count": int(tracker.route_count),
            "route_time": float(tracker.route_time),
            "reward_rail_count": int(outcome.reward_rail_count),
            "controlled_reward_count": int(
                outcome.controlled_reward_count
            ),
            "uncontrolled_reward_count": int(
                outcome.uncontrolled_reward_count
            ),
            "reward_sum": float(outcome.reward_sum),
            "reward_min": float(outcome.reward_min),
            "reward_max": float(outcome.reward_max),
            "pass_delta_count": int(tracker.pass_delta_count),
            "boundary_carry_count": int(tracker.boundary_carry_count),
            "boundary_carry_time": float(tracker.boundary_carry_time),
            "boundary_time_subtracted": float(
                tracker.boundary_time_subtracted
            ),
            "ambiguous_pass_count": int(tracker.ambiguous_pass_count),
            "invalid_pass_delta_count": int(
                tracker.invalid_pass_delta_count
            ),
            "bootstrapped": bool(tracker.bootstrapped),
            "reward_applied": bool(reward_applied),
            "skip_reason": skip_reason,
        }
        self.completion_diagnostic_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        with self.completion_diagnostic_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(
                record,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()

    def _write_reward_cycle_diagnostic(
        self,
        *,
        env_step,
        episode_id,
        tracker,
        previous_state,
        current_state,
        previous_job_id,
        current_job_id,
        outcome,
    ) -> None:
        writer = self.reward_diagnostic_writer
        if writer is None:
            return
        global_step = self._current_global_step()
        window_bounds = writer.window_bounds(global_step)
        if window_bounds is None:
            return
        window_start, _, _ = window_bounds
        signed_excess = outcome.signed_excess
        cycle_rail_reward_raw = outcome.cycle_rail_reward_raw
        attributions = []
        if (
            cycle_rail_reward_raw is not None
            and np.isfinite(tracker.route_time)
            and tracker.route_time > 0
        ):
            for rail_id, elapsed in tracker.rail_time_by_id.items():
                raw = float(
                    cycle_rail_reward_raw * elapsed / tracker.route_time
                )
                weighted = float(self.config.rail_tat_weight * raw)
                postclip = (
                    float(np.clip(
                        weighted,
                        -self.config.rail_tat_clip,
                        self.config.rail_tat_clip,
                    ))
                    if self.config.rail_tat_clip is not None else weighted
                )
                attributions.append({
                    "rail_id": int(rail_id),
                    "actual_elapsed": float(elapsed),
                    "elapsed_share": float(elapsed / tracker.route_time),
                    "raw_reward": raw,
                    "weighted_preclip": weighted,
                    "postclip": postclip,
                    "clip_applied": not np.isclose(weighted, postclip),
                    "controlled": int(rail_id) in self._controlled_rail_id_set,
                })
        elapsed_share_sum = sum(item["elapsed_share"] for item in attributions)
        all_raw = sum(item["raw_reward"] for item in attributions)
        controlled_raw = sum(
            item["raw_reward"] for item in attributions if item["controlled"]
        )
        uncontrolled_raw = all_raw - controlled_raw
        preclip_abs_sum = sum(
            abs(item["weighted_preclip"]) for item in attributions
        )
        postclip_abs_sum = sum(abs(item["postclip"]) for item in attributions)
        clip_removed_abs_sum = preclip_abs_sum - postclip_abs_sum
        clipped_count = sum(item["clip_applied"] for item in attributions)
        effective = outcome.effective_oht_tat
        free_flow_available = bool(
            tracker.completed_rail_occurrence_count > 0
            and tracker.free_flow_missing_count == 0
            and tracker.free_flow_invalid_count == 0
            and tracker.route_free_flow_time > 0
        )
        route_free_flow = (
            float(tracker.route_free_flow_time)
            if free_flow_available else None
        )
        route_free_flow_ratio = (
            float(tracker.route_time / route_free_flow)
            if route_free_flow is not None else None
        )
        occurrence_ids = [item.rail_id for item in tracker.rail_occurrences]
        weighted_cycle_reward = (
            self.config.rail_tat_weight * cycle_rail_reward_raw
            if cycle_rail_reward_raw is not None else None
        )
        postclip_cycle_reward = (
            float(np.clip(
                weighted_cycle_reward,
                -self.config.rail_tat_clip,
                self.config.rail_tat_clip,
            ))
            if weighted_cycle_reward is not None
            and self.config.rail_tat_clip is not None
            else weighted_cycle_reward
        )
        tolerance = 1e-12
        record = {
            "reward_version": REWARD_VERSION,
            "global_step": global_step,
            "episode_id": int(episode_id),
            "episode_step": int(env_step),
            "sim_time": self._diagnostic_sim_time,
            "oht_id": int(tracker.oht_id),
            "previous_job_id": int(previous_job_id),
            "current_job_id": int(current_job_id),
            "previous_job_ids": sorted(int(v) for v in tracker.previous_job_ids),
            "cycle_start_step": tracker.cycle_start_step,
            "cycle_start_global_step": tracker.cycle_start_global_step,
            "cycle_end_step": int(env_step),
            "cycle_elapsed_steps": (
                int(env_step) - int(tracker.cycle_start_step)
                if tracker.cycle_start_step is not None else None
            ),
            "previous_state": int(previous_state),
            "current_state": int(current_state),
            "bootstrapped": bool(tracker.bootstrapped),
            "cycle_fully_observed": bool(
                not tracker.bootstrapped
                and tracker.cycle_start_global_step is not None
                and tracker.cycle_start_global_step >= window_start
            ),
            "reward_applied": bool(outcome.reward_applied),
            "skip_reason": outcome.skip_reason,
            "tat_segment_count": int(tracker.tat_segment_count),
            "effective_oht_tat": effective,
            "closed_segment_oht_tat_sum": float(
                tracker.closed_segment_oht_tat_sum
            ),
            "final_segment_state5_oht_tat": (
                tracker.current_segment_last_state5_oht_tat
            ),
            "last_valid_cmd_tat": tracker.last_valid_cmd_tat,
            "tat_reference": float(self.config.tat_reference),
            "signed_excess": signed_excess,
            "rail_reward_mode": self.config.rail_reward_mode,
            "rail_free_flow_neutral_ratio": (
                self.config.rail_free_flow_neutral_ratio
            ),
            "rail_tat_weight": self.config.rail_tat_weight,
            "rail_tat_clip": self.config.rail_tat_clip,
            "route_ratio": route_free_flow_ratio,
            "route_free_flow_ratio": route_free_flow_ratio,
            "cycle_rail_reward_raw": cycle_rail_reward_raw,
            "cycle_rail_reward_weighted_preclip": weighted_cycle_reward,
            "cycle_rail_reward_postclip": postclip_cycle_reward,
            "cycle_is_positive_reward": bool(
                cycle_rail_reward_raw is not None
                and cycle_rail_reward_raw > tolerance
            ),
            "cycle_is_negative_reward": bool(
                cycle_rail_reward_raw is not None
                and cycle_rail_reward_raw < -tolerance
            ),
            "cycle_is_zero_reward": bool(
                cycle_rail_reward_raw is not None
                and abs(cycle_rail_reward_raw) <= tolerance
            ),
            "route_time": float(tracker.route_time),
            "route_count": int(tracker.route_count),
            "pass_delta_count": int(tracker.pass_delta_count),
            "rail_unique_count": len(tracker.rail_time_by_id),
            "boundary_carry_count": int(tracker.boundary_carry_count),
            "boundary_carry_time": float(tracker.boundary_carry_time),
            "boundary_time_subtracted": float(
                tracker.boundary_time_subtracted
            ),
            "ambiguous_pass_count": int(tracker.ambiguous_pass_count),
            "invalid_pass_delta_count": int(tracker.invalid_pass_delta_count),
            "route_free_flow_available": free_flow_available,
            "route_free_flow_time": route_free_flow,
            "completed_rail_occurrence_count": int(
                tracker.completed_rail_occurrence_count
            ),
            "path_occurrence_count": int(
                tracker.completed_rail_occurrence_count
            ),
            "free_flow_missing_count": int(tracker.free_flow_missing_count),
            "free_flow_invalid_count": int(tracker.free_flow_invalid_count),
            "route_delay_time": (
                float(tracker.route_time - route_free_flow)
                if route_free_flow is not None else None
            ),
            "route_delay_ratio": (
                float(route_free_flow_ratio - 1.0)
                if route_free_flow_ratio is not None else None
            ),
            "effective_to_freeflow_ratio": (
                float(effective / route_free_flow)
                if effective is not None and route_free_flow is not None else None
            ),
            "service_residual_time": (
                float(effective - tracker.route_time)
                if effective is not None else None
            ),
            "same_rail_revisit_count": (
                len(occurrence_ids) - len(set(occurrence_ids))
            ),
            "rail_occurrences": [
                {
                    "occurrence_index": item.occurrence_index,
                    "rail_id": item.rail_id,
                    "state": item.state,
                    "actual_elapsed": item.actual_elapsed,
                    "distance_per_velocity": item.distance_per_velocity,
                    "controlled": item.controlled,
                }
                for item in tracker.rail_occurrences
            ],
            "rail_attributions": attributions,
            "elapsed_share_sum": elapsed_share_sum if attributions else None,
            "elapsed_share_sum_error": (
                abs(1.0 - elapsed_share_sum) if attributions else None
            ),
            "all_rail_raw_attribution_sum": all_raw if attributions else None,
            "raw_attribution_sum": all_raw if attributions else None,
            "controlled_raw_attribution_sum": (
                controlled_raw if attributions else None
            ),
            "controlled_raw_reward_sum": (
                controlled_raw if attributions else None
            ),
            "uncontrolled_raw_attribution_sum": (
                uncontrolled_raw if attributions else None
            ),
            "uncontrolled_raw_reward_sum": (
                uncontrolled_raw if attributions else None
            ),
            "raw_attribution_sum_error": (
                abs(all_raw - cycle_rail_reward_raw)
                if attributions and cycle_rail_reward_raw is not None else None
            ),
            "controlled_elapsed_ratio": (
                sum(item["elapsed_share"] for item in attributions
                    if item["controlled"])
                if attributions else None
            ),
            "weighted_scaling_error_max": (
                max(abs(
                    item["weighted_preclip"]
                    - self.config.rail_tat_weight * item["raw_reward"]
                ) for item in attributions) if attributions else None
            ),
            "clip_contract_error_max": (
                max(abs(
                    item["postclip"]
                    - (
                        np.clip(
                            item["weighted_preclip"],
                            -self.config.rail_tat_clip,
                            self.config.rail_tat_clip,
                        ) if self.config.rail_tat_clip is not None
                        else item["weighted_preclip"]
                    )
                ) for item in attributions) if attributions else None
            ),
            "rail_tat_weight": float(self.config.rail_tat_weight),
            "preclip_abs_sum": preclip_abs_sum,
            "weighted_preclip_abs_sum": preclip_abs_sum,
            "postclip_abs_sum": postclip_abs_sum,
            "clip_removed_abs_sum": clip_removed_abs_sum,
            "clip_removed_ratio": (
                clip_removed_abs_sum / preclip_abs_sum
                if preclip_abs_sum > 0 else 0.0
            ),
            "clipped_assignment_count": int(clipped_count),
            "assignment_count": len(attributions),
            "clip_assignment_ratio": (
                clipped_count / len(attributions) if attributions else 0.0
            ),
        }
        if self.config.rail_reward_mode == RAIL_REWARD_BASELINE_RATIO:
            record["rail_baseline_ratio_reference"] = (
                self.config.rail_baseline_ratio_reference
            )
            record["baseline_ratio_error"] = (
                float(
                    route_free_flow_ratio
                    - self.config.rail_baseline_ratio_reference
                )
                if route_free_flow_ratio is not None else None
            )
        writer.append_cycle(global_step, record)

    def build(
        self,
        pclient,
        *,
        applied_action,
        previous_applied_action,
        env_step: int,
        episode_id: int,
    ) -> ControlledRewardBatch:
        count = len(self.topology.controlled_rail_ids)
        action = np.asarray(applied_action, dtype=np.float64).reshape(-1)
        if action.shape != (count,) or not np.isfinite(action).all():
            raise ContextualRewardError("applied_action must be finite [controlled]")
        if previous_applied_action is None:
            previous = action
        else:
            previous = np.asarray(previous_applied_action, dtype=np.float64).reshape(-1)
            if previous.shape != (count,) or not np.isfinite(previous).all():
                raise ContextualRewardError(
                    "previous_applied_action must be finite [controlled]"
                )

        global_raw = self._global_raw(pclient)
        local_raw = self._local_raw(pclient)
        self.reward_steps += 1

        global_component = self.config.global_alpha * global_raw
        local_scaled = local_raw / self.config.local_reward_scale
        local_component = self.config.local_alpha * local_scaled
        rail_reward_raw = self._rail_reward_raw(
            pclient,
            env_step=env_step,
            episode_id=episode_id,
        )
        (
            rail_reward_weighted_preclip,
            rail_reward_postclip,
        ) = self._scale_rail_reward(rail_reward_raw)
        if self.config.action_mode == REGION_B_RL:
            control_delta = np.abs(
                (0.5 + 0.5 * action) - (0.5 + 0.5 * previous)
            )
            smooth_weight = self.config.smooth_b_rl_weight
        else:
            control_delta = np.abs(action - previous)
            smooth_weight = self.config.smooth_exp_residual_weight
        smooth = smooth_weight * control_delta
        total = (
            global_component + local_component + rail_reward_postclip - smooth
        )
        arrays = {
            "total": total, "local_raw": local_raw,
            "local_scaled": local_scaled, "local_component": local_component,
            "rail_reward_raw": rail_reward_raw,
            "rail_reward_weighted_preclip": rail_reward_weighted_preclip,
            "rail_reward_postclip": rail_reward_postclip,
            "smooth_control_delta": control_delta,
            "smooth_penalty": smooth,
            "controlled_rail_ids": self.topology.controlled_rail_ids,
            "local_oht_raw": self._last_local_terms["oht"],
            "local_predicted_raw": self._last_local_terms["predicted"],
            "local_stop_raw": self._last_local_terms["stop"],
            "local_idle_raw": self._last_local_terms["idle"],
            "local_capacity_raw": self._last_local_terms["capacity"],
            "idle_oht_observation": self._last_idle_oht_observation,
        }
        frozen_arrays = {}
        for name, value in arrays.items():
            array = np.ascontiguousarray(np.asarray(value).copy())
            if array.shape != (count,) or not np.isfinite(array).all():
                raise ContextualRewardError(f"{name} is not finite [{count}]")
            array.setflags(write=False)
            frozen_arrays[name] = array
        return ControlledRewardBatch(
            **frozen_arrays,
            rail_tat_event_count=int(self._last_rail_tat_event_count),
            global_raw=float(global_raw),
            global_component=float(global_component),
            total_tat_level=float(self._last_global_terms["total_tat"]),
            tat_signal_available=float(
                self._last_global_terms["total_tat"] > 0.0
            ),
            tat_error=float(self._last_global_terms["tat_error"]),
            tat_raw_preclip=float(self._last_global_terms["tat_raw_preclip"]),
            tat_raw_postclip=float(self._last_global_terms["tat_raw_postclip"]),
            tat_confidence=float(self._last_global_terms["tat_confidence"]),
            tat_raw_ramped=float(self._last_global_terms["tat_raw_ramped"]),
            completed_episode=float(
                self._last_global_terms["completed_episode"]
            ),
            completed_delta=float(self._last_global_terms["completed_delta"]),
            op_rate=float(self._last_global_terms["op_rate"]),
            op_reference=float(self._last_global_terms["op_reference"]),
            op_error=float(self._last_global_terms["op_error"]),
            op_delta=float(self._last_global_terms["op_delta"]),
            op_raw=float(self._last_global_terms["op_raw"]),
            backlog=float(self._last_global_terms["backlog"]),
            backlog_raw=float(self._last_global_terms["backlog_raw"]),
            waiting=float(self._last_global_terms["waiting"]),
            queued=float(self._last_global_terms["queued"]),
            smooth_weight_effective=float(smooth_weight),
            env_step=int(env_step),
            episode_id=int(episode_id),
        )

    def diagnostics(self, batch: ControlledRewardBatch) -> dict[str, float]:
        rail_tat_sum = float(batch.rail_tat_penalty.sum())
        rail_tat_vector_mean = float(batch.rail_tat_penalty.mean())
        rail_tat_event_mean = (
            rail_tat_sum / batch.rail_tat_event_count
            if batch.rail_tat_event_count > 0 else 0.0
        )
        global_contribution = np.full_like(
            batch.local_component, batch.global_component
        )
        rail_tat_contribution = -batch.rail_tat_penalty
        smooth_contribution = -batch.smooth_penalty
        contributions = (
            global_contribution,
            batch.local_component,
            rail_tat_contribution,
            smooth_contribution,
        )
        abs_means = np.asarray(
            [float(np.mean(np.abs(value))) for value in contributions],
            dtype=np.float64,
        )
        abs_denominator = float(abs_means.sum()) + np.finfo(np.float64).eps
        abs_shares = abs_means / abs_denominator
        rail_nonzero = np.abs(batch.rail_reward_postclip[
            ~np.isclose(batch.rail_reward_postclip, 0.0)
        ])
        rail_active_representative = (
            float(np.median(rail_nonzero)) if rail_nonzero.size else 0.0
        )
        global_representative = float(abs(batch.global_component))
        local_representative = float(np.mean(np.abs(batch.local_component)))
        scale_values = np.asarray([
            global_representative,
            local_representative,
            rail_active_representative,
        ])
        positive_scales = scale_values[scale_values > 0.0]
        scale_mean = float(positive_scales.mean()) if positive_scales.size else 0.0
        balance_error = (
            float(np.mean(np.abs(positive_scales - scale_mean)) / scale_mean)
            if scale_mean > 0.0 else 0.0
        )
        result = {
            "reward/global_raw": batch.global_raw,
            "reward/global/total_tat": batch.total_tat_level,
            "reward/global/tat_reference": self.config.tat_reference,
            "reward/global/tat_error": batch.tat_error,
            "reward/global/tat_weight": self.config.tat_weight,
            "reward/global/tat_signal_available": (
                batch.tat_signal_available
            ),
            "reward/global/tat_raw_preclip": batch.tat_raw_preclip,
            "reward/global/tat_raw_postclip": batch.tat_raw_postclip,
            "reward/global/tat_confidence": batch.tat_confidence,
            "reward/global/tat_confidence_n0": self.config.tat_confidence_n0,
            "reward/global/tat_raw_ramped": batch.tat_raw_ramped,
            "reward/global/tat_raw": batch.tat_raw_ramped,
            "reward/global/tat_component": (
                self.config.global_alpha * batch.tat_raw_ramped
            ),
            "reward/global/completed_episode": batch.completed_episode,
            "reward/global/completed_delta": batch.completed_delta,
            "reward/global/op_rate": batch.op_rate,
            "reward/global/op_reference": batch.op_reference,
            "reward/global/op_error": batch.op_error,
            "reward/global/op_delta": batch.op_delta,
            "reward/global/op_weight": self.config.op_weight,
            "reward/global/op_raw": batch.op_raw,
            "reward/global/backlog": batch.backlog,
            "reward/global/backlog_weight": self.config.backlog_weight,
            "reward/global/backlog_raw": batch.backlog_raw,
            "reward/global/backlog_component": (
                self.config.global_alpha * batch.backlog_raw
            ),
            "reward/global/op_component": (
                self.config.global_alpha * batch.op_raw
            ),
            "reward/global/global_component": batch.global_component,
            "reward/global/raw_sum": batch.global_raw,
            "reward/global/raw_decomposition_error": abs(
                batch.global_raw
                - batch.tat_raw_ramped - batch.op_raw - batch.backlog_raw
            ),
            "reward/global_component": batch.global_component,
            "reward/total_tat_level": batch.total_tat_level,
            "reward/local_raw_mean": float(batch.local_raw.mean()),
            "reward/local_raw_std": float(batch.local_raw.std()),
            "reward/local_scaled_mean": float(batch.local_scaled.mean()),
            "reward/local_scaled_std": float(batch.local_scaled.std()),
            "reward/local_component_mean": float(batch.local_component.mean()),
            "reward/local_component_std": float(batch.local_component.std()),
            "reward/local/predicted_oht_weight": (
                self.config.local_predicted_oht_weight
            ),
            "reward/local/predicted_oht_component_abs_mean": float(
                self.config.local_alpha
                * np.mean(np.abs(batch.local_predicted_raw))
                / self.config.local_reward_scale
            ),
            "reward/local/local_reward_scale": self.config.local_reward_scale,
            "reward/local/local_component_abs_mean": local_representative,
            "reward/rail_tat_raw_mean": float(batch.rail_tat_raw.mean()),
            "reward/rail_tat_weighted_preclip_mean": float(
                batch.rail_tat_weighted_preclip.mean()
            ),
            "reward/rail_tat_penalty_mean": rail_tat_vector_mean,
            "reward/rail_tat_penalty_max": float(batch.rail_tat_penalty.max()),
            "reward/rail_tat_event_count": float(
                batch.rail_tat_event_count
            ),
            "reward/rail_tat_sum": rail_tat_sum,
            "reward/rail_tat_vector_mean": rail_tat_vector_mean,
            "reward/rail_tat_event_mean": float(rail_tat_event_mean),
            "reward/rail_tat_cycle_count": float(
                self._last_rail_tat_cycle_count
            ),
            "reward/rail_tat_controlled_assignment_count": float(
                self._last_rail_tat_controlled_assignment_count
            ),
            "reward/rail_tat_uncontrolled_assignment_count": float(
                self._last_rail_tat_uncontrolled_assignment_count
            ),
            "reward/smooth_penalty_mean": float(batch.smooth_penalty.mean()),
            "reward/smooth_penalty_max": float(batch.smooth_penalty.max()),
            "reward/smooth_control_delta_mean": float(
                batch.smooth_control_delta.mean()
            ),
            "reward/smooth_control_delta_max": float(
                batch.smooth_control_delta.max()
            ),
            "reward/total_mean": float(batch.total.mean()),
            "reward/total_std": float(batch.total.std()),
            "reward/total_min": float(batch.total.min()),
            "reward/total_max": float(batch.total.max()),
            "reward/finite_ratio": float(np.isfinite(batch.total).mean()),
            "reward/local/raw_decomposition_error_max": float(np.max(np.abs(
                batch.local_raw
                - batch.local_oht_raw
                - batch.local_predicted_raw
                - batch.local_stop_raw
                - batch.local_idle_raw
                - batch.local_capacity_raw
            ))),
            "reward/contribution/global_mean": float(
                global_contribution.mean()
            ),
            "reward/contribution/local_mean": float(
                batch.local_component.mean()
            ),
            "reward/contribution/rail_tat_mean": float(
                rail_tat_contribution.mean()
            ),
            "reward/contribution/smooth_mean": float(
                smooth_contribution.mean()
            ),
            "reward/contribution/total_mean": float(batch.total.mean()),
            "reward/contribution/sum_error": abs(float(
                batch.total.mean()
                - sum(float(value.mean()) for value in contributions)
            )),
            "reward/scale/global_abs_mean": float(abs_means[0]),
            "reward/scale/local_abs_mean": float(abs_means[1]),
            "reward/scale/rail_tat_abs_mean": float(abs_means[2]),
            "reward/scale/smooth_abs_mean": float(abs_means[3]),
            "reward/scale/total_abs_mean": float(
                np.mean(np.abs(batch.total))
            ),
            "reward/scale/global_abs_share": float(abs_shares[0]),
            "reward/scale/local_abs_share": float(abs_shares[1]),
            "reward/scale/rail_tat_abs_share": float(abs_shares[2]),
            "reward/scale/smooth_abs_share": float(abs_shares[3]),
            "reward/scale/abs_share_sum_error": abs(
                1.0 - float(abs_shares.sum())
            ),
            "reward/scale/global_representative": global_representative,
            "reward/scale/local_representative": local_representative,
            "reward/scale/rail_active_representative": (
                rail_active_representative
            ),
            "reward/scale/global_to_local": (
                global_representative / local_representative
                if local_representative > 0.0 else 0.0
            ),
            "reward/scale/global_to_rail": (
                global_representative / rail_active_representative
                if rail_active_representative > 0.0 else 0.0
            ),
            "reward/scale/local_to_rail": (
                local_representative / rail_active_representative
                if rail_active_representative > 0.0 else 0.0
            ),
            "reward/scale/main_balance_error": balance_error,
            "reward/rail/neutral_ratio": (
                self.config.rail_free_flow_neutral_ratio
            ),
            "reward/rail/weight": self.config.rail_tat_weight,
            "reward/rail/clip": float(self.config.rail_tat_clip or 0.0),
            "reward/rail/nonzero_raw_abs_mean": float(np.mean(np.abs(
                batch.rail_reward_raw[~np.isclose(batch.rail_reward_raw, 0.0)]
            ))) if np.any(~np.isclose(batch.rail_reward_raw, 0.0)) else 0.0,
            "reward/rail/nonzero_weighted_abs_mean": float(np.mean(np.abs(
                batch.rail_reward_weighted_preclip[
                    ~np.isclose(batch.rail_reward_weighted_preclip, 0.0)
                ]
            ))) if np.any(~np.isclose(
                batch.rail_reward_weighted_preclip, 0.0
            )) else 0.0,
            "reward/rail/nonzero_postclip_abs_mean": (
                float(rail_nonzero.mean()) if rail_nonzero.size else 0.0
            ),
            "reward/config/global_alpha": self.config.global_alpha,
            "reward/config/local_alpha": self.config.local_alpha,
            "reward/config/local_reward_scale": self.config.local_reward_scale,
            "reward/config/rail_tat_weight": self.config.rail_tat_weight,
            "reward/config/rail_tat_clip": float(
                self.config.rail_tat_clip
                if self.config.rail_tat_clip is not None else 0.0
            ),
            "reward/config/tat_raw_clip": float(
                self.config.tat_raw_clip
                if self.config.tat_raw_clip is not None else 0.0
            ),
            "reward/config/smooth_weight_effective": (
                batch.smooth_weight_effective
            ),
        }
        for name, values in (
            ("oht", batch.local_oht_raw),
            ("predicted", batch.local_predicted_raw),
            ("stop", batch.local_stop_raw),
            ("idle", batch.local_idle_raw),
            ("capacity", batch.local_capacity_raw),
        ):
            result[f"reward/local/{name}_raw_mean"] = float(values.mean())
            result[f"reward/local/{name}_raw_std"] = float(values.std())
            result[f"reward/local/{name}_raw_abs_mean"] = float(
                np.abs(values).mean()
            )
        local_term_abs = {
            name: result[f"reward/local/{name}_raw_abs_mean"]
            for name in ("oht", "predicted", "stop", "capacity")
        }
        local_term_denominator = (
            sum(local_term_abs.values()) + np.finfo(np.float64).eps
        )
        for name, output in (
            ("predicted", "predicted_abs_share"),
            ("oht", "oht_abs_share"),
            ("stop", "stop_abs_share"),
            ("capacity", "capacity_abs_share"),
        ):
            result[f"reward/local/{output}"] = (
                local_term_abs[name] / local_term_denominator
            )
        if not np.isfinite(tuple(result.values())).all():
            raise ContextualRewardError("reward diagnostics contain NaN or Inf")
        return result

    def save_normalizers(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        local, global_state = (
            self.local_normalizer.state_dict(),
            self.global_normalizer.state_dict(),
        )
        np.savez_compressed(
            target,
            reward_version=np.asarray(REWARD_VERSION),
            topology_hash=np.asarray(self.topology.topology_hash),
            mapping_hash=np.asarray(self.topology.mapping_hash),
            reward_steps=np.asarray(self.reward_steps),
            local_mean=local["mean"], local_m2=local["m2"],
            local_count=np.asarray(local["count"]),
            local_update_calls=np.asarray(local["update_calls"]),
            local_frozen=np.asarray(local["frozen"]),
            global_mean=global_state["mean"], global_m2=global_state["m2"],
            global_count=np.asarray(global_state["count"]),
            global_update_calls=np.asarray(global_state["update_calls"]),
            global_frozen=np.asarray(global_state["frozen"]),
        )
        return target

    def load_normalizers(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as saved:
            saved_reward_version = (
                str(saved["reward_version"].item())
                if "reward_version" in saved.files else None
            )
            if saved_reward_version != REWARD_VERSION:
                raise ContextualRewardError(
                    "reward normalizer version mismatch: "
                    f"saved={saved_reward_version!r}, "
                    f"runtime={REWARD_VERSION!r}. A fresh normalizer is "
                    "required because statistics from another reward contract "
                    "are incompatible."
                )
            if str(saved["topology_hash"].item()) != self.topology.topology_hash:
                raise ContextualRewardError("reward normalizer topology hash mismatch")
            if str(saved["mapping_hash"].item()) != self.topology.mapping_hash:
                raise ContextualRewardError("reward normalizer mapping hash mismatch")
            for prefix, normalizer in (
                ("local", self.local_normalizer),
                ("global", self.global_normalizer),
            ):
                normalizer.load_state_dict({
                    "dim": 1,
                    "mean": saved[f"{prefix}_mean"],
                    "m2": saved[f"{prefix}_m2"],
                    "count": int(saved[f"{prefix}_count"].item()),
                    "update_calls": int(saved[f"{prefix}_update_calls"].item()),
                    "frozen": bool(saved[f"{prefix}_frozen"].item()),
                })
            self.reward_steps = int(saved["reward_steps"].item())
