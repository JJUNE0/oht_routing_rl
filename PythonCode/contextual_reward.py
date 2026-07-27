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
from contextual_topology import ContextualTopology

REWARD_VERSION = (
    "contextual_controlled_reward_v7_oht_cycle_segments_boundary_delta"
)

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
    rail_tat_weight: float = 1.0
    action_mode: str = REGION_B_RL
    smooth_b_rl_weight: float = 0.5
    smooth_exp_residual_weight: float = 0.5
    tat_weight: float = 9.2
    op_weight: float = 5.0
    backlog_weight: float = 0.01
    use_tat: bool = True
    use_op: bool = False
    use_backlog: bool = True
    tat_reference: float = 2.90706 * 60.0
    tat_ema_beta: float = 0.05
    freeze_after_env_steps: int = 30_000
    normalizer_epsilon: float = 1e-6
    global_clip: float | None = 5.0
    local_clip: float | None = None

    def __post_init__(self):
        numeric = (
            self.global_alpha, self.local_alpha, self.rail_tat_weight,
            self.smooth_b_rl_weight, self.smooth_exp_residual_weight,
            self.tat_weight, self.op_weight,
            self.backlog_weight, self.tat_reference, self.tat_ema_beta,
            self.normalizer_epsilon,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("reward config contains NaN or Inf")
        if self.tat_reference <= 0 or self.normalizer_epsilon <= 0:
            raise ValueError("tat_reference and normalizer_epsilon must be positive")
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
    global_normalized: float
    global_component: float
    local_raw: np.ndarray
    local_normalized: np.ndarray
    local_component: np.ndarray
    rail_tat_penalty: np.ndarray
    rail_tat_event_count: int
    smooth_control_delta: np.ndarray
    smooth_penalty: np.ndarray
    controlled_rail_ids: np.ndarray
    marginal_tat_ema: float
    tat_signal_available: float
    env_step: int
    episode_id: int


@dataclass
class OHTCycleTracker:
    oht_id: int
    cycle_active: bool = False
    bootstrapped: bool = False
    cycle_start_step: int | None = None
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

    def clear_cycle(self) -> None:
        self.cycle_active = False
        self.bootstrapped = False
        self.cycle_start_step = None
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

    @property
    def last_valid_oht_tat(self) -> float | None:
        """Compatibility alias used by diagnostics and older tests."""
        return self.current_segment_last_valid_oht_tat

    @property
    def last_unloading_oht_tat(self) -> float | None:
        """Compatibility alias used by diagnostics and older tests."""
        return self.current_segment_last_state5_oht_tat


@dataclass
class RailPassTemporalTracker:
    """Packet-to-packet state for one OHT's current rail occurrence."""

    inflight_rail_id: int | None = None
    inflight_state: int | None = None
    inflight_elapsed: float | None = None

    def clear(self) -> None:
        self.inflight_rail_id = None
        self.inflight_state = None
        self.inflight_elapsed = None


@dataclass(frozen=True)
class CycleRewardOutcome:
    reward_applied: bool
    skip_reason: str | None
    effective_oht_tat: float | None = None
    signed_excess: float | None = None
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
    ):
        self.topology = topology
        self.config = config or ContextualRewardConfig()
        self.completion_diagnostic_path = (
            Path(completion_diagnostic_path)
            if completion_diagnostic_path is not None
            else None
        )
        self._global_step_provider = global_step_provider
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
        self._prev_completed: float | None = None
        self._prev_tat_sum = 0.0
        self._prev_op_rate: float | None = None
        self._tat_ema = 0.0
        self._last_rail_tat_event_count = 0
        self._oht_cycle_trackers.clear()
        self._rail_pass_trackers.clear()

    def reset_episode(self) -> None:
        """Clear only episode-temporal state; training statistics persist."""
        self._reset_temporal()

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
        cur_tat_sum = cur_tat * completed
        waiting, queued = self._job_backlog(pclient)

        if self._prev_completed is None:
            raw = 0.0
        else:
            delta_completed = completed - self._prev_completed
            if delta_completed > 0:
                marginal_tat = (
                    cur_tat_sum - self._prev_tat_sum
                ) / delta_completed
                beta = cfg.tat_ema_beta
                self._tat_ema = (
                    marginal_tat if self._tat_ema <= 0
                    else (1 - beta) * self._tat_ema + beta * marginal_tat
                )
            # No completed command means there is no TAT observation. Treating
            # the initial zero EMA as a real zero-second TAT awarded the maximum
            # positive TAT reward while the factory state was unchanged.
            tat_term = (
                (cfg.tat_reference - self._tat_ema) / cfg.tat_reference
                if self._tat_ema > 0
                else 0.0
            )
            op_delta = float(self._prev_op_rate) - cur_op
            raw = (
                (cfg.tat_weight * tat_term if cfg.use_tat else 0.0)
                + (cfg.op_weight * op_delta if cfg.use_op else 0.0)
                - (cfg.backlog_weight * (waiting + queued)
                   if cfg.use_backlog else 0.0)
            )
        self._prev_completed = completed
        self._prev_tat_sum = cur_tat_sum
        self._prev_op_rate = cur_op
        return float(raw)

    def _local_raw(self, pclient) -> np.ndarray:
        result = np.empty(len(self.topology.controlled_rail_ids), dtype=np.float64)
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
            result[row] = -(
                0.3 * oht_count
                + 0.2 * float(getattr(rail, "PredictedOHTCount"))
                + 0.3 * avg_stop
                + 0.1 * float(getattr(rail, "IdleOHTCount"))
                + 0.1 * capacity
            )
        if not np.isfinite(result).all():
            raise ContextualRewardError("local reward input contains NaN or Inf")
        return result

    def _rail_tat(
        self,
        pclient,
        *,
        env_step: int,
        episode_id: int = 0,
    ) -> np.ndarray:
        credit: dict[int, float] = {}
        event_count = 0
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
                if outcome.reward_applied:
                    event_count += outcome.controlled_reward_count
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
        self._last_rail_tat_event_count = int(event_count)
        return result * self.config.rail_tat_weight

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

    @staticmethod
    def _start_oht_cycle(
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
        if not tracker.cycle_tat_valid:
            return CycleRewardOutcome(
                False, tracker.last_tat_skip_reason or "invalid_cycle_tat"
            )
        if tracker.current_segment_last_state5_oht_tat is None:
            return CycleRewardOutcome(False, (
                tracker.last_tat_skip_reason or "missing_final_state_5_oht_tat"
            ))
        if (
            not tracker.rail_time_by_id
            or not np.isfinite(tracker.route_time)
            or tracker.route_time <= 0
        ):
            return CycleRewardOutcome(False, "missing_valid_cycle_route")
        effective_oht_tat = (
            tracker.closed_segment_oht_tat_sum
            + tracker.current_segment_last_state5_oht_tat
        )
        excess = (
            effective_oht_tat - self.config.tat_reference
        ) / self.config.tat_reference
        controlled = set(
            int(value) for value in self.topology.controlled_rail_ids
        )
        actual_controlled_rewards = []
        for rail_id, elapsed in tracker.rail_time_by_id.items():
            penalty = excess * elapsed / tracker.route_time
            credit[rail_id] = credit.get(rail_id, 0.0) + penalty
            if rail_id in controlled:
                actual_controlled_rewards.append(
                    -penalty * self.config.rail_tat_weight
                )
        controlled_count = sum(
            rail_id in controlled for rail_id in tracker.rail_time_by_id
        )
        values = np.asarray(actual_controlled_rewards, dtype=np.float64)
        return CycleRewardOutcome(
            True,
            None,
            effective_oht_tat=float(effective_oht_tat),
            signed_excess=float(excess),
            cycle_reward_unclipped=float(
                -excess * self.config.rail_tat_weight
            ),
            cycle_reward_clipped=float(
                -excess * self.config.rail_tat_weight
            ),
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
        outcome = outcome or CycleRewardOutcome(
            bool(reward_applied), skip_reason
        )
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
            )
            stream.write("\n")
            stream.flush()

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
        global_norm = float(
            self.global_normalizer.normalize([[global_raw]], name="global_reward")[0, 0]
        )
        local_norm = self.local_normalizer.normalize(
            local_raw[:, None], name="local_reward"
        )[:, 0]
        # Contractual normalize-before-update; exactly one batch update each.
        self.global_normalizer.update([[global_raw]], name="global_reward")
        self.local_normalizer.update(local_raw[:, None], name="local_reward")
        self.reward_steps += 1
        if self.reward_steps >= self.config.freeze_after_env_steps:
            self.global_normalizer.freeze()
            self.local_normalizer.freeze()

        global_component = self.config.global_alpha * global_norm
        local_component = self.config.local_alpha * local_norm
        rail_tat = self._rail_tat(
            pclient,
            env_step=env_step,
            episode_id=episode_id,
        )
        if self.config.action_mode == REGION_B_RL:
            control_delta = np.abs(
                (0.5 + 0.5 * action) - (0.5 + 0.5 * previous)
            )
            smooth_weight = self.config.smooth_b_rl_weight
        else:
            control_delta = np.abs(action - previous)
            smooth_weight = self.config.smooth_exp_residual_weight
        smooth = smooth_weight * control_delta
        total = global_component + local_component - rail_tat - smooth
        arrays = {
            "total": total, "local_raw": local_raw,
            "local_normalized": local_norm, "local_component": local_component,
            "rail_tat_penalty": rail_tat,
            "smooth_control_delta": control_delta,
            "smooth_penalty": smooth,
            "controlled_rail_ids": self.topology.controlled_rail_ids,
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
            global_normalized=float(global_norm),
            global_component=float(global_component),
            marginal_tat_ema=float(self._tat_ema),
            tat_signal_available=float(self._tat_ema > 0),
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
        return {
            "reward/global_raw": batch.global_raw,
            "reward/global_normalized": batch.global_normalized,
            "reward/global_component": batch.global_component,
            "reward/marginal_tat_ema": batch.marginal_tat_ema,
            "reward/tat_signal_available": batch.tat_signal_available,
            "reward/local_raw_mean": float(batch.local_raw.mean()),
            "reward/local_raw_std": float(batch.local_raw.std()),
            "reward/local_normalized_mean": float(batch.local_normalized.mean()),
            "reward/local_normalized_std": float(batch.local_normalized.std()),
            "reward/local_component_mean": float(batch.local_component.mean()),
            "reward/local_component_std": float(batch.local_component.std()),
            "reward/rail_tat_penalty_mean": rail_tat_vector_mean,
            "reward/rail_tat_penalty_max": float(batch.rail_tat_penalty.max()),
            "reward/rail_tat_event_count": float(
                batch.rail_tat_event_count
            ),
            "reward/rail_tat_sum": rail_tat_sum,
            "reward/rail_tat_vector_mean": rail_tat_vector_mean,
            "reward/rail_tat_event_mean": float(rail_tat_event_mean),
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
            "normalizer/local_reward_update_calls": float(
                self.local_normalizer.update_calls
            ),
            "normalizer/local_reward_sample_count": float(
                self.local_normalizer.count
            ),
            "normalizer/global_reward_update_calls": float(
                self.global_normalizer.update_calls
            ),
            "normalizer/global_reward_sample_count": float(
                self.global_normalizer.count
            ),
        }

    def save_normalizers(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        local, global_state = (
            self.local_normalizer.state_dict(),
            self.global_normalizer.state_dict(),
        )
        np.savez_compressed(
            target,
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
