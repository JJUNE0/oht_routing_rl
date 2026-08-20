"""Reward N rail-cycle tracking and free-flow credit assignment."""

import json
from dataclasses import dataclass, field

import numpy as np

from Oht import OHTState


ACTIVE_OHT_CYCLE_STATES = frozenset({
    int(OHTState.MOVE_TO_LOAD),
    int(OHTState.LOADING),
    int(OHTState.MOVE_TO_UNLOAD),
    int(OHTState.UNLOADING),
})


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
    current_segment_last_state5_cmd_tat: float | None = None
    current_segment_last_state5_command_id: int | None = None
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
        self.current_segment_last_state5_cmd_tat = None
        self.current_segment_last_state5_command_id = None
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


class ContextualRailRewardMixin:
    """OHT cycle state machine used by the locked Reward N rail term."""

    def _rail_reward_raw(
        self,
        pclient,
        *,
        env_step: int,
        episode_id: int = 0,
    ) -> np.ndarray:
        self._route_ratio_sample_added = False
        credit: dict[int, float] = {}
        cycle_count = 0
        controlled_assignment_count = 0
        uncontrolled_assignment_count = 0
        diagnostic_active = self._cycle_diagnostic_active()
        self._diagnostic_rails = getattr(pclient, "RAILLINE_DIC", {})
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
            current_cmd_id = self._positive_command_id(selected)
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
                        current_cmd_id,
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
                        current_cmd_id,
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
                        current_cmd_id,
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
                    current_cmd_id,
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
                    current_cmd_id,
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
        pairs = (
            values.items()
            if hasattr(values, "items")
            else (
                (getattr(value, "CmdID", 0), value)
                for value in (values or [])
            )
        )
        entries = {}
        for key, value in pairs:
            try:
                command_id = int(key or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            entries[command_id] = value
        return entries

    @staticmethod
    def _select_oht_tat_entry(oht, entries):
        if not entries:
            return None, "missing_tat_entry"
        candidates = set()
        for candidate in (
            getattr(oht, "JobID", 0),
            getattr(oht, "DispatchedCommand", 0),
        ):
            try:
                candidate = int(candidate or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if candidate in entries:
                candidates.add(candidate)
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
        try:
            result = float(getattr(value, name, 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if np.isfinite(result) and result > 0 else None

    @staticmethod
    def _finite_nonnegative_attr(value, name) -> float | None:
        if value is None:
            return None
        try:
            result = float(getattr(value, name, 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if np.isfinite(result) and result >= 0 else None

    @staticmethod
    def _positive_command_id(value) -> int | None:
        if value is None:
            return None
        try:
            result = int(getattr(value, "CmdID", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if result > 0 else None

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
        self._start_tat_segment(tracker, int(job_id), increment=True)

    @staticmethod
    def _start_tat_segment(tracker, job_id, *, increment) -> None:
        tracker.current_segment_job_id = int(job_id)
        tracker.current_segment_last_valid_oht_tat = None
        tracker.current_segment_last_state5_oht_tat = None
        tracker.current_segment_last_state5_cmd_tat = None
        tracker.current_segment_last_state5_command_id = None
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
        cmd_id,
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
        if int(state) == int(OHTState.UNLOADING):
            if cmd_tat is not None and cmd_id is not None:
                tracker.current_segment_last_state5_cmd_tat = float(cmd_tat)
                tracker.current_segment_last_state5_command_id = int(cmd_id)
            else:
                # Do not fall back to an earlier lifecycle sample in cycle
                # diagnostics; only the actual state-5 record is valid.
                tracker.current_segment_last_state5_cmd_tat = None
                tracker.current_segment_last_state5_command_id = None
        return unexplained_reset

    def _finalize_oht_cycle(self, tracker, credit) -> CycleRewardOutcome:
        if tracker.bootstrapped:
            return CycleRewardOutcome(False, "bootstrapped_incomplete_cycle")
        if (
            not tracker.rail_time_by_id
            or not np.isfinite(tracker.route_time)
            or tracker.route_time <= 0
        ):
            return CycleRewardOutcome(
                False,
                "invalid_free_flow_cycle",
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
        cycle_rail_reward_raw = float(
            self.config.rail_free_flow_neutral_ratio
            - route_free_flow_ratio
        )

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
        outcome = CycleRewardOutcome(
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
        if route_free_flow_ratio is not None:
            self._recent_route_ratios.append(float(route_free_flow_ratio))
            self._recent_route_negative.append(
                float(cycle_rail_reward_raw < 0.0)
            )
            self._route_ratio_sample_added = True
        return outcome

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
            "reward_version": self.reward_version,
            "reward_contract_version": self.reward_contract_version,
            "reward_tat_version": self.reward_tat_version,
            "reward_normalization_version": (
                self.reward_normalization_version
            ),
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
        writer.append_cycle(global_step, record)
