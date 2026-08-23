"""Rolling completed-command TAT derived from live OHT lifecycle packets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from simulator.oht import OHTState


RECENT_COMPLETED_TAT_WINDOW_SECONDS = 300.0


@dataclass(frozen=True)
class RecentCompletedTatSnapshot:
    mean_s: float = 0.0
    p90_s: float = 0.0
    event_count: int = 0
    available: bool = False
    window_age_s: float = 0.0
    events_added: int = 0
    duplicate_event_count: int = 0
    missing_state5_tat_count: int = 0
    ambiguous_entry_count: int = 0


@dataclass
class _OHTCompletionState:
    previous_state: int
    bootstrapped_in_state5: bool = False
    state5_command_id: int | None = None
    state5_command_tat_s: float | None = None


class RecentCompletedTatTracker:
    """Track the final command TAT for completions in a simulation-time window.

    ``CmdCompleteTat`` contains an in-flight command TAT on repeated packets.
    A value therefore becomes a completion sample only at the verified
    ``UNLOADING -> IDLE/MOVE_TO_LOAD/LOADING`` lifecycle boundary.  Command IDs
    are deduplicated for the episode so repeated packets cannot overweight a
    job.
    """

    _COMPLETION_SUCCESSORS = frozenset({
        int(OHTState.IDLE),
        int(OHTState.MOVE_TO_LOAD),
        int(OHTState.LOADING),
    })

    def __init__(
        self,
        *,
        window_seconds: float = RECENT_COMPLETED_TAT_WINDOW_SECONDS,
    ):
        window_seconds = float(window_seconds)
        if not np.isfinite(window_seconds) or window_seconds <= 0.0:
            raise ValueError("recent completed TAT window must be positive")
        self.window_seconds = window_seconds
        self.reset_episode()

    def reset_episode(self) -> None:
        self._oht_states: dict[int, _OHTCompletionState] = {}
        self._events: deque[tuple[float, int, float]] = deque()
        self._completed_command_ids: set[int] = set()
        self._episode_start_sim_time: float | None = None
        self._last_sim_time: float | None = None
        self._duplicate_event_count = 0
        self._missing_state5_tat_count = 0
        self._ambiguous_entry_count = 0
        self._last_snapshot = RecentCompletedTatSnapshot()

    @property
    def snapshot(self) -> RecentCompletedTatSnapshot:
        return self._last_snapshot

    @staticmethod
    def _entries(oht) -> dict[int, object]:
        values = getattr(oht, "CmdCompleteTat", None)
        pairs = (
            values.items()
            if hasattr(values, "items")
            else (
                (getattr(value, "CmdID", 0), value)
                for value in (values or [])
            )
        )
        result = {}
        for key, value in pairs:
            try:
                command_id = int(key or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if command_id > 0:
                result[command_id] = value
        return result

    def _selected_state5_tat(self, oht) -> tuple[int, float] | None:
        entries = self._entries(oht)
        candidates = set()
        for candidate in (
            getattr(oht, "JobID", 0),
            getattr(oht, "DispatchedCommand", 0),
        ):
            try:
                command_id = int(candidate or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if command_id in entries:
                candidates.add(command_id)
        if len(candidates) == 1:
            command_id = next(iter(candidates))
        elif len(candidates) > 1:
            self._ambiguous_entry_count += 1
            return None
        elif len(entries) == 1:
            command_id = next(iter(entries))
        else:
            if len(entries) > 1:
                self._ambiguous_entry_count += 1
            return None
        try:
            command_tat_s = float(
                getattr(entries[command_id], "CmdTat", 0.0) or 0.0
            )
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(command_tat_s) or command_tat_s <= 0.0:
            return None
        return int(command_id), command_tat_s

    def _expire(self, sim_time: float) -> None:
        cutoff = sim_time - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _make_snapshot(
        self, sim_time: float, *, events_added: int
    ) -> RecentCompletedTatSnapshot:
        self._expire(sim_time)
        values = np.asarray(
            [event[2] for event in self._events], dtype=np.float64
        )
        start = (
            sim_time
            if self._episode_start_sim_time is None
            else self._episode_start_sim_time
        )
        age = max(0.0, sim_time - start)
        available = bool(age >= self.window_seconds and values.size > 0)
        return RecentCompletedTatSnapshot(
            mean_s=float(values.mean()) if values.size else 0.0,
            p90_s=float(np.percentile(values, 90)) if values.size else 0.0,
            event_count=int(values.size),
            available=available,
            window_age_s=float(min(age, self.window_seconds)),
            events_added=int(events_added),
            duplicate_event_count=int(self._duplicate_event_count),
            missing_state5_tat_count=int(self._missing_state5_tat_count),
            ambiguous_entry_count=int(self._ambiguous_entry_count),
        )

    def update(self, pclient) -> RecentCompletedTatSnapshot:
        try:
            sim_time = float(getattr(pclient, "SimTime", 0.0))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("simulation time for recent TAT is invalid") from error
        if not np.isfinite(sim_time) or sim_time < 0.0:
            raise ValueError("simulation time for recent TAT must be non-negative")
        if self._last_sim_time is not None and sim_time < self._last_sim_time:
            self.reset_episode()
        if self._episode_start_sim_time is None:
            self._episode_start_sim_time = sim_time
        if self._last_sim_time is not None and sim_time == self._last_sim_time:
            return self._last_snapshot

        events_added = 0
        ohts = getattr(pclient, "OHT_DIC", {})
        observed_ids = set()
        for key, oht in ohts.items():
            oht_id = int(getattr(oht, "ID", key))
            observed_ids.add(oht_id)
            state = int(getattr(oht, "State", OHTState.NULL))
            tracked = self._oht_states.get(oht_id)
            if tracked is None:
                tracked = _OHTCompletionState(
                    previous_state=state,
                    bootstrapped_in_state5=(state == int(OHTState.UNLOADING)),
                )
                self._oht_states[oht_id] = tracked
            elif (
                tracked.previous_state == int(OHTState.UNLOADING)
                and state in self._COMPLETION_SUCCESSORS
            ):
                if tracked.bootstrapped_in_state5:
                    tracked.bootstrapped_in_state5 = False
                elif (
                    tracked.state5_command_id is None
                    or tracked.state5_command_tat_s is None
                ):
                    self._missing_state5_tat_count += 1
                elif tracked.state5_command_id in self._completed_command_ids:
                    self._duplicate_event_count += 1
                else:
                    self._completed_command_ids.add(tracked.state5_command_id)
                    self._events.append((
                        sim_time,
                        tracked.state5_command_id,
                        tracked.state5_command_tat_s,
                    ))
                    events_added += 1
                tracked.state5_command_id = None
                tracked.state5_command_tat_s = None

            if state == int(OHTState.UNLOADING):
                selected = self._selected_state5_tat(oht)
                if selected is not None:
                    tracked.state5_command_id = selected[0]
                    tracked.state5_command_tat_s = selected[1]
            elif tracked.previous_state != int(OHTState.UNLOADING):
                tracked.state5_command_id = None
                tracked.state5_command_tat_s = None
            tracked.previous_state = state

        for stale_id in set(self._oht_states).difference(observed_ids):
            self._oht_states.pop(stale_id, None)

        self._last_sim_time = sim_time
        self._last_snapshot = self._make_snapshot(
            sim_time, events_added=events_added
        )
        return self._last_snapshot


__all__ = (
    "RECENT_COMPLETED_TAT_WINDOW_SECONDS",
    "RecentCompletedTatSnapshot",
    "RecentCompletedTatTracker",
)
