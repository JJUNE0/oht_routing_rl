"""The evaluate.py activation-window TAT, distinct from input/episode duration."""
from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3

from simulator.evaluate import start_time, end_time, tat_total_base, trcnt_total_base

# Meeting contract for the current input (model start 2026-02-16 07:00).
MEETING_WINDOW_START = "2026-02-16 07:00:00"
MEETING_WINDOW_END = "2026-02-16 19:00:00"


def score_meeting_window(database, *, full_rollout=False):
    """Twelve-hour score without reusing the ten-hour reference constants."""
    return score_final_window(database, full_rollout=full_rollout,
                              window_start=MEETING_WINDOW_START, window_end=MEETING_WINDOW_END)


def score_final_window(database, *, full_rollout=False, window_start=None, window_end=None):
    """Read native result timestamps with the same strict boundaries as evaluate.py.

    A short or interrupted rollout can have a deceptively good partial average.
    Only a caller-confirmed full rollout covering the window gets a target verdict.
    Command coverage remains explicit; the saved baseline count is not inferred
    from the rows that happen to have completed in this result.
    """
    window_start = window_start or start_time
    window_end = window_end or end_time
    if datetime.fromisoformat(window_end) <= datetime.fromisoformat(window_start):
        raise ValueError("Evaluation end time must be after start time")
    legacy_reference = (datetime.fromisoformat(window_start) == datetime.fromisoformat(start_time)
                        and datetime.fromisoformat(window_end) == datetime.fromisoformat(end_time))
    path = Path(database).resolve()
    seconds, count = 0.0, 0
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        last_completion = connection.execute("SELECT MAX(COMPLETED_TIME) FROM COMMAND_LOG").fetchone()[0]
        rows = connection.execute(
            "SELECT ACTIVATED_TIME, COMPLETED_TIME FROM COMMAND_LOG "
            "WHERE ACTIVATED_TIME > ? AND ACTIVATED_TIME < ?", (window_start, window_end),
        )
        selected = 0
        for activated, completed in rows:
            selected += 1
            if not completed:
                continue
            seconds += (datetime.fromisoformat(completed) - datetime.fromisoformat(activated)).total_seconds()
            count += 1
    tat_s = seconds / count if count else None
    complete = bool(full_rollout and last_completion and
                    datetime.fromisoformat(last_completion) >= datetime.fromisoformat(window_end))
    baseline = tat_total_base * 60.0 if legacy_reference else None
    target = baseline * 0.95 if baseline is not None else None
    return {
        "eval_window_start": window_start,
        "eval_window_end": window_end,
        "eval_baseline_tat_s": baseline,
        "eval_target_tat_s": target,
        "eval_command_count": selected,
        "eval_completed_samples": count,
        "eval_baseline_command_count": trcnt_total_base if legacy_reference else None,
        "eval_command_coverage": selected / trcnt_total_base if legacy_reference else None,
        "eval_tat_s": tat_s,
        "eval_improvement_pct": (1.0 - tat_s / baseline) * 100.0 if tat_s is not None and baseline is not None else None,
        "eval_window_complete": complete,
        "eval_tat_target_met": tat_s <= target if complete and tat_s is not None and target is not None else None,
    }
