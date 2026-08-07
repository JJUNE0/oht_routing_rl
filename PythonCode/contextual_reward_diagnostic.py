"""Bounded JSONL/W&B diagnostics for contextual rail reward contracts."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


DIAGNOSTIC_SCHEMA_VERSION = "contextual_reward_diagnostic_v15_balanced_neutral2"
DEFAULT_DIAGNOSTIC_WINDOWS = (
    (0, 1_000, "00000_01000"),
    (10_000, 11_000, "10000_11000"),
    (20_000, 21_000, "20000_21000"),
)


def parse_diagnostic_windows(value=None):
    if value is None or value == "":
        return DEFAULT_DIAGNOSTIC_WINDOWS
    if not isinstance(value, str):
        windows = tuple(value)
    else:
        windows = []
        for item in value.split(","):
            start_text, end_text = item.strip().split(":", 1)
            start, end = int(start_text), int(end_text)
            windows.append((start, end, f"{start:05d}_{end:05d}"))
        windows = tuple(windows)
    normalized = []
    previous_end = -1
    for item in windows:
        if len(item) == 2:
            start, end = item
            name = f"{int(start):05d}_{int(end):05d}"
        elif len(item) == 3:
            start, end, name = item
        else:
            raise ValueError("diagnostic windows must contain start/end[/name]")
        start, end = int(start), int(end)
        if start < 0 or end <= start or start < previous_end:
            raise ValueError("diagnostic windows must be ordered and non-overlapping")
        normalized.append((start, end, str(name)))
        previous_end = end
    return tuple(normalized)


def diagnostic_window_name(global_step: int, windows=None) -> str | None:
    step = int(global_step)
    for start, end, name in parse_diagnostic_windows(windows):
        if start <= step < end:
            return name
    return None


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


class RewardDiagnosticWriter:
    """Append windowed records and retain only bounded cycle samples."""

    def __init__(self, directory, windows=None, *, cycle_buffer_size=10_000):
        self.directory = Path(directory) if directory is not None else None
        self.windows = parse_diagnostic_windows(windows)
        if int(cycle_buffer_size) <= 0:
            raise ValueError("cycle_buffer_size must be positive")
        self._cycle_records = defaultdict(
            lambda: deque(maxlen=int(cycle_buffer_size))
        )

    def window_name(self, global_step):
        return diagnostic_window_name(global_step, self.windows)

    @property
    def writes_json(self):
        return self.directory is not None

    def window_bounds(self, global_step):
        step = int(global_step)
        for start, end, name in self.windows:
            if start <= step < end:
                return int(start), int(end), str(name)
        return None

    def _append(self, prefix, window, record):
        if self.directory is None:
            return None
        path = self.directory / f"{prefix}_window_{window}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        safe = _json_safe(record)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            json.dump(
                safe,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
        return path

    def append_step(self, global_step, record):
        window = self.window_name(global_step)
        if window is None:
            return None
        payload = dict(record)
        payload["schema_version"] = DIAGNOSTIC_SCHEMA_VERSION
        payload["diagnostic_window"] = window
        return self._append("reward_step", window, payload)

    def append_cycle(self, global_step, record):
        window = self.window_name(global_step)
        if window is None:
            return None
        payload = dict(record)
        payload["schema_version"] = DIAGNOSTIC_SCHEMA_VERSION
        payload["diagnostic_window"] = window
        self._cycle_records[window].append(_json_safe(payload))
        return self._append("rail_cycle", window, payload)

    @staticmethod
    def _finite(records, key):
        return np.asarray(
            [float(item[key]) for item in records if item.get(key) is not None],
            dtype=np.float64,
        )

    def cycle_summary(self, global_step):
        window = self.window_name(global_step)
        records = tuple(self._cycle_records.get(window, ()))
        prefix = "reward/rail/"
        result = {
            prefix + "mode": (
                records[-1].get("rail_reward_mode") if records else None
            ),
            prefix + "free_flow_neutral_ratio": (
                records[-1].get("rail_free_flow_neutral_ratio")
                if records else None
            ),
            prefix + "neutral_ratio": (
                records[-1].get("rail_free_flow_neutral_ratio")
                if records else None
            ),
            prefix + "weight": (
                records[-1].get("rail_tat_weight") if records else None
            ),
            prefix + "clip": (
                records[-1].get("rail_tat_clip") if records else None
            ),
            prefix + "cycle_count": float(len(records)),
            prefix + "reward_applied_cycle_count": float(sum(
                bool(item.get("reward_applied")) for item in records
            )),
            prefix + "skipped_cycle_count": float(sum(
                not bool(item.get("reward_applied")) for item in records
            )),
        }
        for key, output in (
            ("effective_oht_tat", "effective_oht_tat"),
            ("signed_excess", "signed_excess"),
            ("route_time", "route_time"),
            ("route_free_flow_time", "route_free_flow_time"),
            ("route_delay_ratio", "route_delay_ratio"),
            ("effective_to_freeflow_ratio", "effective_to_freeflow_ratio"),
            ("service_residual_time", "service_residual_time"),
        ):
            values = self._finite(records, key)
            result[prefix + output + "_mean"] = (
                float(values.mean()) if values.size else 0.0
            )
            if key in {"effective_oht_tat", "signed_excess", "route_delay_ratio"}:
                for percentile in (50, 90, 95):
                    if key == "signed_excess" and percentile == 90:
                        continue
                    if key == "route_delay_ratio" and percentile != 95:
                        continue
                    result[prefix + output + f"_p{percentile}"] = (
                        float(np.percentile(values, percentile))
                        if values.size else 0.0
                    )
        applied = [item for item in records if item.get("reward_applied")]
        denominator = max(1, len(applied))
        result.update({
            prefix + "positive_cycle_ratio": float(sum(
                bool(item.get("cycle_is_positive_reward"))
                for item in applied
            ) / denominator),
            prefix + "negative_cycle_ratio": float(sum(
                bool(item.get("cycle_is_negative_reward"))
                for item in applied
            ) / denominator),
            prefix + "zero_cycle_ratio": float(sum(
                bool(item.get("cycle_is_zero_reward"))
                for item in applied
            ) / denominator),
            prefix + "route_free_flow_available_ratio": float(sum(
                bool(item.get("route_free_flow_available")) for item in records
            ) / max(1, len(records))),
        })
        for key in ("clip_assignment_ratio", "clip_removed_ratio"):
            values = self._finite(records, key)
            result[prefix + key] = float(values.mean()) if values.size else 0.0
        for key, output in (
            ("route_free_flow_ratio", "route_ratio"),
            ("cycle_rail_reward_raw", "cycle_reward_raw"),
        ):
            values = self._finite(applied, key)
            result[prefix + output + "_mean"] = (
                float(values.mean()) if values.size else 0.0
            )
            percentiles = (50, 90, 95, 10) if key == "cycle_rail_reward_raw" else (50, 90, 95)
            for percentile in percentiles:
                result[prefix + output + f"_p{percentile}"] = (
                    float(np.percentile(values, percentile))
                    if values.size else 0.0
                )
        raw_nonzero = []
        weighted_abs = []
        postclip_nonzero = []
        for item in applied:
            for attribution in item.get("rail_attributions", ()):
                raw = attribution.get("raw_reward")
                weighted = attribution.get("weighted_preclip")
                postclip = attribution.get("postclip")
                if raw is not None and raw != 0:
                    raw_nonzero.append(abs(float(raw)))
                if weighted is not None:
                    weighted_abs.append(abs(float(weighted)))
                if postclip is not None and postclip != 0:
                    postclip_nonzero.append(abs(float(postclip)))
        result[prefix + "nonzero_reward_abs_mean"] = (
            float(np.mean(raw_nonzero)) if raw_nonzero else 0.0
        )
        result[prefix + "nonzero_reward_abs_p50"] = (
            float(np.percentile(raw_nonzero, 50)) if raw_nonzero else 0.0
        )
        result[prefix + "nonzero_reward_abs_p95"] = (
            float(np.percentile(raw_nonzero, 95)) if raw_nonzero else 0.0
        )
        result[prefix + "weighted_preclip_abs_mean"] = (
            float(np.mean(weighted_abs)) if weighted_abs else 0.0
        )
        result[prefix + "postclip_abs_mean"] = (
            float(np.mean(postclip_nonzero)) if postclip_nonzero else 0.0
        )
        result[prefix + "nonzero_raw_abs_mean"] = result[
            prefix + "nonzero_reward_abs_mean"
        ]
        result[prefix + "nonzero_weighted_abs_mean"] = result[
            prefix + "weighted_preclip_abs_mean"
        ]
        result[prefix + "nonzero_postclip_abs_mean"] = result[
            prefix + "postclip_abs_mean"
        ]
        return result
