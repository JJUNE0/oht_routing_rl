"""Bounded JSONL/W&B diagnostics for contextual rail reward contracts."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


DIAGNOSTIC_SCHEMA_VERSION = "contextual_reward_diagnostic_v16_leading_indicators"
DEFAULT_DIAGNOSTIC_WINDOWS = (
    (0, 1_000, "00000_01000"),
    (10_000, 11_000, "10000_11000"),
    (20_000, 21_000, "20000_21000"),
)


class RunningPearson:
    """Constant-memory Pearson accumulator with safe constant handling."""

    def __init__(self):
        self.count = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.m2_x = 0.0
        self.m2_y = 0.0
        self.covariance = 0.0

    def update(self, x, y):
        x, y = float(x), float(y)
        if not np.isfinite((x, y)).all():
            return
        self.count += 1
        delta_x = x - self.mean_x
        delta_y = y - self.mean_y
        self.mean_x += delta_x / self.count
        self.mean_y += delta_y / self.count
        self.m2_x += delta_x * (x - self.mean_x)
        self.m2_y += delta_y * (y - self.mean_y)
        self.covariance += delta_x * (y - self.mean_y)

    @property
    def correlation(self):
        if self.count < 2:
            return 0.0
        denominator = np.sqrt(max(0.0, self.m2_x) * max(0.0, self.m2_y))
        if denominator <= np.finfo(np.float64).eps:
            return 0.0
        return float(np.clip(self.covariance / denominator, -1.0, 1.0))


class RunningMoments:
    """Welford statistics used only by the diagnostic composite."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def standardize_then_update(self, value):
        value = float(value)
        available = self.count >= 2
        variance = self.m2 / max(1, self.count - 1)
        z_score = (
            (value - self.mean) / np.sqrt(variance)
            if available and variance > np.finfo(np.float64).eps else 0.0
        )
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        return float(z_score), bool(available)


class LeadingIndicatorTracker:
    """Episode-local, bounded leading-indicator and lead/lag diagnostics."""

    ROLLING_HORIZONS = (100, 300, 500, 1_000)
    LEADLAG_HORIZONS = (200, 500, 1_000)
    LEADLAG_FEATURES = (
        "predicted_mean",
        "predicted_p90",
        "predicted_p95",
        "backlog",
        "backlog_delta_300",
        "queue",
        "idle",
        "idle_delta_300",
        "op",
        "op_ema300",
        "transferring",
        "route_ratio_p90",
        "route_ratio_p95",
        "flow_imbalance",
        "composite",
    )
    COMPOSITE_FEATURES = (
        "predicted_p90",
        "backlog",
        "idle",
        "transferring",
        "route_ratio_p90",
        "positive_flow_imbalance",
    )

    def __init__(self):
        self.reset_episode()

    def reset_episode(self):
        self._history = deque(maxlen=max(self.ROLLING_HORIZONS) + 1)
        self._arrival_ticks = deque(maxlen=max(self.ROLLING_HORIZONS))
        self._completion_ticks = deque(maxlen=max(self.ROLLING_HORIZONS))
        self._seen_job_ids = set()
        self._current_job_ids = set()
        self._retired_job_ids = set()
        self._arrival_bootstrapped = False
        self._arrival_available = True
        self._job_id_reuse_count = 0
        self._op_ema = {horizon: None for horizon in (100, 300, 500)}
        self._op_consecutive_above_080 = 0
        self._moments = {
            name: RunningMoments() for name in self.COMPOSITE_FEATURES
        }
        self._correlations = {
            (feature, horizon): RunningPearson()
            for feature in self.LEADLAG_FEATURES
            for horizon in self.LEADLAG_HORIZONS
        }
        self._leadlag_sample_counts = {
            horizon: 0 for horizon in self.LEADLAG_HORIZONS
        }

    @property
    def history_size(self):
        return len(self._history)

    @staticmethod
    def _finite(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        return value if np.isfinite(value) else float(default)

    def _delta(self, name, current, horizon):
        if len(self._history) < horizon:
            return 0.0
        return float(current - self._history[-horizon][name])

    @staticmethod
    def _job_ids(pclient):
        # PClient parses Job.ID from the simulator command ID field. The first
        # snapshot is bootstrapped, and a retired ID that reappears disables
        # arrival availability instead of silently treating reuse as arrival.
        values = []
        for key, job in getattr(pclient, "JOB_DIC", {}).items():
            value = getattr(job, "ID", key)
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value != 0:
                values.append(value)
        return values

    @staticmethod
    def _distribution(values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        result = {
            "mean": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "std": 0.0,
            "top10_mean": 0.0,
            "top50_mean": 0.0,
            "top100_mean": 0.0,
            "fraction_gt_5": 0.0,
            "fraction_gt_10": 0.0,
        }
        if not values.size:
            return result
        result.update({
            "mean": float(values.mean()),
            "max": float(values.max()),
            "std": float(values.std()),
            "fraction_gt_5": float(np.mean(values > 5.0)),
            "fraction_gt_10": float(np.mean(values > 10.0)),
        })
        for percentile in (50, 75, 90, 95, 99):
            result[f"p{percentile}"] = float(
                np.percentile(values, percentile)
            )
        descending = np.sort(values)[::-1]
        for count in (10, 50, 100):
            result[f"top{count}_mean"] = float(
                descending[:min(count, descending.size)].mean()
            )
        return result

    def update(
        self,
        pclient,
        *,
        predicted_oht,
        route_ratio_diagnostics=None,
        idle_reserve_target=200.0,
        idle_reserve_scale=50.0,
    ):
        route = dict(route_ratio_diagnostics or {})
        waiting = self._finite(getattr(pclient, "WaitingCommandCount", 0))
        queued = self._finite(getattr(pclient, "QueuedCommandCount", 0))
        backlog = waiting + queued
        tat = self._finite(getattr(pclient, "TotalTat", 0))
        op = self._finite(getattr(pclient, "TotalOhtOperationRate", 0))
        completed = max(
            0.0, self._finite(getattr(pclient, "CompletedCommandCount", 0))
        )
        transferring = max(
            0.0, self._finite(getattr(pclient, "TransferCommandCount", 0))
        )
        state_counts = {state: 0.0 for state in range(6)}
        for oht in getattr(pclient, "OHT_DIC", {}).values():
            try:
                state = int(getattr(oht, "State", -1))
            except (TypeError, ValueError):
                continue
            if state in state_counts:
                state_counts[state] += 1.0
        idle = state_counts[0]

        job_ids = self._job_ids(pclient)
        current_job_ids = set(job_ids)
        duplicate_job_ids = len(job_ids) - len(current_job_ids)
        if not self._arrival_bootstrapped:
            new_job_count = 0.0
            self._seen_job_ids.update(current_job_ids)
            self._arrival_bootstrapped = True
        else:
            reused = current_job_ids.intersection(self._retired_job_ids)
            if reused:
                self._job_id_reuse_count += len(reused)
                self._arrival_available = False
            new_ids = current_job_ids.difference(self._seen_job_ids)
            new_job_count = float(len(new_ids))
            self._seen_job_ids.update(new_ids)
            self._retired_job_ids.update(
                self._current_job_ids.difference(current_job_ids)
            )
        self._current_job_ids = current_job_ids
        self._arrival_ticks.append(new_job_count)
        self._completion_ticks.append(completed)

        predicted = self._distribution(predicted_oht)
        for horizon in self._op_ema:
            previous = self._op_ema[horizon]
            alpha = 2.0 / (horizon + 1.0)
            self._op_ema[horizon] = (
                op if previous is None else alpha * op + (1.0 - alpha) * previous
            )
        self._op_consecutive_above_080 = (
            self._op_consecutive_above_080 + 1 if op > 0.80 else 0
        )

        deltas = {
            "backlog": {
                horizon: self._delta("backlog", backlog, horizon)
                for horizon in self.ROLLING_HORIZONS
            },
            "queued": {
                horizon: self._delta("queued", queued, horizon)
                for horizon in (100, 300, 500)
            },
            "waiting": {
                horizon: self._delta("waiting", waiting, horizon)
                for horizon in (100, 300, 500)
            },
            "idle": {
                horizon: self._delta("idle", idle, horizon)
                for horizon in self.ROLLING_HORIZONS
            },
            "op": {
                horizon: self._delta("op", op, horizon)
                for horizon in (100, 300, 500)
            },
            "transferring": {
                300: self._delta("transferring", transferring, 300)
            },
            "move_to_unload": {
                300: self._delta("move_to_unload", state_counts[4], 300)
            },
        }

        flow = {}
        for horizon in self.ROLLING_HORIZONS:
            arrival_count = float(sum(list(self._arrival_ticks)[-horizon:]))
            completion_count = float(
                sum(list(self._completion_ticks)[-horizon:])
            )
            imbalance_count = arrival_count - completion_count
            flow[horizon] = {
                "arrival_count": arrival_count,
                "completion_count": completion_count,
                "arrival_rate": arrival_count / horizon,
                "completion_rate": completion_count / horizon,
                "imbalance_count": imbalance_count,
                "imbalance_rate": imbalance_count / max(completion_count, 1.0),
            }

        route_available = bool(route.get("lead/route_ratio/available", 0.0))
        route_p90 = self._finite(route.get("lead/route_ratio/p90", 0.0))
        route_p95 = self._finite(route.get("lead/route_ratio/p95", 0.0))
        composite_inputs = {
            "predicted_p90": predicted["p90"],
            "backlog": backlog,
            "idle": idle,
            "transferring": transferring,
            "route_ratio_p90": route_p90,
            "positive_flow_imbalance": max(
                0.0, flow[300]["imbalance_count"]
            ),
        }
        z_scores = {}
        z_available = []
        for name, value in composite_inputs.items():
            if name == "route_ratio_p90" and not route_available:
                z_score, available = 0.0, False
            else:
                z_score, available = (
                    self._moments[name].standardize_then_update(value)
                )
            z_scores[name] = z_score
            z_available.append(available)
        composite_available = bool(
            route_available and self._arrival_available and all(z_available)
        )
        composite = (
            z_scores["predicted_p90"]
            + z_scores["backlog"]
            - z_scores["idle"]
            + z_scores["transferring"]
            + z_scores["route_ratio_p90"]
            + z_scores["positive_flow_imbalance"]
            if composite_available else 0.0
        )

        snapshot = {
            "tat": tat,
            "op_target": op,
            "queue_target": queued,
            "idle_target": idle,
            "backlog": backlog,
            "queued": queued,
            "waiting": waiting,
            "idle": idle,
            "op": op,
            "transferring": transferring,
            "move_to_unload": state_counts[4],
            "predicted_mean": predicted["mean"],
            "predicted_p90": predicted["p90"],
            "predicted_p95": predicted["p95"],
            "backlog_delta_300": deltas["backlog"][300],
            "idle_delta_300": deltas["idle"][300],
            "op_ema300": float(self._op_ema[300]),
            "route_ratio_p90": route_p90 if route_available else None,
            "route_ratio_p95": route_p95 if route_available else None,
            "flow_imbalance": flow[300]["imbalance_count"]
            if self._arrival_available else None,
            "composite": composite if composite_available else None,
        }
        for horizon in self.LEADLAG_HORIZONS:
            if len(self._history) < horizon:
                continue
            past = self._history[-horizon]
            future_tat_delta = tat - past["tat"]
            self._leadlag_sample_counts[horizon] += 1
            for feature in self.LEADLAG_FEATURES:
                past_value = past.get(feature)
                if past_value is not None:
                    self._correlations[(feature, horizon)].update(
                        past_value, future_tat_delta
                    )
        self._history.append(snapshot)

        result = {
            "lead/backlog/value": backlog,
            "lead/queued/value": queued,
            "lead/waiting/value": waiting,
            "lead/idle/value": idle,
            "lead/idle/reserve_signal": float(np.clip(
                max(0.0, float(idle_reserve_target) - idle)
                / float(idle_reserve_scale),
                0.0,
                1.0,
            )),
            "lead/idle/below_target": float(
                idle < float(idle_reserve_target)
            ),
            "lead/op/value": op,
            "lead/op/above_078": float(op > 0.78),
            "lead/op/above_080": float(op > 0.80),
            "lead/op/consecutive_above_080": float(
                self._op_consecutive_above_080
            ),
            "lead/oht/idle": idle,
            "lead/oht/move_to_load": state_counts[2],
            "lead/oht/loading": state_counts[3],
            "lead/oht/move_to_unload": state_counts[4],
            "lead/oht/unloading": state_counts[5],
            "lead/oht/transferring": transferring,
            "lead/oht/transferring_delta_300": deltas["transferring"][300],
            "lead/oht/move_to_unload_delta_300": (
                deltas["move_to_unload"][300]
            ),
            "lead/flow/arrival_available": float(self._arrival_available),
            "lead/flow/new_job_count": new_job_count,
            "lead/flow/duplicate_job_id_count": float(duplicate_job_ids),
            "lead/flow/job_id_reuse_count": float(self._job_id_reuse_count),
            "lead/composite_pressure": composite,
            "lead/composite_available": float(composite_available),
        }
        for name, values in deltas.items():
            if name in {"transferring", "move_to_unload"}:
                continue
            for horizon, value in values.items():
                result[f"lead/{name}/delta_{horizon}"] = value
        for horizon, value in self._op_ema.items():
            result[f"lead/op/ema_{horizon}"] = float(value)
        for name, value in predicted.items():
            result[f"lead/predicted_oht/{name}"] = value
        result["lead/predicted_oht/available"] = float(
            np.asarray(predicted_oht).size > 0
        )
        result.update(route)
        for horizon, values in flow.items():
            for name, value in values.items():
                result[f"lead/flow/{name}_{horizon}"] = value
        for horizon in self.LEADLAG_HORIZONS:
            result[f"leadlag/sample_count_{horizon}"] = float(
                self._leadlag_sample_counts[horizon]
            )
            for feature in self.LEADLAG_FEATURES:
                result[
                    f"leadlag/{feature}_vs_future_tat_{horizon}"
                ] = self._correlations[(feature, horizon)].correlation
        if not np.isfinite(tuple(result.values())).all():
            raise ValueError("leading diagnostics contain NaN or Inf")
        return result


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
