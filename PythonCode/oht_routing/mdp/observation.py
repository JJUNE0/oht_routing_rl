"""Physical-once contextual observation construction for per-rail TD7.

This module intentionally contains no model, action, reward, replay, or learner
logic. Local normalization is performed on the physical rail array before any
controlled-center or neighbor gather so a rail contributes exactly once per
environment step.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from oht_routing.mdp.topology import ContextualTopology, TOPOLOGY_VERSION
from oht_routing.version import (
    CONTEXTUAL_VERSION,
    is_compatible_contextual_version,
)


LOCAL_PHYSICAL_FEATURE_NAMES = (
    "free_flow_time_s",
    "port_count",
    "incoming_degree",
    "outgoing_degree",
    "oht_density",
    "predicted_oht_count",
    "next_10_route_oht_count",
    "reservation_port_count",
    "idle_count",
    "stage_count",
    "move_to_load_count",
    "loading_count",
    "move_to_unload_count",
    "unloading_count",
    "stop_time_sum",
    "stopped_oht_count",
)
GLOBAL_FEATURE_NAMES = (
    "total_tat_s",
    "operation_rate",
    "queued_ratio",
    "waiting_ratio",
    "transferring_ratio",
    "mean_reassign",
    "idle_oht_ratio",
    "stage_oht_ratio",
    "move_to_load_ratio",
    "loading_ratio",
    "move_to_unload_ratio",
    "unloading_ratio",
    "stopped_oht_ratio",
    "mean_stop_time_s",
    "backlog_delta_60s",
    "tat_delta_60s",
    "completion_rate_60s",
)
RELATION_FEATURE_NAMES = (
    "directed_hop",
    "cumulative_free_flow_time_s",
)

LOCAL_PHYSICAL_DIM = len(LOCAL_PHYSICAL_FEATURE_NAMES)
GLOBAL_DIM = len(GLOBAL_FEATURE_NAMES)
RELATION_DIM = len(RELATION_FEATURE_NAMES)
TREND_WINDOW_SECONDS = 60.0
VALID_OHT_STATES = frozenset(range(6))


class ObservationContractError(RuntimeError):
    """Raised when runtime observation data violates the fixed contract."""


@dataclass(frozen=True)
class ObservationNormalizerConfig:
    """Normalizer lifecycle shared by local and global dynamic features."""

    freeze_after_env_steps: int = 10_000
    epsilon: float = 1e-6
    clip: float | None = 10.0

    def __post_init__(self):
        if int(self.freeze_after_env_steps) < 0:
            raise ValueError("freeze_after_env_steps must be non-negative")
        if not np.isfinite(self.epsilon) or float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.clip is not None and (
            not np.isfinite(self.clip) or float(self.clip) <= 0.0
        ):
            raise ValueError("clip must be finite and positive when provided")


class RunningFeatureNormalizer:
    """Deterministic batch Welford normalizer with explicit freeze state."""

    def __init__(self, dim: int, *, epsilon: float = 1e-6, clip: float | None = 10.0):
        self.dim = int(dim)
        self.epsilon = float(epsilon)
        self.clip = None if clip is None else float(clip)
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)
        self.count = 0
        self.update_calls = 0
        self.frozen = False

    def _validate(self, values, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.dim:
            raise ObservationContractError(
                f"{name} shape must be [N, {self.dim}], got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ObservationContractError(f"{name} contains NaN or Inf")
        return array

    def normalize(self, values, *, name: str = "features") -> np.ndarray:
        array = self._validate(values, name=name)
        if self.count == 0:
            result = array.copy()
        else:
            variance = self.m2 / float(self.count)
            scale = np.sqrt(np.maximum(variance, self.epsilon))
            result = (array - self.mean) / scale
        if self.clip is not None:
            result = np.clip(result, -self.clip, self.clip)
        return np.ascontiguousarray(result.astype(np.float32))

    def update(self, values, *, name: str = "features") -> None:
        array = self._validate(values, name=name)
        if self.frozen:
            return
        batch_count = int(array.shape[0])
        if batch_count == 0:
            return
        batch_mean = array.mean(axis=0)
        centered = array - batch_mean
        batch_m2 = np.square(centered).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (batch_count / total)
            self.m2 = (
                self.m2
                + batch_m2
                + np.square(delta) * (self.count * batch_count / total)
            )
            self.count = total
        self.update_calls += 1

    def freeze(self) -> None:
        self.frozen = True

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
            "count": self.count,
            "update_calls": self.update_calls,
            "frozen": self.frozen,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["dim"]) != self.dim:
            raise ObservationContractError(
                f"normalizer dim mismatch: saved={state['dim']}, current={self.dim}"
            )
        saved_epsilon = float(state.get("epsilon", self.epsilon))
        saved_clip = state.get("clip", self.clip)
        saved_clip = None if saved_clip is None else float(saved_clip)
        if saved_epsilon != self.epsilon:
            raise ObservationContractError(
                "normalizer epsilon mismatch: "
                f"saved={saved_epsilon}, current={self.epsilon}"
            )
        if saved_clip != self.clip:
            raise ObservationContractError(
                "normalizer clip mismatch: "
                f"saved={saved_clip}, current={self.clip}"
            )
        mean = np.asarray(state["mean"], dtype=np.float64)
        m2 = np.asarray(state["m2"], dtype=np.float64)
        if mean.shape != (self.dim,) or m2.shape != (self.dim,):
            raise ObservationContractError("invalid saved normalizer vector shape")
        if not np.isfinite(mean).all() or not np.isfinite(m2).all():
            raise ObservationContractError("saved normalizer contains NaN or Inf")
        if (m2 < 0.0).any():
            raise ObservationContractError("saved normalizer m2 must be non-negative")
        count = int(state["count"])
        update_calls = int(state.get("update_calls", 0))
        if count < 0 or update_calls < 0:
            raise ObservationContractError(
                "saved normalizer counts must be non-negative"
            )
        self.mean = mean.copy()
        self.m2 = m2.copy()
        self.count = count
        self.update_calls = update_calls
        self.frozen = bool(state["frozen"])


@dataclass(frozen=True)
class ContextualObservationBatch:
    center_local: np.ndarray
    incoming_local: np.ndarray
    outgoing_local: np.ndarray
    center_rail_index: np.ndarray
    incoming_rail_indices: np.ndarray
    outgoing_rail_indices: np.ndarray
    incoming_relation: np.ndarray
    outgoing_relation: np.ndarray
    global_state: np.ndarray
    previous_applied_action: np.ndarray
    controlled_rail_ids: np.ndarray
    topology_hash: str
    mapping_hash: str
    physical_local_raw: np.ndarray | None = None
    global_raw: np.ndarray | None = None


def _cache_identity(cache_path: str | Path) -> tuple[str, str, str]:
    path = Path(cache_path)
    if not path.is_file():
        raise ObservationContractError(f"topology cache does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as cache:
            return (
                str(cache["topology_version"].item()),
                str(cache["topology_hash"].item()),
                str(cache["mapping_hash"].item()),
            )
    except (KeyError, OSError, ValueError) as error:
        raise ObservationContractError(
            f"invalid topology cache identity: path={path}, error={error}"
        ) from error


class ContextualObservationBuilder:
    """Build fixed contextual observations from one physical snapshot."""

    def __init__(
        self,
        topology: ContextualTopology,
        *,
        cache_path: str | Path,
        normalizer_config: ObservationNormalizerConfig | None = None,
    ):
        self.topology = topology
        self.config = normalizer_config or ObservationNormalizerConfig()
        version, topology_hash, mapping_hash = _cache_identity(cache_path)
        if version != TOPOLOGY_VERSION:
            raise ObservationContractError(
                f"topology cache version mismatch: runtime={TOPOLOGY_VERSION}, "
                f"cache={version}"
            )
        if topology_hash != topology.topology_hash:
            raise ObservationContractError(
                "topology hash mismatch: "
                f"runtime={topology.topology_hash}, cache={topology_hash}"
            )
        if mapping_hash != topology.mapping_hash:
            raise ObservationContractError(
                "mapping hash mismatch: "
                f"runtime={topology.mapping_hash}, cache={mapping_hash}"
            )

        self.local_normalizer = RunningFeatureNormalizer(
            LOCAL_PHYSICAL_DIM,
            epsilon=self.config.epsilon,
            clip=self.config.clip,
        )
        self.global_normalizer = RunningFeatureNormalizer(
            GLOBAL_DIM, epsilon=self.config.epsilon, clip=self.config.clip
        )
        self.relation_normalizer = RunningFeatureNormalizer(
            RELATION_DIM, epsilon=self.config.epsilon, clip=self.config.clip
        )
        self.env_steps = 0
        self._trend_history: deque[tuple[float, float, float, float]] = deque()
        self.last_diagnostics: dict[str, float] = {}
        self._physical_distance_mm: np.ndarray | None = None

        self._rail_id_to_physical_row = {
            int(rail_id): row
            for row, rail_id in enumerate(topology.all_rail_ids.tolist())
        }
        self._incoming_physical_rows = self._neighbor_rows(
            topology.incoming_neighbor_ids, "incoming"
        )
        self._outgoing_physical_rows = self._neighbor_rows(
            topology.outgoing_neighbor_ids, "outgoing"
        )
        relation_raw = np.concatenate(
            (
                self._relation_raw(
                    topology.incoming_hops, topology.incoming_travel_time
                ).reshape(-1, RELATION_DIM),
                self._relation_raw(
                    topology.outgoing_hops, topology.outgoing_travel_time
                ).reshape(-1, RELATION_DIM),
            ),
            axis=0,
        )
        self.relation_normalizer.update(relation_raw, name="relation_raw")
        self.relation_normalizer.freeze()
        self._incoming_relation = self.relation_normalizer.normalize(
            self._relation_raw(
                topology.incoming_hops, topology.incoming_travel_time
            ).reshape(-1, RELATION_DIM),
            name="incoming_relation_raw",
        ).reshape(topology.incoming_hops.shape + (RELATION_DIM,))
        self._outgoing_relation = self.relation_normalizer.normalize(
            self._relation_raw(
                topology.outgoing_hops, topology.outgoing_travel_time
            ).reshape(-1, RELATION_DIM),
            name="outgoing_relation_raw",
        ).reshape(topology.outgoing_hops.shape + (RELATION_DIM,))
        self._incoming_relation = np.ascontiguousarray(self._incoming_relation)
        self._outgoing_relation = np.ascontiguousarray(self._outgoing_relation)

    @staticmethod
    def _relation_raw(hops, travel_time) -> np.ndarray:
        result = np.stack(
            (
                np.asarray(hops, dtype=np.float64),
                np.asarray(travel_time, dtype=np.float64),
            ),
            axis=-1,
        )
        if not np.isfinite(result).all():
            raise ObservationContractError("topology relation contains NaN or Inf")
        return result

    def _neighbor_rows(self, neighbor_ids, direction: str) -> np.ndarray:
        ids = np.asarray(neighbor_ids, dtype=np.int64)
        try:
            rows = np.asarray(
                [
                    self._rail_id_to_physical_row[int(rail_id)]
                    for rail_id in ids.reshape(-1)
                ],
                dtype=np.int64,
            ).reshape(ids.shape)
        except KeyError as error:
            raise ObservationContractError(
                f"{direction} neighbor references unknown physical rail ID: {error}"
            ) from error
        return np.ascontiguousarray(rows)

    @staticmethod
    def _mean_reassign(pclient) -> float:
        jobs = list(getattr(pclient, "JOB_DIC", {}).values())
        if not jobs:
            return 0.0
        reassign = []
        for job in jobs:
            value = float(getattr(job, "ReAssignCount", 0) or 0)
            if not np.isfinite(value) or value < 0.0:
                raise ObservationContractError(
                    "job ReAssignCount must be finite and non-negative, "
                    f"got {value}"
                )
            reassign.append(value)
        return float(np.mean(reassign)) if reassign else 0.0

    @staticmethod
    def _finite_nonnegative(value, *, name: str) -> float:
        result = float(value or 0.0)
        if not np.isfinite(result) or result < 0.0:
            raise ObservationContractError(
                f"{name} must be finite and non-negative, got {result}"
            )
        return result

    @staticmethod
    def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
        if left.size < 2 or right.size < 2:
            return 0.0
        left_std = float(left.std())
        right_std = float(right.std())
        if (
            not np.isfinite(left_std)
            or not np.isfinite(right_std)
            or left_std <= 0.0
            or right_std <= 0.0
        ):
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    @staticmethod
    def _pearson_available(left: np.ndarray, right: np.ndarray) -> float:
        return float(
            left.size >= 2
            and float(left.std()) > 0.0
            and float(right.std()) > 0.0
        )

    def reset_episode(self) -> None:
        """Clear episode-local 60-second trend history."""
        self._trend_history.clear()

    def _trend_features(
        self,
        *,
        sim_time_s: float,
        backlog: float,
        total_tat_s: float,
        completed_count: float,
    ) -> tuple[float, float, float]:
        now = float(sim_time_s)
        if not np.isfinite(now):
            raise ObservationContractError(
                f"simulation time must be finite, got {now}"
            )
        if self._trend_history and now < self._trend_history[-1][0]:
            self._trend_history.clear()
        if self._trend_history and now == self._trend_history[-1][0]:
            self._trend_history.pop()
        entry = (
            now,
            self._finite_nonnegative(backlog, name="trend backlog"),
            self._finite_nonnegative(total_tat_s, name="trend total TAT"),
            self._finite_nonnegative(
                completed_count, name="trend completed count"
            ),
        )
        self._trend_history.append(entry)
        cutoff = now - TREND_WINDOW_SECONDS
        # Retain the newest sample at or before the 60-second cutoff as the
        # level-delta anchor. Packet completion counts are summed only over
        # samples strictly newer than the cutoff.
        while (
            len(self._trend_history) >= 2
            and self._trend_history[1][0] <= cutoff
        ):
            self._trend_history.popleft()
        if self._trend_history[0][0] > cutoff:
            return 0.0, 0.0, 0.0
        _, anchor_backlog, anchor_tat, _ = self._trend_history[0]
        completion_rate = sum(
            item[3] for item in self._trend_history if item[0] > cutoff
        ) / TREND_WINDOW_SECONDS
        return (
            float(entry[1] - anchor_backlog),
            float(entry[2] - anchor_tat),
            float(completion_rate),
        )

    def diagnostics(self) -> dict[str, float]:
        return dict(self.last_diagnostics)

    @property
    def physical_distance_mm(self) -> np.ndarray:
        """Static float64 rail distances used for exact replay density rebuilds."""
        if self._physical_distance_mm is None:
            raise ObservationContractError(
                "physical rail distances are unavailable before the first "
                "raw observation build"
            )
        return self._physical_distance_mm

    def build_raw(
        self,
        pclient,
        *,
        next_10_route_oht_count: Mapping[int, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        rail_lines = getattr(pclient, "RAILLINE_DIC", {})
        runtime_ohts = getattr(pclient, "OHT_DIC", {})
        ohts = {int(oht_id): oht for oht_id, oht in runtime_ohts.items()}
        if len(ohts) != len(runtime_ohts):
            raise ObservationContractError(
                "OHT_DIC contains duplicate IDs after integer canonicalization"
            )
        actual_ids = {int(rail_id) for rail_id in rail_lines}
        expected_ids = set(self._rail_id_to_physical_row)
        if actual_ids != expected_ids:
            raise ObservationContractError(
                "runtime physical rail IDs differ from audited topology: "
                f"missing={sorted(expected_ids - actual_ids)[:20]}, "
                f"unexpected={sorted(actual_ids - expected_ids)[:20]}"
            )

        physical_local = np.empty(
            (len(self.topology.all_rail_ids), LOCAL_PHYSICAL_DIM),
            dtype=np.float64,
        )
        physical_distance_mm = np.empty(
            len(self.topology.all_rail_ids), dtype=np.float64
        )
        global_state_counts = np.zeros(6, dtype=np.float64)
        global_stop_time_sum = 0.0
        global_stopped_oht_count = 0.0
        for oht_id, oht in ohts.items():
            state = int(getattr(oht, "State", -1))
            if state not in VALID_OHT_STATES:
                raise ObservationContractError(
                    f"OHT {oht_id} has unsupported state {state}"
                )
            stop_time = self._finite_nonnegative(
                getattr(oht, "StopTime", 0.0),
                name=f"OHT {oht_id} StopTime",
            )
            global_state_counts[state] += 1.0
            global_stop_time_sum += stop_time
            global_stopped_oht_count += float(stop_time > 0.0)

        oht_placement: dict[int, int] = {}
        for physical_row, rail_id_value in enumerate(self.topology.all_rail_ids):
            rail_id = int(rail_id_value)
            rail = rail_lines[rail_id]
            rail_oht_ids = tuple(int(value) for value in getattr(rail, "OhtList"))
            state_counts = np.zeros(6, dtype=np.float64)
            stop_time_sum = 0.0
            stopped_oht_count = 0.0
            for oht_id in rail_oht_ids:
                if oht_id not in ohts:
                    raise ObservationContractError(
                        f"rail {rail_id} references unknown OHT {oht_id}"
                    )
                previous_rail = oht_placement.get(oht_id)
                if previous_rail is not None:
                    raise ObservationContractError(
                        f"OHT {oht_id} is listed more than once: "
                        f"rail={rail_id}, previous_rail={previous_rail}"
                    )
                oht_placement[oht_id] = rail_id
                oht = ohts[oht_id]
                state = int(getattr(oht, "State", -1))
                if state not in VALID_OHT_STATES:
                    raise ObservationContractError(
                        f"OHT {oht_id} has unsupported state {state}"
                    )
                stop_time = self._finite_nonnegative(
                    getattr(oht, "StopTime", 0.0),
                    name=f"OHT {oht_id} StopTime",
                )
                state_counts[state] += 1.0
                stop_time_sum += stop_time
                stopped_oht_count += float(stop_time > 0.0)
            distance_mm = self._finite_nonnegative(
                getattr(rail, "Distance", 0.0),
                name=f"rail {rail_id} Distance",
            )
            if distance_mm <= 0.0:
                raise ObservationContractError(
                    f"rail {rail_id} Distance must be positive"
                )
            physical_distance_mm[physical_row] = distance_mm
            density_per_meter = len(rail_oht_ids) / (distance_mm / 1_000.0)
            free_flow_time_s = self._finite_nonnegative(
                getattr(rail, "DistancePerVelocity"),
                name=f"rail {rail_id} DistancePerVelocity",
            )
            port_count = self._finite_nonnegative(
                getattr(rail, "PortCount"),
                name=f"rail {rail_id} PortCount",
            )
            predicted_oht_count = self._finite_nonnegative(
                getattr(rail, "PredictedOHTCount"),
                name=f"rail {rail_id} PredictedOHTCount",
            )
            route_oht_count = self._finite_nonnegative(
                next_10_route_oht_count.get(rail_id, 0.0),
                name=f"rail {rail_id} next_10_route_oht_count",
            )
            reservation_port_count = self._finite_nonnegative(
                getattr(rail, "ReservationPortCount"),
                name=f"rail {rail_id} ReservationPortCount",
            )
            physical_local[physical_row] = (
                free_flow_time_s,
                port_count,
                float(len(getattr(rail, "LevelJoiningLineIDList"))),
                float(len(getattr(rail, "DivergingLineIDList"))),
                float(density_per_meter),
                predicted_oht_count,
                route_oht_count,
                reservation_port_count,
                *state_counts.tolist(),
                float(stop_time_sum),
                float(stopped_oht_count),
            )

        if self._physical_distance_mm is None:
            physical_distance_mm = np.ascontiguousarray(physical_distance_mm)
            physical_distance_mm.setflags(write=False)
            self._physical_distance_mm = physical_distance_mm
        elif not np.array_equal(
            self._physical_distance_mm, physical_distance_mm
        ):
            raise ObservationContractError(
                "runtime physical rail Distance changed after initialization"
            )

        placed_oht_ids = set(oht_placement)
        expected_oht_ids = set(ohts)
        if placed_oht_ids != expected_oht_ids:
            raise ObservationContractError(
                "rail OhtList placement does not exactly cover OHT_DIC: "
                f"missing={sorted(expected_oht_ids - placed_oht_ids)[:20]}, "
                f"unexpected={sorted(placed_oht_ids - expected_oht_ids)[:20]}"
            )

        total_oht_count = len(ohts)
        denominator = float(max(1, total_oht_count))
        total_tat_s = self._finite_nonnegative(
            getattr(pclient, "TotalTat"), name="total TAT"
        )
        operation_rate = self._finite_nonnegative(
            getattr(pclient, "TotalOhtOperationRate"),
            name="OHT operation rate",
        )
        queued = self._finite_nonnegative(
            getattr(pclient, "QueuedCommandCount", 0), name="queued count"
        )
        waiting = self._finite_nonnegative(
            getattr(pclient, "WaitingCommandCount", 0), name="waiting count"
        )
        transferring = self._finite_nonnegative(
            getattr(pclient, "TransferCommandCount", 0),
            name="transferring count",
        )
        completed = self._finite_nonnegative(
            getattr(pclient, "CompletedCommandCount", 0),
            name="completed count",
        )
        backlog_delta, tat_delta, completion_rate = self._trend_features(
            sim_time_s=float(getattr(pclient, "SimTime", 0.0)),
            backlog=queued + waiting,
            total_tat_s=total_tat_s,
            completed_count=completed,
        )
        global_raw = np.asarray(
            (
                total_tat_s,
                operation_rate,
                queued / denominator,
                waiting / denominator,
                transferring / denominator,
                self._mean_reassign(pclient),
                *(global_state_counts / denominator).tolist(),
                global_stopped_oht_count / denominator,
                global_stop_time_sum / denominator,
                backlog_delta,
                tat_delta,
                completion_rate,
            ),
            dtype=np.float64,
        )
        predicted = physical_local[:, 5]
        route_ahead = physical_local[:, 6]
        union_nonzero = (predicted > 0.0) | (route_ahead > 0.0)
        predicted_union = predicted[union_nonzero]
        route_ahead_union = route_ahead[union_nonzero]
        self.last_diagnostics = {
            "observation/predicted_route10_pearson": self._safe_pearson(
                predicted, route_ahead
            ),
            "observation/predicted_route10_mae": float(
                np.abs(predicted - route_ahead).mean()
            ),
            "observation/predicted_route10_nonzero_agreement": float(
                ((predicted > 0.0) == (route_ahead > 0.0)).mean()
            ),
            "observation/predicted_route10_both_nonzero": float(
                ((predicted > 0.0) & (route_ahead > 0.0)).mean()
            ),
            "observation/predicted_route10_pearson_available": (
                self._pearson_available(predicted, route_ahead)
            ),
            "observation/predicted_route10_union_nonzero_pearson": (
                self._safe_pearson(predicted_union, route_ahead_union)
            ),
            "observation/predicted_route10_union_nonzero_pearson_available": (
                self._pearson_available(predicted_union, route_ahead_union)
            ),
            "observation/predicted_route10_union_nonzero_ratio": float(
                union_nonzero.mean()
            ),
        }
        if not np.isfinite(physical_local).all():
            raise ObservationContractError("physical_local_raw contains NaN or Inf")
        if not np.isfinite(global_raw).all():
            raise ObservationContractError("global_raw contains NaN or Inf")
        return physical_local, global_raw

    def build(
        self,
        pclient,
        *,
        next_10_route_oht_count: Mapping[int, float],
        previous_applied_action=None,
    ) -> ContextualObservationBatch:
        physical_raw, global_raw = self.build_raw(
            pclient,
            next_10_route_oht_count=next_10_route_oht_count,
        )

        # Contractual order: normalize the whole physical snapshot using the
        # pre-step statistics, gather, then update each dynamic normalizer once.
        physical_norm = self.local_normalizer.normalize(
            physical_raw, name="physical_local_raw"
        )
        global_norm = self.global_normalizer.normalize(
            global_raw, name="global_raw"
        )[0]

        center = physical_norm[self.topology.controlled_row_to_physical_index]
        incoming = physical_norm[self._incoming_physical_rows]
        outgoing = physical_norm[self._outgoing_physical_rows]

        controlled_count = len(self.topology.controlled_rail_ids)
        if previous_applied_action is None:
            previous_action = np.zeros((controlled_count, 1), np.float32)
        else:
            previous_action = np.asarray(
                previous_applied_action, dtype=np.float32
            ).reshape(-1, 1)
            if previous_action.shape != (controlled_count, 1):
                raise ObservationContractError(
                    "previous_applied_action shape mismatch: "
                    f"actual={previous_action.shape}, "
                    f"expected=({controlled_count}, 1)"
                )
            if not np.isfinite(previous_action).all():
                raise ObservationContractError(
                    "previous_applied_action contains NaN or Inf"
                )
            if (np.abs(previous_action) > 1.0 + 1e-6).any():
                raise ObservationContractError(
                    "previous_applied_action must be in [-1, 1]"
                )

        self.local_normalizer.update(physical_raw, name="physical_local_raw")
        self.global_normalizer.update(global_raw, name="global_raw")
        self.env_steps += 1
        if self.env_steps >= int(self.config.freeze_after_env_steps):
            self.local_normalizer.freeze()
            self.global_normalizer.freeze()

        batch = ContextualObservationBatch(
            center_local=np.ascontiguousarray(center),
            incoming_local=np.ascontiguousarray(incoming),
            outgoing_local=np.ascontiguousarray(outgoing),
            center_rail_index=np.ascontiguousarray(
                self.topology.controlled_row_to_physical_index.copy()
            ),
            incoming_rail_indices=np.ascontiguousarray(
                self._incoming_physical_rows.copy()
            ),
            outgoing_rail_indices=np.ascontiguousarray(
                self._outgoing_physical_rows.copy()
            ),
            incoming_relation=self._incoming_relation,
            outgoing_relation=self._outgoing_relation,
            global_state=np.ascontiguousarray(global_norm),
            previous_applied_action=np.ascontiguousarray(previous_action),
            controlled_rail_ids=np.ascontiguousarray(
                self.topology.controlled_rail_ids.copy()
            ),
            topology_hash=self.topology.topology_hash,
            mapping_hash=self.topology.mapping_hash,
            physical_local_raw=np.ascontiguousarray(
                physical_raw.astype(np.float32)
            ),
            global_raw=np.ascontiguousarray(global_raw.astype(np.float32)),
        )
        self._validate_output(batch)
        return batch

    def _validate_output(self, batch: ContextualObservationBatch) -> None:
        controlled_count = len(self.topology.controlled_rail_ids)
        neighbor_count = int(self.topology.incoming_neighbor_ids.shape[1])
        expected = {
            "center_local": (controlled_count, LOCAL_PHYSICAL_DIM),
            "incoming_local": (
                controlled_count, neighbor_count, LOCAL_PHYSICAL_DIM
            ),
            "outgoing_local": (
                controlled_count, neighbor_count, LOCAL_PHYSICAL_DIM
            ),
            "center_rail_index": (controlled_count,),
            "incoming_rail_indices": (controlled_count, neighbor_count),
            "outgoing_rail_indices": (controlled_count, neighbor_count),
            "incoming_relation": (
                controlled_count, neighbor_count, RELATION_DIM
            ),
            "outgoing_relation": (
                controlled_count, neighbor_count, RELATION_DIM
            ),
            "global_state": (GLOBAL_DIM,),
            "previous_applied_action": (controlled_count, 1),
            "controlled_rail_ids": (controlled_count,),
        }
        for name, shape in expected.items():
            array = getattr(batch, name)
            if array.shape != shape:
                raise ObservationContractError(
                    f"{name} shape mismatch: actual={array.shape}, expected={shape}"
                )
            if not array.flags.c_contiguous:
                raise ObservationContractError(f"{name} is not C-contiguous")
            if not np.isfinite(array).all():
                raise ObservationContractError(f"{name} contains NaN or Inf")
        index_names = (
            "center_rail_index",
            "incoming_rail_indices",
            "outgoing_rail_indices",
        )
        for name in index_names:
            array = getattr(batch, name)
            if not np.issubdtype(array.dtype, np.integer):
                raise ObservationContractError(f"{name} must be integral")
            if array.size and (
                int(array.min()) < 0
                or int(array.max()) >= len(self.topology.all_rail_ids)
            ):
                raise ObservationContractError(
                    f"{name} contains an out-of-range physical Rail index"
                )
        raw_expected = {
            "physical_local_raw": (
                len(self.topology.all_rail_ids), LOCAL_PHYSICAL_DIM
            ),
            "global_raw": (GLOBAL_DIM,),
        }
        for name, shape in raw_expected.items():
            array = getattr(batch, name)
            if array is None or array.shape != shape:
                raise ObservationContractError(
                    f"{name} shape mismatch: actual="
                    f"{None if array is None else array.shape}, expected={shape}"
                )
            if not array.flags.c_contiguous or not np.isfinite(array).all():
                raise ObservationContractError(
                    f"{name} must be finite and C-contiguous"
                )

    def save_normalizers(
        self, path: str | Path, *, require_frozen: bool = False
    ) -> Path:
        target = Path(path)
        if require_frozen and not (
            self.local_normalizer.frozen
            and self.global_normalizer.frozen
            and self.local_normalizer.count > 0
            and self.global_normalizer.count > 0
        ):
            raise ObservationContractError(
                "refusing to save a warm-up bypass snapshot before both "
                "state normalizers are populated and frozen"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        local = self.local_normalizer.state_dict()
        global_state = self.global_normalizer.state_dict()
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                version=np.asarray(CONTEXTUAL_VERSION),
                topology_version=np.asarray(TOPOLOGY_VERSION),
                topology_hash=np.asarray(self.topology.topology_hash),
                mapping_hash=np.asarray(self.topology.mapping_hash),
                local_physical_feature_names=np.asarray(
                    LOCAL_PHYSICAL_FEATURE_NAMES
                ),
                global_feature_names=np.asarray(GLOBAL_FEATURE_NAMES),
                relation_feature_names=np.asarray(RELATION_FEATURE_NAMES),
                local_physical_dim=np.asarray(
                    LOCAL_PHYSICAL_DIM, dtype=np.int64
                ),
                global_dim=np.asarray(GLOBAL_DIM, dtype=np.int64),
                relation_dim=np.asarray(RELATION_DIM, dtype=np.int64),
                env_steps=np.asarray(self.env_steps, dtype=np.int64),
                local_epsilon=np.asarray(local["epsilon"], dtype=np.float64),
                local_clip_is_none=np.asarray(local["clip"] is None),
                local_clip=np.asarray(
                    0.0 if local["clip"] is None else local["clip"],
                    dtype=np.float64,
                ),
                local_mean=local["mean"],
                local_m2=local["m2"],
                local_count=np.asarray(local["count"], dtype=np.int64),
                local_update_calls=np.asarray(
                    local["update_calls"], dtype=np.int64
                ),
                local_frozen=np.asarray(local["frozen"]),
                global_epsilon=np.asarray(
                    global_state["epsilon"], dtype=np.float64
                ),
                global_clip_is_none=np.asarray(
                    global_state["clip"] is None
                ),
                global_clip=np.asarray(
                    0.0
                    if global_state["clip"] is None
                    else global_state["clip"],
                    dtype=np.float64,
                ),
                global_mean=global_state["mean"],
                global_m2=global_state["m2"],
                global_count=np.asarray(
                    global_state["count"], dtype=np.int64
                ),
                global_update_calls=np.asarray(
                    global_state["update_calls"], dtype=np.int64
                ),
                global_frozen=np.asarray(global_state["frozen"]),
            )
        temporary.replace(target)
        return target

    def load_normalizers(
        self, path: str | Path, *, require_frozen: bool = False
    ) -> None:
        source = Path(path)
        if not source.is_file():
            raise ObservationContractError(
                f"state normalizer snapshot does not exist: {source}"
            )
        try:
            with np.load(source, allow_pickle=False) as saved:
                saved_version = str(saved["version"].item())
                if not is_compatible_contextual_version(saved_version):
                    raise ObservationContractError(
                        "saved state normalizer version mismatch: "
                        f"saved={saved_version!r}, current={CONTEXTUAL_VERSION!r}"
                    )
                expected_scalars = {
                    "topology_version": TOPOLOGY_VERSION,
                    "topology_hash": self.topology.topology_hash,
                    "mapping_hash": self.topology.mapping_hash,
                }
                for key, expected in expected_scalars.items():
                    actual = str(saved[key].item())
                    if actual != expected:
                        raise ObservationContractError(
                            f"saved state normalizer {key} mismatch: "
                            f"saved={actual!r}, current={expected!r}"
                        )
                expected_features = {
                    "local_physical_feature_names": (
                        LOCAL_PHYSICAL_FEATURE_NAMES
                    ),
                    "global_feature_names": GLOBAL_FEATURE_NAMES,
                    "relation_feature_names": RELATION_FEATURE_NAMES,
                }
                for key, expected in expected_features.items():
                    actual = tuple(str(value) for value in saved[key].tolist())
                    if actual != tuple(expected):
                        raise ObservationContractError(
                            f"saved state normalizer {key} order mismatch: "
                            f"saved={actual!r}, current={tuple(expected)!r}"
                        )
                expected_dims = {
                    "local_physical_dim": LOCAL_PHYSICAL_DIM,
                    "global_dim": GLOBAL_DIM,
                    "relation_dim": RELATION_DIM,
                }
                for key, expected in expected_dims.items():
                    actual = int(saved[key].item())
                    if actual != expected:
                        raise ObservationContractError(
                            f"saved state normalizer {key} mismatch: "
                            f"saved={actual}, current={expected}"
                        )

                local_clip = (
                    None
                    if bool(saved["local_clip_is_none"].item())
                    else float(saved["local_clip"].item())
                )
                global_clip = (
                    None
                    if bool(saved["global_clip_is_none"].item())
                    else float(saved["global_clip"].item())
                )
                local_state = {
                    "dim": LOCAL_PHYSICAL_DIM,
                    "epsilon": float(saved["local_epsilon"].item()),
                    "clip": local_clip,
                    "mean": saved["local_mean"].copy(),
                    "m2": saved["local_m2"].copy(),
                    "count": int(saved["local_count"].item()),
                    "update_calls": int(saved["local_update_calls"].item()),
                    "frozen": bool(saved["local_frozen"].item()),
                }
                global_state = {
                    "dim": GLOBAL_DIM,
                    "epsilon": float(saved["global_epsilon"].item()),
                    "clip": global_clip,
                    "mean": saved["global_mean"].copy(),
                    "m2": saved["global_m2"].copy(),
                    "count": int(saved["global_count"].item()),
                    "update_calls": int(
                        saved["global_update_calls"].item()
                    ),
                    "frozen": bool(saved["global_frozen"].item()),
                }
                env_steps = int(saved["env_steps"].item())
        except ObservationContractError:
            raise
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ObservationContractError(
                f"invalid state normalizer snapshot: path={source}, error={error}"
            ) from error

        local_candidate = RunningFeatureNormalizer(
            LOCAL_PHYSICAL_DIM,
            epsilon=self.local_normalizer.epsilon,
            clip=self.local_normalizer.clip,
        )
        global_candidate = RunningFeatureNormalizer(
            GLOBAL_DIM,
            epsilon=self.global_normalizer.epsilon,
            clip=self.global_normalizer.clip,
        )
        local_candidate.load_state_dict(local_state)
        global_candidate.load_state_dict(global_state)
        if env_steps < 0:
            raise ObservationContractError(
                "saved state normalizer env_steps must be non-negative"
            )
        if require_frozen and not (
            local_candidate.frozen
            and global_candidate.frozen
            and local_candidate.count > 0
            and global_candidate.count > 0
        ):
            raise ObservationContractError(
                "warm-up bypass requires populated, frozen local and global "
                "state normalizers"
            )

        self.local_normalizer.load_state_dict(local_state)
        self.global_normalizer.load_state_dict(global_state)
        self.env_steps = env_steps
