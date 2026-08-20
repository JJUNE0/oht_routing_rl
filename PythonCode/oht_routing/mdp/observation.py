"""Physical-once contextual observation construction for per-rail TD7.

This module intentionally contains no model, action, reward, replay, or learner
logic. Local normalization is performed on the physical rail array before any
controlled-center or neighbor gather so a rail contributes exactly once per
environment step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from oht_routing.mdp.topology import ContextualTopology, TOPOLOGY_VERSION
from oht_routing.version import CONTEXTUAL_VERSION


LOCAL_FEATURE_NAMES = (
    "current_oht_count",
    "idle_oht_count",
    "predicted_oht_count",
    "parameter_dw",
    "parameter_c",
    "distance_per_velocity",
    "port_count",
    "diverging_line_count",
)
GLOBAL_FEATURE_NAMES = (
    "total_tat",
    "total_oht_operation_rate",
    "n_queued",
    "n_waiting",
    "n_transferring",
    "mean_reassign",
)
RELATION_FEATURE_NAMES = (
    "directed_hop",
    "cumulative_distance_per_velocity",
)

LOCAL_DIM = len(LOCAL_FEATURE_NAMES)
GLOBAL_DIM = len(GLOBAL_FEATURE_NAMES)
RELATION_DIM = len(RELATION_FEATURE_NAMES)
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
            LOCAL_DIM, epsilon=self.config.epsilon, clip=self.config.clip
        )
        self.global_normalizer = RunningFeatureNormalizer(
            GLOBAL_DIM, epsilon=self.config.epsilon, clip=self.config.clip
        )
        self.relation_normalizer = RunningFeatureNormalizer(
            RELATION_DIM, epsilon=self.config.epsilon, clip=self.config.clip
        )
        self.env_steps = 0

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
    def _job_stats(pclient) -> tuple[float, float, float, float]:
        jobs = list(getattr(pclient, "JOB_DIC", {}).values())
        if not jobs:
            return 0.0, 0.0, 0.0, 0.0
        states = [int(getattr(job, "State", 0) or 0) for job in jobs]
        reassign = [
            max(0.0, float(getattr(job, "ReAssignCount", 0) or 0))
            for job in jobs
        ]
        return (
            float(states.count(1)),
            float(states.count(3)),
            float(states.count(5)),
            float(np.mean(reassign)) if reassign else 0.0,
        )

    def build_raw(
        self,
        pclient,
        *,
        parameter_dw: Mapping[int, float],
        parameter_c: Mapping[int, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        rail_lines = getattr(pclient, "RAILLINE_DIC", {})
        actual_ids = {int(rail_id) for rail_id in rail_lines}
        expected_ids = set(self._rail_id_to_physical_row)
        if actual_ids != expected_ids:
            raise ObservationContractError(
                "runtime physical rail IDs differ from audited topology: "
                f"missing={sorted(expected_ids - actual_ids)[:20]}, "
                f"unexpected={sorted(actual_ids - expected_ids)[:20]}"
            )

        physical_local = np.empty(
            (len(self.topology.all_rail_ids), LOCAL_DIM), dtype=np.float64
        )
        for physical_row, rail_id_value in enumerate(self.topology.all_rail_ids):
            rail_id = int(rail_id_value)
            rail = rail_lines[rail_id]
            physical_local[physical_row] = (
                float(len(getattr(rail, "OhtList"))),
                float(getattr(rail, "IdleOHTCount")),
                float(getattr(rail, "PredictedOHTCount")),
                float(parameter_dw.get(rail_id, 1.0)),
                float(parameter_c.get(rail_id, 0.0)),
                float(getattr(rail, "DistancePerVelocity")),
                float(getattr(rail, "PortCount")),
                float(getattr(rail, "DivergingLineCount")),
            )

        n_queued, n_waiting, n_transferring, mean_reassign = self._job_stats(
            pclient
        )
        global_raw = np.asarray(
            (
                float(getattr(pclient, "TotalTat")),
                float(getattr(pclient, "TotalOhtOperationRate")),
                n_queued,
                n_waiting,
                n_transferring,
                mean_reassign,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(physical_local).all():
            raise ObservationContractError("physical_local_raw contains NaN or Inf")
        if not np.isfinite(global_raw).all():
            raise ObservationContractError("global_raw contains NaN or Inf")
        return physical_local, global_raw

    def build(
        self,
        pclient,
        *,
        parameter_dw: Mapping[int, float],
        parameter_c: Mapping[int, float],
        previous_applied_action=None,
    ) -> ContextualObservationBatch:
        physical_raw, global_raw = self.build_raw(
            pclient, parameter_dw=parameter_dw, parameter_c=parameter_c
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
        expected = {
            "center_local": (controlled_count, LOCAL_DIM),
            "incoming_local": (controlled_count, 10, LOCAL_DIM),
            "outgoing_local": (controlled_count, 10, LOCAL_DIM),
            "incoming_relation": (controlled_count, 10, RELATION_DIM),
            "outgoing_relation": (controlled_count, 10, RELATION_DIM),
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
        raw_expected = {
            "physical_local_raw": (len(self.topology.all_rail_ids), LOCAL_DIM),
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
                local_feature_names=np.asarray(LOCAL_FEATURE_NAMES),
                global_feature_names=np.asarray(GLOBAL_FEATURE_NAMES),
                relation_feature_names=np.asarray(RELATION_FEATURE_NAMES),
                local_dim=np.asarray(LOCAL_DIM, dtype=np.int64),
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
                expected_scalars = {
                    "version": CONTEXTUAL_VERSION,
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
                    "local_feature_names": LOCAL_FEATURE_NAMES,
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
                    "local_dim": LOCAL_DIM,
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
                    "dim": LOCAL_DIM,
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
            LOCAL_DIM,
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
