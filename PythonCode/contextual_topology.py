"""Deterministic topology preprocessing for contextual per-rail TD7.

This module is intentionally independent from the region-token implementation.
It consumes the live ``pclient.RAILLINE_DIC`` after simulator initialization and
produces fixed [num_controlled_rails, neighbor_count] incoming/outgoing mappings.

The physical graph and neighbor source retain every rail. Rails that cannot meet
the fixed directional-neighbor contract are allowed only when their exact ID set
matches the configured boundary-center contract. Boundary rails are excluded
from controlled center rows, never from the physical graph or neighbor source.
There is no padding, repetition, or other fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


TOPOLOGY_VERSION = "directed_10in_10out_controlled_centers_v2"
DEFAULT_NEIGHBOR_COUNT = 10
DEFAULT_EXPECTED_BOUNDARY_RAIL_IDS = frozenset({3250, 3251, 3252})
BOUNDARY_POLICY = "baseline_cost_no_action_reward_replay_loss"


class TopologyAuditError(RuntimeError):
    """Raised when the live topology cannot satisfy the contextual contract."""


@dataclass(frozen=True)
class NeighborCandidate:
    rail_id: int
    hop: int
    travel_time: float


@dataclass(frozen=True)
class ContextualTopology:
    all_rail_ids: np.ndarray
    controlled_rail_ids: np.ndarray
    boundary_rail_ids: np.ndarray
    physical_index_to_controlled_row: np.ndarray
    controlled_row_to_physical_index: np.ndarray
    incoming_neighbor_ids: np.ndarray
    outgoing_neighbor_ids: np.ndarray
    incoming_hops: np.ndarray
    outgoing_hops: np.ndarray
    incoming_travel_time: np.ndarray
    outgoing_travel_time: np.ndarray
    topology_hash: str
    mapping_hash: str
    audit: dict

    @property
    def rail_ids(self) -> np.ndarray:
        """Backward-compatible alias for controlled center rail IDs."""
        return self.controlled_rail_ids

    def save_cache(self, path: str | Path) -> Path:
        """Persist the fixed mapping. Refuses to overwrite an incompatible map."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with np.load(target, allow_pickle=False) as old:
                old_files = set(old.files)
                old_hash = (
                    str(old["mapping_hash"].item())
                    if "mapping_hash" in old_files
                    else "<missing>"
                )
                old_topology_hash = (
                    str(old["topology_hash"].item())
                    if "topology_hash" in old_files
                    else "<missing>"
                )
                old_version = (
                    str(old["topology_version"].item())
                    if "topology_version" in old_files
                    else "<missing>"
                )
                old_boundary_policy = (
                    str(old["boundary_policy"].item())
                    if "boundary_policy" in old_files
                    else "<missing>"
                )
            if (
                old_hash != self.mapping_hash
                or old_topology_hash != self.topology_hash
                or old_version != TOPOLOGY_VERSION
                or old_boundary_policy != BOUNDARY_POLICY
            ):
                raise TopologyAuditError(
                    "existing topology cache does not match the audited topology: "
                    f"path={target}, existing_mapping={old_hash}, "
                    f"current_mapping={self.mapping_hash}, "
                    f"existing_topology={old_topology_hash}, "
                    f"current_topology={self.topology_hash}, "
                    f"existing_version={old_version}, current_version={TOPOLOGY_VERSION}, "
                    f"existing_boundary_policy={old_boundary_policy}, "
                    f"current_boundary_policy={BOUNDARY_POLICY}"
                )
            return target
        np.savez_compressed(
            target,
            topology_version=np.asarray(TOPOLOGY_VERSION),
            boundary_policy=np.asarray(BOUNDARY_POLICY),
            all_rail_ids=self.all_rail_ids,
            controlled_rail_ids=self.controlled_rail_ids,
            boundary_rail_ids=self.boundary_rail_ids,
            physical_index_to_controlled_row=self.physical_index_to_controlled_row,
            controlled_row_to_physical_index=self.controlled_row_to_physical_index,
            incoming_neighbor_ids=self.incoming_neighbor_ids,
            outgoing_neighbor_ids=self.outgoing_neighbor_ids,
            incoming_hops=self.incoming_hops,
            outgoing_hops=self.outgoing_hops,
            incoming_travel_time=self.incoming_travel_time,
            outgoing_travel_time=self.outgoing_travel_time,
            topology_hash=np.asarray(self.topology_hash),
            mapping_hash=np.asarray(self.mapping_hash),
        )
        return target


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return target


def _as_unique_int_tuple(values: Iterable[int], *, owner: int, field: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != len(set(result)):
        raise TopologyAuditError(
            f"duplicate rail ID in {field}: rail={owner}, values={result}"
        )
    return tuple(sorted(result))


def _extract_live_graph(
    rail_lines: Mapping[int, object],
    *,
    validate_declared_predecessors: bool,
) -> tuple[
    tuple[int, ...],
    Dict[int, tuple[int, ...]],
    Dict[int, tuple[int, ...]],
    Dict[int, float],
]:
    if not rail_lines:
        raise TopologyAuditError("RAILLINE_DIC is empty; topology audit cannot run")

    rail_ids = tuple(sorted(int(rail_id) for rail_id in rail_lines))
    if len(rail_ids) != len(set(rail_ids)):
        raise TopologyAuditError("RAILLINE_DIC contains duplicate integer rail IDs")
    rail_set = set(rail_ids)

    successors: Dict[int, tuple[int, ...]] = {}
    travel_time: Dict[int, float] = {}
    for rail_id in rail_ids:
        rail = rail_lines[rail_id]
        object_id = int(getattr(rail, "ID", rail_id))
        if object_id != rail_id:
            raise TopologyAuditError(
                f"RAILLINE_DIC key/object ID mismatch: key={rail_id}, object.ID={object_id}"
            )

        value = float(getattr(rail, "DistancePerVelocity"))
        if not math.isfinite(value) or value < 0.0:
            raise TopologyAuditError(
                f"invalid DistancePerVelocity: rail={rail_id}, value={value}"
            )
        travel_time[rail_id] = value

        outgoing = _as_unique_int_tuple(
            getattr(rail, "DivergingLineIDList"),
            owner=rail_id,
            field="DivergingLineIDList",
        )
        invalid = sorted(set(outgoing) - rail_set)
        if invalid:
            raise TopologyAuditError(
                f"successor references unknown rail: rail={rail_id}, unknown={invalid}"
            )
        successors[rail_id] = outgoing

    predecessor_sets: Dict[int, set[int]] = {rail_id: set() for rail_id in rail_ids}
    for source, targets in successors.items():
        for target in targets:
            predecessor_sets[target].add(source)
    predecessors = {
        rail_id: tuple(sorted(predecessor_sets[rail_id])) for rail_id in rail_ids
    }

    if validate_declared_predecessors:
        mismatches = []
        for rail_id in rail_ids:
            rail = rail_lines[rail_id]
            declared = _as_unique_int_tuple(
                getattr(rail, "LevelJoiningLineIDList"),
                owner=rail_id,
                field="LevelJoiningLineIDList",
            )
            invalid = sorted(set(declared) - rail_set)
            if invalid:
                raise TopologyAuditError(
                    f"predecessor references unknown rail: rail={rail_id}, unknown={invalid}"
                )
            derived = predecessors[rail_id]
            if declared != derived:
                mismatches.append(
                    {
                        "rail_id": rail_id,
                        "declared": list(declared),
                        "derived_from_successors": list(derived),
                    }
                )
        if mismatches:
            preview = mismatches[:10]
            raise TopologyAuditError(
                "LevelJoiningLineIDList disagrees with reverse DivergingLineIDList; "
                f"mismatch_count={len(mismatches)}, first={preview}"
            )

    return rail_ids, successors, predecessors, travel_time


def rank_directional_neighbors(
    center_rail_id: int,
    adjacency: Mapping[int, Sequence[int]],
    travel_time: Mapping[int, float],
) -> list[NeighborCandidate]:
    """Return all reachable rails ordered by hop, cumulative time, and rail ID.

    Cumulative time is the sum of ``DistancePerVelocity`` for each traversed
    neighbor rail. The center rail's time is excluded. Among paths with the same
    minimum hop count to a node, the minimum cumulative time is retained.
    """
    center = int(center_rail_id)
    if center not in adjacency:
        raise TopologyAuditError(f"center rail is absent from adjacency: rail={center}")

    visited = {center}
    frontier: Dict[int, float] = {center: 0.0}
    candidates: list[NeighborCandidate] = []
    hop = 0

    while frontier:
        hop += 1
        next_frontier: Dict[int, float] = {}
        for source, source_time in sorted(frontier.items()):
            for neighbor_value in adjacency[source]:
                neighbor = int(neighbor_value)
                if neighbor == center or neighbor in visited:
                    continue
                candidate_time = source_time + float(travel_time[neighbor])
                previous = next_frontier.get(neighbor)
                if previous is None or candidate_time < previous:
                    next_frontier[neighbor] = candidate_time

        if not next_frontier:
            break

        level = [
            NeighborCandidate(rail_id=rail_id, hop=hop, travel_time=value)
            for rail_id, value in next_frontier.items()
        ]
        level.sort(key=lambda item: (item.hop, item.travel_time, item.rail_id))
        candidates.extend(level)
        visited.update(next_frontier)
        frontier = next_frontier

    candidates.sort(key=lambda item: (item.hop, item.travel_time, item.rail_id))
    return candidates


def _distribution(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _hop_histogram(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values.astype(np.int64), return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts)}


def _failed_audit(error: Exception, *, neighbor_count: int) -> dict:
    return {
        "status": "failed",
        "topology_version": TOPOLOGY_VERSION,
        "neighbor_count": int(neighbor_count),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def build_contextual_topology(
    rail_lines: Mapping[int, object],
    *,
    neighbor_count: int = DEFAULT_NEIGHBOR_COUNT,
    expected_rail_count: int | None = None,
    expected_boundary_rail_ids: Iterable[int] = DEFAULT_EXPECTED_BOUNDARY_RAIL_IDS,
    validate_declared_predecessors: bool = True,
    audit_path: str | Path | None = None,
    cache_path: str | Path | None = None,
) -> ContextualTopology:
    """Audit live topology and build fixed incoming/outgoing nearest mappings.

    On any violation this function raises ``TopologyAuditError``. If
    ``audit_path`` is provided, a failed JSON artifact is written before raising.
    """
    try:
        count = int(neighbor_count)
        if count != DEFAULT_NEIGHBOR_COUNT:
            raise TopologyAuditError(
                "contextual topology contract requires exactly "
                f"{DEFAULT_NEIGHBOR_COUNT} neighbors per direction: requested={count}"
            )

        all_rail_ids, successors, predecessors, travel_time = _extract_live_graph(
            rail_lines,
            validate_declared_predecessors=validate_declared_predecessors,
        )
        expected_boundary = tuple(
            sorted({int(rail_id) for rail_id in expected_boundary_rail_ids})
        )
        if (
            expected_rail_count is not None
            and len(all_rail_ids) != int(expected_rail_count)
        ):
            raise TopologyAuditError(
                "simulator declared/loaded rail count mismatch: "
                f"declared={int(expected_rail_count)}, loaded={len(all_rail_ids)}"
            )
        unknown_expected_boundary = sorted(
            set(expected_boundary) - set(all_rail_ids)
        )
        if unknown_expected_boundary:
            raise TopologyAuditError(
                "expected boundary rail is absent from physical topology: "
                f"unknown={unknown_expected_boundary}"
            )

        incoming_ranked = {
            rail_id: rank_directional_neighbors(rail_id, predecessors, travel_time)
            for rail_id in all_rail_ids
        }
        outgoing_ranked = {
            rail_id: rank_directional_neighbors(rail_id, successors, travel_time)
            for rail_id in all_rail_ids
        }

        incoming_reachable = [len(incoming_ranked[rail_id]) for rail_id in all_rail_ids]
        outgoing_reachable = [len(outgoing_ranked[rail_id]) for rail_id in all_rail_ids]
        deficient = [
            {
                "rail_id": rail_id,
                "incoming_reachable": len(incoming_ranked[rail_id]),
                "outgoing_reachable": len(outgoing_ranked[rail_id]),
            }
            for rail_id in all_rail_ids
            if len(incoming_ranked[rail_id]) < count
            or len(outgoing_ranked[rail_id]) < count
        ]
        actual_boundary = tuple(item["rail_id"] for item in deficient)
        if actual_boundary != expected_boundary:
            error = TopologyAuditError(
                "actual deficient rail set does not exactly match expected boundary: "
                f"expected={list(expected_boundary)}, actual={list(actual_boundary)}, "
                f"required_neighbors={count}"
            )
            failed = _failed_audit(error, neighbor_count=count)
            failed.update(
                {
                    "physical_rail_count": len(all_rail_ids),
                    "rail_count": len(all_rail_ids),
                    "expected_boundary_rail_ids": list(expected_boundary),
                    "actual_boundary_rail_ids": list(actual_boundary),
                    "reachable_incoming": _distribution(incoming_reachable),
                    "reachable_outgoing": _distribution(outgoing_reachable),
                    "deficient_rails": deficient,
                }
            )
            if audit_path is not None:
                _write_json(audit_path, failed)
            raise error

        boundary_set = set(actual_boundary)
        controlled_rail_ids = tuple(
            rail_id for rail_id in all_rail_ids if rail_id not in boundary_set
        )
        incoming_selected = {
            rail_id: incoming_ranked[rail_id][:count] for rail_id in controlled_rail_ids
        }
        outgoing_selected = {
            rail_id: outgoing_ranked[rail_id][:count] for rail_id in controlled_rail_ids
        }

        def matrix(direction: Mapping[int, Sequence[NeighborCandidate]], field: str, dtype):
            return np.asarray(
                [
                    [getattr(item, field) for item in direction[rail_id]]
                    for rail_id in controlled_rail_ids
                ],
                dtype=dtype,
            )

        incoming_ids = matrix(incoming_selected, "rail_id", np.int64)
        outgoing_ids = matrix(outgoing_selected, "rail_id", np.int64)
        incoming_hops = matrix(incoming_selected, "hop", np.int32)
        outgoing_hops = matrix(outgoing_selected, "hop", np.int32)
        incoming_times = matrix(incoming_selected, "travel_time", np.float64)
        outgoing_times = matrix(outgoing_selected, "travel_time", np.float64)

        expected_shape = (len(controlled_rail_ids), count)
        arrays = {
            "incoming_neighbor_ids": incoming_ids,
            "outgoing_neighbor_ids": outgoing_ids,
            "incoming_hops": incoming_hops,
            "outgoing_hops": outgoing_hops,
            "incoming_travel_time": incoming_times,
            "outgoing_travel_time": outgoing_times,
        }
        for name, array in arrays.items():
            if array.shape != expected_shape:
                raise TopologyAuditError(
                    f"mapping shape mismatch: {name}={array.shape}, expected={expected_shape}"
                )
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise TopologyAuditError(f"mapping contains non-finite values: {name}")

        rail_set = set(all_rail_ids)
        for row, center in enumerate(controlled_rail_ids):
            for name, ids in (
                ("incoming", incoming_ids[row]),
                ("outgoing", outgoing_ids[row]),
            ):
                values = [int(value) for value in ids]
                if center in values:
                    raise TopologyAuditError(
                        f"center rail appears in {name} neighbors: rail={center}"
                    )
                if len(values) != len(set(values)):
                    raise TopologyAuditError(
                        f"duplicate {name} neighbor: rail={center}, values={values}"
                    )
                if not set(values).issubset(rail_set):
                    raise TopologyAuditError(
                        f"unknown {name} neighbor: rail={center}, values={values}"
                    )

        if (incoming_hops < 1).any() or (outgoing_hops < 1).any():
            raise TopologyAuditError("selected mapping contains hop count below one")
        if (incoming_times < 0).any() or (outgoing_times < 0).any():
            raise TopologyAuditError("selected mapping contains negative travel time")

        topology_payload = [
            {
                "rail_id": rail_id,
                "distance_per_velocity": travel_time[rail_id],
                "successors": list(successors[rail_id]),
                "predecessors": list(predecessors[rail_id]),
            }
            for rail_id in all_rail_ids
        ]
        physical_index_by_id = {
            rail_id: index for index, rail_id in enumerate(all_rail_ids)
        }
        controlled_row_by_id = {
            rail_id: row for row, rail_id in enumerate(controlled_rail_ids)
        }
        physical_to_controlled = np.full(
            len(all_rail_ids), -1, dtype=np.int64
        )
        controlled_to_physical = np.asarray(
            [physical_index_by_id[rail_id] for rail_id in controlled_rail_ids],
            dtype=np.int64,
        )
        for rail_id, row in controlled_row_by_id.items():
            physical_to_controlled[physical_index_by_id[rail_id]] = row

        topology_hash = _stable_hash(
            {
                "topology_version": TOPOLOGY_VERSION,
                "boundary_policy": BOUNDARY_POLICY,
                "expected_boundary_rail_ids": list(expected_boundary),
                "actual_boundary_rail_ids": list(actual_boundary),
                "physical_graph": topology_payload,
            }
        )
        mapping_payload = {
            "topology_version": TOPOLOGY_VERSION,
            "boundary_policy": BOUNDARY_POLICY,
            "all_rail_ids": list(all_rail_ids),
            "controlled_rail_ids": list(controlled_rail_ids),
            "boundary_rail_ids": list(actual_boundary),
            "physical_index_to_controlled_row": physical_to_controlled.tolist(),
            "controlled_row_to_physical_index": controlled_to_physical.tolist(),
            "center_mappings": [
            {
                "rail_id": rail_id,
                "incoming": [
                    [item.rail_id, item.hop, item.travel_time]
                    for item in incoming_selected[rail_id]
                ],
                "outgoing": [
                    [item.rail_id, item.hop, item.travel_time]
                    for item in outgoing_selected[rail_id]
                ],
            }
            for rail_id in controlled_rail_ids
            ],
        }
        mapping_hash = _stable_hash(mapping_payload)

        controlled_incoming_reachable = [
            len(incoming_ranked[rail_id]) for rail_id in controlled_rail_ids
        ]
        controlled_outgoing_reachable = [
            len(outgoing_ranked[rail_id]) for rail_id in controlled_rail_ids
        ]
        audit = {
            "status": "passed",
            "topology_version": TOPOLOGY_VERSION,
            "neighbor_count": count,
            "rail_count": len(all_rail_ids),
            "physical_rail_count": len(all_rail_ids),
            "controlled_rail_count": len(controlled_rail_ids),
            "boundary_rail_count": len(actual_boundary),
            "boundary_rail_ids": list(actual_boundary),
            "expected_boundary_rail_ids": list(expected_boundary),
            "boundary_policy": BOUNDARY_POLICY,
            "expected_rail_count": (
                None if expected_rail_count is None else int(expected_rail_count)
            ),
            "reachable_incoming": _distribution(incoming_reachable),
            "reachable_outgoing": _distribution(outgoing_reachable),
            "controlled_reachable_incoming": _distribution(
                controlled_incoming_reachable
            ),
            "controlled_reachable_outgoing": _distribution(
                controlled_outgoing_reachable
            ),
            "selected_incoming_hop_histogram": _hop_histogram(incoming_hops),
            "selected_outgoing_hop_histogram": _hop_histogram(outgoing_hops),
            "selected_incoming_travel_time": _distribution(incoming_times.reshape(-1)),
            "selected_outgoing_travel_time": _distribution(outgoing_times.reshape(-1)),
            "topology_hash": topology_hash,
            "mapping_hash": mapping_hash,
            "declared_predecessors_validated": bool(validate_declared_predecessors),
        }

        result = ContextualTopology(
            all_rail_ids=np.asarray(all_rail_ids, dtype=np.int64),
            controlled_rail_ids=np.asarray(controlled_rail_ids, dtype=np.int64),
            boundary_rail_ids=np.asarray(actual_boundary, dtype=np.int64),
            physical_index_to_controlled_row=physical_to_controlled,
            controlled_row_to_physical_index=controlled_to_physical,
            incoming_neighbor_ids=incoming_ids,
            outgoing_neighbor_ids=outgoing_ids,
            incoming_hops=incoming_hops,
            outgoing_hops=outgoing_hops,
            incoming_travel_time=incoming_times,
            outgoing_travel_time=outgoing_times,
            topology_hash=topology_hash,
            mapping_hash=mapping_hash,
            audit=audit,
        )
        if audit_path is not None:
            _write_json(audit_path, audit)
        if cache_path is not None:
            result.save_cache(cache_path)
        return result
    except TopologyAuditError as error:
        if audit_path is not None:
            target = Path(audit_path)
            keep_existing_failure = False
            if target.exists():
                try:
                    keep_existing_failure = (
                        json.loads(target.read_text(encoding="utf-8")).get("status")
                        == "failed"
                    )
                except (OSError, json.JSONDecodeError):
                    keep_existing_failure = False
            if not keep_existing_failure:
                _write_json(
                    audit_path,
                    _failed_audit(error, neighbor_count=int(neighbor_count)),
                )
        raise


def load_cached_contextual_topology(
    rail_lines: Mapping[int, object],
    *,
    cache_path: str | Path,
    expected_rail_count: int | None = None,
    expected_boundary_rail_ids: Iterable[int] = DEFAULT_EXPECTED_BOUNDARY_RAIL_IDS,
    validate_declared_predecessors: bool = True,
) -> ContextualTopology:
    """Validate a live physical graph in O(N+E) and load its audited mapping.

    The expensive all-center directional traversal belongs to the offline
    topology audit. Runtime recomputes the physical topology hash from live
    rail data and refuses any cache whose version, policy, IDs, mapping payload,
    or topology hash differs.
    """
    all_rail_ids, successors, predecessors, travel_time = _extract_live_graph(
        rail_lines,
        validate_declared_predecessors=validate_declared_predecessors,
    )
    if expected_rail_count is not None and len(all_rail_ids) != int(
        expected_rail_count
    ):
        raise TopologyAuditError(
            "simulator declared/loaded rail count mismatch: "
            f"declared={int(expected_rail_count)}, loaded={len(all_rail_ids)}"
        )
    expected_boundary = tuple(
        sorted({int(rail_id) for rail_id in expected_boundary_rail_ids})
    )
    topology_payload = [
        {
            "rail_id": rail_id,
            "distance_per_velocity": travel_time[rail_id],
            "successors": list(successors[rail_id]),
            "predecessors": list(predecessors[rail_id]),
        }
        for rail_id in all_rail_ids
    ]
    live_topology_hash = _stable_hash(
        {
            "topology_version": TOPOLOGY_VERSION,
            "boundary_policy": BOUNDARY_POLICY,
            "expected_boundary_rail_ids": list(expected_boundary),
            "actual_boundary_rail_ids": list(expected_boundary),
            "physical_graph": topology_payload,
        }
    )

    target = Path(cache_path)
    if not target.is_file():
        raise TopologyAuditError(f"topology cache does not exist: {target}")
    try:
        with np.load(target, allow_pickle=False) as cache:
            version = str(cache["topology_version"].item())
            policy = str(cache["boundary_policy"].item())
            cached_topology_hash = str(cache["topology_hash"].item())
            cached_mapping_hash = str(cache["mapping_hash"].item())
            arrays = {
                name: cache[name].copy()
                for name in (
                    "all_rail_ids",
                    "controlled_rail_ids",
                    "boundary_rail_ids",
                    "physical_index_to_controlled_row",
                    "controlled_row_to_physical_index",
                    "incoming_neighbor_ids",
                    "outgoing_neighbor_ids",
                    "incoming_hops",
                    "outgoing_hops",
                    "incoming_travel_time",
                    "outgoing_travel_time",
                )
            }
    except (KeyError, OSError, ValueError) as error:
        raise TopologyAuditError(
            f"invalid contextual topology cache: path={target}, error={error}"
        ) from error

    if version != TOPOLOGY_VERSION or policy != BOUNDARY_POLICY:
        raise TopologyAuditError(
            "topology cache contract mismatch: "
            f"version={version}, policy={policy}"
        )
    if cached_topology_hash != live_topology_hash:
        raise TopologyAuditError(
            "live topology hash does not match audited cache: "
            f"live={live_topology_hash}, cache={cached_topology_hash}"
        )
    if not np.array_equal(
        arrays["all_rail_ids"], np.asarray(all_rail_ids, dtype=np.int64)
    ):
        raise TopologyAuditError("cached all_rail_ids differ from live topology")
    if not np.array_equal(
        arrays["boundary_rail_ids"],
        np.asarray(expected_boundary, dtype=np.int64),
    ):
        raise TopologyAuditError("cached boundary IDs differ from runtime contract")

    controlled_ids = arrays["controlled_rail_ids"]
    physical_to_controlled = arrays["physical_index_to_controlled_row"]
    controlled_to_physical = arrays["controlled_row_to_physical_index"]
    if not np.array_equal(
        arrays["all_rail_ids"][controlled_to_physical], controlled_ids
    ):
        raise TopologyAuditError("cached physical/controlled index alignment failed")
    if not np.array_equal(
        physical_to_controlled[controlled_to_physical],
        np.arange(len(controlled_ids), dtype=np.int64),
    ):
        raise TopologyAuditError("cached controlled row round-trip failed")

    incoming_ids = arrays["incoming_neighbor_ids"]
    outgoing_ids = arrays["outgoing_neighbor_ids"]
    incoming_hops = arrays["incoming_hops"]
    outgoing_hops = arrays["outgoing_hops"]
    incoming_times = arrays["incoming_travel_time"]
    outgoing_times = arrays["outgoing_travel_time"]
    expected_shape = (len(controlled_ids), DEFAULT_NEIGHBOR_COUNT)
    for name in (
        "incoming_neighbor_ids",
        "outgoing_neighbor_ids",
        "incoming_hops",
        "outgoing_hops",
        "incoming_travel_time",
        "outgoing_travel_time",
    ):
        if arrays[name].shape != expected_shape:
            raise TopologyAuditError(
                f"cached {name} shape mismatch: {arrays[name].shape}"
            )
    mapping_payload = {
        "topology_version": TOPOLOGY_VERSION,
        "boundary_policy": BOUNDARY_POLICY,
        "all_rail_ids": arrays["all_rail_ids"].tolist(),
        "controlled_rail_ids": controlled_ids.tolist(),
        "boundary_rail_ids": arrays["boundary_rail_ids"].tolist(),
        "physical_index_to_controlled_row": physical_to_controlled.tolist(),
        "controlled_row_to_physical_index": controlled_to_physical.tolist(),
        "center_mappings": [
            {
                "rail_id": int(rail_id),
                "incoming": [
                    [
                        int(incoming_ids[row, slot]),
                        int(incoming_hops[row, slot]),
                        float(incoming_times[row, slot]),
                    ]
                    for slot in range(DEFAULT_NEIGHBOR_COUNT)
                ],
                "outgoing": [
                    [
                        int(outgoing_ids[row, slot]),
                        int(outgoing_hops[row, slot]),
                        float(outgoing_times[row, slot]),
                    ]
                    for slot in range(DEFAULT_NEIGHBOR_COUNT)
                ],
            }
            for row, rail_id in enumerate(controlled_ids)
        ],
    }
    computed_mapping_hash = _stable_hash(mapping_payload)
    if computed_mapping_hash != cached_mapping_hash:
        raise TopologyAuditError(
            "cached mapping payload hash mismatch: "
            f"computed={computed_mapping_hash}, cache={cached_mapping_hash}"
        )

    return ContextualTopology(
        all_rail_ids=arrays["all_rail_ids"],
        controlled_rail_ids=controlled_ids,
        boundary_rail_ids=arrays["boundary_rail_ids"],
        physical_index_to_controlled_row=physical_to_controlled,
        controlled_row_to_physical_index=controlled_to_physical,
        incoming_neighbor_ids=incoming_ids,
        outgoing_neighbor_ids=outgoing_ids,
        incoming_hops=incoming_hops,
        outgoing_hops=outgoing_hops,
        incoming_travel_time=incoming_times,
        outgoing_travel_time=outgoing_times,
        topology_hash=cached_topology_hash,
        mapping_hash=cached_mapping_hash,
        audit={
            "status": "loaded_validated_cache",
            "physical_rail_count": len(all_rail_ids),
            "controlled_rail_count": len(controlled_ids),
            "boundary_rail_count": len(expected_boundary),
            "boundary_rail_ids": list(expected_boundary),
            "topology_hash": cached_topology_hash,
            "mapping_hash": cached_mapping_hash,
        },
    )
