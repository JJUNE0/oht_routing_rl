import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contextual_topology import (
    TOPOLOGY_VERSION,
    TopologyAuditError,
    build_contextual_topology,
    load_cached_contextual_topology,
    rank_directional_neighbors,
)


@dataclass
class FakeRail:
    ID: int
    DistancePerVelocity: float = 1.0
    DivergingLineIDList: list[int] = field(default_factory=list)
    LevelJoiningLineIDList: list[int] = field(default_factory=list)


def make_graph(successors, travel_time=None):
    ids = sorted(successors)
    predecessor = {rail_id: [] for rail_id in ids}
    for source, targets in successors.items():
        for target in targets:
            predecessor[target].append(source)
    travel_time = travel_time or {}
    return {
        rail_id: FakeRail(
            ID=rail_id,
            DistancePerVelocity=float(travel_time.get(rail_id, 1.0)),
            DivergingLineIDList=list(successors[rail_id]),
            LevelJoiningLineIDList=sorted(predecessor[rail_id]),
        )
        for rail_id in ids
    }

def make_controlled_boundary_graph(boundary=(3250, 3251, 3252), controlled_size=12):
    """Three-node entry chain feeding a strongly connected controlled core."""
    controlled = list(range(4000, 4000 + controlled_size))
    successors = {
        boundary[0]: [boundary[1]],
        boundary[1]: [boundary[2]],
        boundary[2]: [controlled[0]],
    }
    successors.update(
        {
            rail_id: [
                controlled[(index + 1) % len(controlled)]
            ]
            for index, rail_id in enumerate(controlled)
        }
    )
    return make_graph(successors)


class DirectionalRankingTests(unittest.TestCase):
    def test_straight_graph_orders_by_hop(self):
        adjacency = {i: ([i + 1] if i < 12 else []) for i in range(1, 13)}
        times = {i: 1.0 for i in adjacency}
        ranked = rank_directional_neighbors(1, adjacency, times)
        self.assertEqual([item.rail_id for item in ranked[:10]], list(range(2, 12)))
        self.assertEqual([item.hop for item in ranked[:10]], list(range(1, 11)))

    def test_divergence_same_hop_uses_travel_time_then_id(self):
        adjacency = {1: [2, 3, 4], 2: [], 3: [], 4: []}
        times = {1: 1.0, 2: 3.0, 3: 1.0, 4: 1.0}
        ranked = rank_directional_neighbors(1, adjacency, times)
        self.assertEqual([item.rail_id for item in ranked], [3, 4, 2])

    def test_merge_keeps_minimum_time_among_equal_hop_paths(self):
        adjacency = {1: [2, 3], 2: [4], 3: [4], 4: []}
        times = {1: 0.0, 2: 10.0, 3: 1.0, 4: 2.0}
        ranked = rank_directional_neighbors(1, adjacency, times)
        item = next(candidate for candidate in ranked if candidate.rail_id == 4)
        self.assertEqual(item.hop, 2)
        self.assertEqual(item.travel_time, 3.0)

    def test_cycle_excludes_center_and_duplicates(self):
        adjacency = {1: [2], 2: [3], 3: [1]}
        times = {1: 1.0, 2: 1.0, 3: 1.0}
        ranked = rank_directional_neighbors(1, adjacency, times)
        self.assertEqual([item.rail_id for item in ranked], [2, 3])

    def test_self_edge_is_visited_but_never_selected(self):
        adjacency = {1: [1, 2], 2: [2, 3], 3: [3]}
        times = {1: 1.0, 2: 1.0, 3: 1.0}
        ranked = rank_directional_neighbors(1, adjacency, times)
        self.assertEqual([item.rail_id for item in ranked], [2, 3])

    def test_more_than_ten_same_hop_is_deterministic(self):
        targets = list(range(2, 15))
        adjacency = {1: list(reversed(targets))}
        adjacency.update({target: [] for target in targets})
        times = {rail_id: 1.0 for rail_id in adjacency}
        ranked = rank_directional_neighbors(1, adjacency, times)
        self.assertEqual([item.rail_id for item in ranked[:10]], list(range(2, 12)))


class FullTopologyAuditTests(unittest.TestCase):
    def test_directed_cycle_passes_fixed_ten_contract(self):
        size = 25
        successors = {
            rail_id: [rail_id + 1 if rail_id < size else 1]
            for rail_id in range(1, size + 1)
        }
        result = build_contextual_topology(
            make_graph(successors), expected_boundary_rail_ids=()
        )
        self.assertEqual(result.incoming_neighbor_ids.shape, (size, 10))
        self.assertEqual(result.outgoing_neighbor_ids.shape, (size, 10))
        self.assertTrue((result.incoming_hops >= 1).all())
        self.assertTrue((result.outgoing_hops >= 1).all())
        self.assertEqual(result.audit["status"], "passed")
        self.assertEqual(result.audit["topology_version"], TOPOLOGY_VERSION)

    def test_mapping_hash_reproducible_across_input_order(self):
        size = 25
        successors = {
            rail_id: [rail_id + 1 if rail_id < size else 1]
            for rail_id in range(1, size + 1)
        }
        graph = make_graph(successors)
        reversed_graph = {rail_id: graph[rail_id] for rail_id in reversed(graph)}
        first = build_contextual_topology(graph, expected_boundary_rail_ids=())
        second = build_contextual_topology(
            reversed_graph, expected_boundary_rail_ids=()
        )
        self.assertEqual(first.topology_hash, second.topology_hash)
        self.assertEqual(first.mapping_hash, second.mapping_hash)
        np.testing.assert_array_equal(
            first.outgoing_neighbor_ids, second.outgoing_neighbor_ids
        )

    def test_passed_audit_and_cache_are_written(self):
        size = 25
        successors = {
            rail_id: [rail_id + 1 if rail_id < size else 1]
            for rail_id in range(1, size + 1)
        }
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "topology_neighbor_audit.json"
            cache_path = Path(directory) / "contextual_topology_cache.npz"
            result = build_contextual_topology(
                make_graph(successors),
                expected_boundary_rail_ids=(),
                audit_path=audit_path,
                cache_path=cache_path,
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["neighbor_count"], 10)
            self.assertEqual(audit["mapping_hash"], result.mapping_hash)
            self.assertEqual(audit["topology_hash"], result.topology_hash)
            with np.load(cache_path, allow_pickle=False) as cache:
                self.assertEqual(cache["incoming_neighbor_ids"].shape, (size, 10))
                self.assertEqual(cache["outgoing_neighbor_ids"].shape, (size, 10))
                self.assertEqual(cache["mapping_hash"].item(), result.mapping_hash)

    def test_insufficient_reachable_fails_and_writes_audit(self):
        successors = {
            1: [2],
            2: [3],
            3: [4],
            4: [5],
            5: [],
        }
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "topology_neighbor_audit.json"
            with self.assertRaises(TopologyAuditError):
                build_contextual_topology(
                    make_graph(successors),
                    expected_boundary_rail_ids=(),
                    audit_path=audit_path,
                )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "failed")
            self.assertEqual(audit["rail_count"], 5)
            self.assertEqual(len(audit["deficient_rails"]), 5)

    def test_declared_predecessor_mismatch_fails(self):
        size = 25
        successors = {
            rail_id: [rail_id + 1 if rail_id < size else 1]
            for rail_id in range(1, size + 1)
        }
        graph = make_graph(successors)
        graph[2].LevelJoiningLineIDList = [25]
        with self.assertRaisesRegex(
            TopologyAuditError, "LevelJoiningLineIDList disagrees"
        ):
            build_contextual_topology(graph, expected_boundary_rail_ids=())

    def test_unknown_successor_fails(self):
        graph = make_graph({1: [], 2: []})
        graph[1].DivergingLineIDList = [999]
        with self.assertRaisesRegex(TopologyAuditError, "unknown rail"):
            build_contextual_topology(graph, expected_boundary_rail_ids=())

    def test_neighbor_count_cannot_silently_change(self):
        graph = make_graph({1: [2], 2: [1]})
        with self.assertRaisesRegex(TopologyAuditError, "exactly 10"):
            build_contextual_topology(
                graph, neighbor_count=1, expected_boundary_rail_ids=()
            )

    def test_cache_refuses_another_mapping(self):
        size = 25
        cycle = {
            rail_id: [rail_id + 1 if rail_id < size else 1]
            for rail_id in range(1, size + 1)
        }
        reverse_cycle = {
            rail_id: [rail_id - 1 if rail_id > 1 else size]
            for rail_id in range(1, size + 1)
        }
        first = build_contextual_topology(
            make_graph(cycle), expected_boundary_rail_ids=()
        )
        second = build_contextual_topology(
            make_graph(reverse_cycle), expected_boundary_rail_ids=()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contextual_topology_cache.npz"
            first.save_cache(path)
            with self.assertRaisesRegex(TopologyAuditError, "does not match"):
                second.save_cache(path)

    def test_controlled_centers_exclude_exact_boundary_and_keep_fixed_shape(self):
        result = build_contextual_topology(make_controlled_boundary_graph())
        self.assertEqual(result.all_rail_ids.shape, (15,))
        self.assertEqual(result.controlled_rail_ids.shape, (12,))
        self.assertEqual(result.boundary_rail_ids.tolist(), [3250, 3251, 3252])
        self.assertEqual(result.incoming_neighbor_ids.shape, (12, 10))
        self.assertEqual(result.outgoing_neighbor_ids.shape, (12, 10))
        self.assertTrue(
            set(result.boundary_rail_ids).isdisjoint(result.controlled_rail_ids)
        )
        self.assertGreaterEqual(
            result.audit["controlled_reachable_incoming"]["min"], 10
        )
        self.assertGreaterEqual(
            result.audit["controlled_reachable_outgoing"]["min"], 10
        )

    def test_boundary_remains_available_as_neighbor_source(self):
        result = build_contextual_topology(make_controlled_boundary_graph())
        neighbor_ids = set(result.incoming_neighbor_ids.reshape(-1).tolist())
        self.assertTrue(neighbor_ids.intersection({3250, 3251, 3252}))

    def test_unexpected_fourth_boundary_fails_fast(self):
        graph = make_controlled_boundary_graph()
        successors = {
            rail_id: list(rail.DivergingLineIDList)
            for rail_id, rail in graph.items()
        }
        successors[3249] = [3250]
        graph = make_graph(successors)
        with self.assertRaisesRegex(
            TopologyAuditError, "does not exactly match expected boundary"
        ):
            build_contextual_topology(graph)

    def test_physical_controlled_index_round_trip(self):
        result = build_contextual_topology(make_controlled_boundary_graph())
        for controlled_row, physical_index in enumerate(
            result.controlled_row_to_physical_index
        ):
            self.assertEqual(
                result.physical_index_to_controlled_row[physical_index],
                controlled_row,
            )
            self.assertEqual(
                result.all_rail_ids[physical_index],
                result.controlled_rail_ids[controlled_row],
            )
        boundary_physical_indices = [
            int(np.where(result.all_rail_ids == rail_id)[0][0])
            for rail_id in result.boundary_rail_ids
        ]
        self.assertTrue(
            (
                result.physical_index_to_controlled_row[
                    boundary_physical_indices
                ]
                == -1
            ).all()
        )

    def test_v1_cache_is_incompatible(self):
        result = build_contextual_topology(make_controlled_boundary_graph())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contextual_topology_cache.npz"
            np.savez_compressed(
                path,
                topology_version=np.asarray("directed_10in_10out_v1"),
                topology_hash=np.asarray(result.topology_hash),
                mapping_hash=np.asarray(result.mapping_hash),
            )
            with self.assertRaisesRegex(TopologyAuditError, "does not match"):
                result.save_cache(path)

    def test_runtime_cache_loader_reuses_audited_mapping(self):
        graph = make_controlled_boundary_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.npz"
            audited = build_contextual_topology(graph, cache_path=path)
            loaded = load_cached_contextual_topology(
                graph,
                cache_path=path,
                expected_rail_count=len(graph),
            )
            self.assertEqual(loaded.audit["status"], "loaded_validated_cache")
            self.assertEqual(loaded.topology_hash, audited.topology_hash)
            self.assertEqual(loaded.mapping_hash, audited.mapping_hash)
            np.testing.assert_array_equal(
                loaded.incoming_neighbor_ids,
                audited.incoming_neighbor_ids,
            )

    def test_runtime_cache_loader_rejects_changed_live_graph(self):
        graph = make_controlled_boundary_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.npz"
            build_contextual_topology(graph, cache_path=path)
            graph[4000].DistancePerVelocity += 0.25
            with self.assertRaisesRegex(
                TopologyAuditError, "live topology hash does not match"
            ):
                load_cached_contextual_topology(graph, cache_path=path)


if __name__ == "__main__":
    unittest.main()
