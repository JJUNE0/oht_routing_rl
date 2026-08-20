import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from oht_routing.utils import check_rail_topology


@dataclass
class FakeRail:
    ID: int
    DistancePerVelocity: float = 1.0
    DivergingLineIDList: list[int] = field(default_factory=list)
    LevelJoiningLineIDList: list[int] = field(default_factory=list)


def make_graph(successors):
    predecessors = {rail_id: [] for rail_id in successors}
    for source, targets in successors.items():
        for target in targets:
            predecessors[target].append(source)
    return {
        rail_id: FakeRail(
            ID=rail_id,
            DivergingLineIDList=list(successors[rail_id]),
            LevelJoiningLineIDList=sorted(predecessors[rail_id]),
        )
        for rail_id in sorted(successors)
    }

def make_controlled_boundary_graph():
    controlled = list(range(4000, 4012))
    successors = {
        3250: [3251],
        3251: [3252],
        3252: [controlled[0]],
    }
    successors.update(
        {
            rail_id: [controlled[(index + 1) % len(controlled)]]
            for index, rail_id in enumerate(controlled)
        }
    )
    return make_graph(successors)


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeServerSocket:
    def __init__(self):
        self.bound = None
        self.closed = False
        self.connection = FakeConnection()

    def setsockopt(self, *args):
        pass

    def bind(self, address):
        self.bound = address

    def listen(self, backlog):
        pass

    def accept(self):
        return self.connection, ("127.0.0.1", 50000)

    def close(self):
        self.closed = True


class FakePClient:
    def __init__(
        self,
        rail_lines,
        declared_count=None,
        initialized_graph=None,
    ):
        self.RAILLINE_DIC = rail_lines
        self.RAILINE_COUNT = (
            len(rail_lines) if declared_count is None else declared_count
        )
        self.initialized_graph = initialized_graph
        self.standard_data_calls = 0

    def WriteAdminLog(self, message):
        pass

    def RecieveSimulationStandardData(self):
        self.standard_data_calls += 1
        if self.initialized_graph is None:
            raise AssertionError("unexpected wait for simulator initialization")
        self.RAILLINE_DIC = self.initialized_graph
        self.RAILINE_COUNT = len(self.initialized_graph)
        return 2


class CheckRailTopologyTests(unittest.TestCase):
    def test_default_output_dir_is_fixed_topology_cache_directory(self):
        project_root = Path(__file__).resolve().parents[2]
        self.assertEqual(
            check_rail_topology.DEFAULT_OUTPUT_DIR,
            project_root
            / "PythonCode"
            / "oht_routing"
            / "topology"
            / "cache",
        )

    def run_audit(
        self,
        graph,
        directory,
        declared_count=None,
        initialized_graph=None,
    ):
        server = FakeServerSocket()
        live_client = FakePClient(
            graph,
            declared_count=declared_count,
            initialized_graph=initialized_graph,
        )
        with (
            patch.object(
                check_rail_topology.socket,
                "socket",
                return_value=server,
            ),
            patch.object(
                check_rail_topology.PClient,
                "PClient",
                return_value=live_client,
            ),
        ):
            status = check_rail_topology.run_topology_audit(
                port=9100,
                output_dir=directory,
            )
        return status, server, live_client

    def test_pass_writes_audit_and_cache_then_closes_socket(self):
        graph = make_controlled_boundary_graph()
        with tempfile.TemporaryDirectory() as directory:
            status, server, _ = self.run_audit(graph, directory)
            audit_path = Path(directory) / "topology_neighbor_audit.json"
            cache_path = Path(directory) / "contextual_topology_cache.npz"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

            self.assertEqual(status, 0)
            self.assertTrue(server.closed)
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["expected_rail_count"], len(graph))
            self.assertEqual(audit["controlled_rail_count"], 12)
            self.assertEqual(audit["boundary_rail_count"], 3)
            self.assertTrue(cache_path.is_file())

    def test_insufficient_neighbors_fails_without_cache(self):
        successors = {
            3250: [3251],
            3251: [3252],
            3252: [4000],
            4000: [4001],
            4001: [],
        }
        with tempfile.TemporaryDirectory() as directory:
            status, server, _ = self.run_audit(make_graph(successors), directory)
            audit_path = Path(directory) / "topology_neighbor_audit.json"
            cache_path = Path(directory) / "contextual_topology_cache.npz"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

            self.assertEqual(status, 2)
            self.assertTrue(server.closed)
            self.assertEqual(audit["status"], "failed")
            self.assertEqual(len(audit["deficient_rails"]), 5)
            self.assertFalse(cache_path.exists())

    def test_declared_and_loaded_count_mismatch_fails(self):
        successors = {1: [2], 2: [1]}
        with tempfile.TemporaryDirectory() as directory:
            status, _, _ = self.run_audit(
                make_graph(successors),
                directory,
                declared_count=3,
            )
            audit_path = Path(directory) / "topology_neighbor_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

            self.assertEqual(status, 2)
            self.assertEqual(audit["status"], "failed")
            self.assertIn("declared/loaded rail count mismatch", audit["error"])

    def test_zero_rail_handshake_waits_for_v2_reinitialization(self):
        initialized_graph = make_controlled_boundary_graph()
        with tempfile.TemporaryDirectory() as directory:
            status, server, live_client = self.run_audit(
                {},
                directory,
                declared_count=0,
                initialized_graph=initialized_graph,
            )
            audit_path = Path(directory) / "topology_neighbor_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

            self.assertEqual(status, 0)
            self.assertTrue(server.closed)
            self.assertEqual(live_client.standard_data_calls, 1)
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["rail_count"], len(initialized_graph))


if __name__ == "__main__":
    unittest.main()
