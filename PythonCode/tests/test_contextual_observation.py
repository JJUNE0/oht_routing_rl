import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from oht_routing.mdp.observation import (
    GLOBAL_DIM,
    GLOBAL_FEATURE_NAMES,
    LOCAL_PHYSICAL_DIM,
    LOCAL_PHYSICAL_FEATURE_NAMES,
    RELATION_DIM,
    RELATION_FEATURE_NAMES,
    ContextualObservationBuilder,
    ObservationContractError,
    ObservationNormalizerConfig,
)
from oht_routing.mdp.topology import TOPOLOGY_VERSION, ContextualTopology
from oht_routing.version import CONTEXTUAL_VERSION


BOUNDARY_IDS = (3250, 3251, 3252)
PHYSICAL_COUNT = 4999
CONTROLLED_COUNT = PHYSICAL_COUNT - len(BOUNDARY_IDS)
NEIGHBOR_COUNT = 15


@dataclass
class FakeRail:
    ID: int
    Distance: float = 2_000.0
    OhtList: list[int] = field(default_factory=list)
    PredictedOHTCount: float = 0.0
    ReservationPortCount: float = 0.0
    DistancePerVelocity: float = 1.0
    PortCount: float = 1.0
    LevelJoiningLineIDList: list[int] = field(default_factory=list)
    DivergingLineIDList: list[int] = field(default_factory=list)


@dataclass
class FakeOHT:
    State: int
    StopTime: float


@dataclass
class FakeJob:
    State: int
    ReAssignCount: float


class FakePClient:
    def __init__(self, rail_lines, ohts=None):
        self.RAILLINE_DIC = rail_lines
        self.OHT_DIC = {} if ohts is None else ohts
        self.TotalTat = 120.0
        self.TotalOhtOperationRate = 0.75
        self.QueuedCommandCount = 3
        self.WaitingCommandCount = 2
        self.TransferCommandCount = 1
        self.CompletedCommandCount = 0
        self.SimTime = 0.0
        self.JOB_DIC = {
            1: FakeJob(1, 1),
            2: FakeJob(3, 2),
            3: FakeJob(5, 3),
        }


def make_topology():
    all_ids = np.arange(PHYSICAL_COUNT, dtype=np.int64)
    boundary = np.asarray(BOUNDARY_IDS, dtype=np.int64)
    controlled = np.asarray(
        [rail_id for rail_id in all_ids if rail_id not in set(BOUNDARY_IDS)],
        dtype=np.int64,
    )
    physical_to_controlled = np.full(PHYSICAL_COUNT, -1, dtype=np.int64)
    controlled_to_physical = controlled.copy()
    physical_to_controlled[controlled_to_physical] = np.arange(
        CONTROLLED_COUNT, dtype=np.int64
    )

    offsets = np.arange(1, NEIGHBOR_COUNT + 1, dtype=np.int64)
    incoming = (
        controlled_to_physical[:, None] - offsets[None, :]
    ) % PHYSICAL_COUNT
    outgoing = (
        controlled_to_physical[:, None] + offsets[None, :]
    ) % PHYSICAL_COUNT
    # A non-controlled boundary rail remains legal contextual input.
    incoming[0, 0] = BOUNDARY_IDS[0]
    hops = np.broadcast_to(offsets, incoming.shape).astype(np.int32).copy()
    return ContextualTopology(
        all_rail_ids=all_ids,
        controlled_rail_ids=controlled,
        boundary_rail_ids=boundary,
        physical_index_to_controlled_row=physical_to_controlled,
        controlled_row_to_physical_index=controlled_to_physical,
        incoming_neighbor_ids=np.ascontiguousarray(incoming),
        outgoing_neighbor_ids=np.ascontiguousarray(outgoing),
        incoming_hops=hops,
        outgoing_hops=hops.copy(),
        incoming_travel_time=hops.astype(np.float64) * 0.5,
        outgoing_travel_time=hops.astype(np.float64) * 0.75,
        topology_hash="a" * 64,
        mapping_hash="b" * 64,
        audit={},
    )


def make_runtime(reverse=False):
    ids = list(range(PHYSICAL_COUNT))
    if reverse:
        ids.reverse()
    rails = {}
    for rail_id in ids:
        rails[rail_id] = FakeRail(
            ID=rail_id,
            Distance=float(1_000 * (1 + rail_id % 4)),
            PredictedOHTCount=float(rail_id % 9),
            ReservationPortCount=float(rail_id % 3),
            DistancePerVelocity=1.0 + (rail_id % 11) * 0.1,
            PortCount=float(1 + rail_id % 4),
            LevelJoiningLineIDList=list(range(1 + rail_id % 3)),
            DivergingLineIDList=list(range(1 + rail_id % 2)),
        )

    # Every OHT is placed on exactly one physical rail, matching the live
    # simulator contract used to derive per-rail composition and congestion.
    definitions = (
        (100, 0, 0.0, 10),
        (101, 1, 2.0, 10),
        (102, 2, 3.0, 20),
        (103, 3, 0.0, 30),
        (104, 4, 4.0, 40),
        (105, 5, 0.0, 50),
    )
    ohts = {}
    iterable = reversed(definitions) if reverse else definitions
    for oht_id, state, stop_time, rail_id in iterable:
        ohts[oht_id] = FakeOHT(State=state, StopTime=stop_time)
        rails[rail_id].OhtList.append(oht_id)
    return rails, ohts


def make_rails(reverse=False):
    """Legacy reward/runtime fixture with no authoritative OHT dictionary."""
    ids = list(range(PHYSICAL_COUNT))
    if reverse:
        ids.reverse()
    rails = {}
    for rail_id in ids:
        rails[rail_id] = FakeRail(
            ID=rail_id,
            Distance=float(1_000 * (1 + rail_id % 4)),
            OhtList=[0] * (rail_id % 5),
            PredictedOHTCount=float(rail_id % 7),
            ReservationPortCount=float(rail_id % 3),
            DistancePerVelocity=1.0 + (rail_id % 11) * 0.1,
            PortCount=float(1 + rail_id % 4),
            LevelJoiningLineIDList=list(range(1 + rail_id % 3)),
            DivergingLineIDList=list(range(1 + rail_id % 2)),
        )
        rails[rail_id].IdleOHTCount = float(rail_id % 3)
        rails[rail_id].DivergingLineCount = float(1 + rail_id % 2)
    return rails


def route_ahead_counts():
    return {
        rail_id: float(rail_id % 9) for rail_id in range(PHYSICAL_COUNT)
    }


class ContextualObservationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.topology = make_topology()
        self.cache_path = Path(self.directory.name) / "topology.npz"
        self.topology.save_cache(self.cache_path)
        self.route_ahead = route_ahead_counts()

    def builder(self, **kwargs):
        return ContextualObservationBuilder(
            self.topology, cache_path=self.cache_path, **kwargs
        )

    def pclient(self, *, reverse=False):
        rails, ohts = make_runtime(reverse=reverse)
        return FakePClient(rails, ohts)

    def build(self, builder=None, *, reverse=False, previous=None):
        builder = builder or self.builder()
        pclient = self.pclient(reverse=reverse)
        batch = builder.build(
            pclient,
            next_10_route_oht_count=self.route_ahead,
            previous_applied_action=previous,
        )
        return builder, pclient, batch

    def test_v3_feature_contract_names_and_dimensions(self):
        self.assertEqual(LOCAL_PHYSICAL_DIM, 16)
        self.assertEqual(GLOBAL_DIM, 17)
        self.assertEqual(RELATION_DIM, 2)
        self.assertEqual(
            LOCAL_PHYSICAL_FEATURE_NAMES,
            (
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
            ),
        )
        self.assertEqual(
            GLOBAL_FEATURE_NAMES,
            (
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
            ),
        )
        self.assertEqual(
            RELATION_FEATURE_NAMES,
            ("directed_hop", "cumulative_free_flow_time_s"),
        )

    def test_exact_output_shapes_indices_finite_and_contiguous(self):
        _, _, batch = self.build()
        expected = {
            "center_local": (CONTROLLED_COUNT, LOCAL_PHYSICAL_DIM),
            "incoming_local": (
                CONTROLLED_COUNT,
                NEIGHBOR_COUNT,
                LOCAL_PHYSICAL_DIM,
            ),
            "outgoing_local": (
                CONTROLLED_COUNT,
                NEIGHBOR_COUNT,
                LOCAL_PHYSICAL_DIM,
            ),
            "center_rail_index": (CONTROLLED_COUNT,),
            "incoming_rail_indices": (CONTROLLED_COUNT, NEIGHBOR_COUNT),
            "outgoing_rail_indices": (CONTROLLED_COUNT, NEIGHBOR_COUNT),
            "incoming_relation": (CONTROLLED_COUNT, NEIGHBOR_COUNT, 2),
            "outgoing_relation": (CONTROLLED_COUNT, NEIGHBOR_COUNT, 2),
            "global_state": (GLOBAL_DIM,),
            "previous_applied_action": (CONTROLLED_COUNT, 1),
        }
        for name, shape in expected.items():
            array = getattr(batch, name)
            self.assertEqual(array.shape, shape, name)
            self.assertTrue(np.isfinite(array).all(), name)
            self.assertTrue(array.flags.c_contiguous, name)
        for name in (
            "center_rail_index",
            "incoming_rail_indices",
            "outgoing_rail_indices",
        ):
            self.assertTrue(np.issubdtype(getattr(batch, name).dtype, np.integer))

    def test_each_local_physical_feature_has_the_declared_meaning(self):
        pclient = self.pclient()
        physical, _ = self.builder().build_raw(
            pclient, next_10_route_oht_count=self.route_ahead
        )
        rail_id = 10
        expected = np.asarray(
            (
                2.0,
                3.0,
                2.0,
                1.0,
                2.0 / 3.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                1.0,
            ),
            dtype=np.float64,
        )
        np.testing.assert_allclose(physical[rail_id], expected)
        self.assertEqual(
            physical[rail_id, 8:14].sum(),
            len(pclient.RAILLINE_DIC[rail_id].OhtList),
        )

    def test_each_global_feature_has_the_declared_meaning(self):
        _, global_raw = self.builder().build_raw(
            self.pclient(), next_10_route_oht_count=self.route_ahead
        )
        expected = np.asarray(
            (
                120.0,
                0.75,
                3.0 / 6.0,
                2.0 / 6.0,
                1.0 / 6.0,
                2.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                3.0 / 6.0,
                9.0 / 6.0,
                0.0,
                0.0,
                0.0,
            ),
            dtype=np.float64,
        )
        np.testing.assert_allclose(global_raw, expected)

    def test_strict_sixty_second_trend_same_time_replace_and_reset(self):
        builder = self.builder()
        self.assertEqual(
            builder._trend_features(
                sim_time_s=0.0,
                backlog=10.0,
                total_tat_s=100.0,
                completed_count=99.0,
            ),
            (0.0, 0.0, 0.0),
        )
        for second in range(1, 60):
            trend = builder._trend_features(
                sim_time_s=float(second),
                backlog=10.0 + second,
                total_tat_s=100.0 + 2.0 * second,
                completed_count=1.0,
            )
            self.assertEqual(trend, (0.0, 0.0, 0.0))

        trend = builder._trend_features(
            sim_time_s=60.0,
            backlog=70.0,
            total_tat_s=220.0,
            completed_count=1.0,
        )
        np.testing.assert_allclose(trend, (60.0, 120.0, 1.0))

        replaced = builder._trend_features(
            sim_time_s=60.0,
            backlog=75.0,
            total_tat_s=230.0,
            completed_count=7.0,
        )
        np.testing.assert_allclose(replaced, (65.0, 130.0, 66.0 / 60.0))

        builder.reset_episode()
        self.assertEqual(
            builder._trend_features(
                sim_time_s=61.0,
                backlog=90.0,
                total_tat_s=300.0,
                completed_count=5.0,
            ),
            (0.0, 0.0, 0.0),
        )

    def test_sim_time_rollback_starts_a_new_trend_window(self):
        builder = self.builder()
        for second in range(61):
            builder._trend_features(
                sim_time_s=float(second),
                backlog=float(second),
                total_tat_s=float(second),
                completed_count=1.0,
            )
        self.assertEqual(
            builder._trend_features(
                sim_time_s=1.0,
                backlog=5.0,
                total_tat_s=7.0,
                completed_count=3.0,
            ),
            (0.0, 0.0, 0.0),
        )

    def test_predicted_route_ahead_diagnostics_are_exact(self):
        builder = self.builder()
        builder.build_raw(
            self.pclient(), next_10_route_oht_count=self.route_ahead
        )
        diagnostics = builder.diagnostics()
        self.assertAlmostEqual(
            diagnostics["observation/predicted_route10_pearson"], 1.0
        )
        self.assertEqual(diagnostics["observation/predicted_route10_mae"], 0.0)
        self.assertEqual(
            diagnostics["observation/predicted_route10_nonzero_agreement"], 1.0
        )
        expected_both_nonzero = sum(
            rail_id % 9 > 0 for rail_id in range(PHYSICAL_COUNT)
        ) / PHYSICAL_COUNT
        self.assertAlmostEqual(
            diagnostics["observation/predicted_route10_both_nonzero"],
            expected_both_nonzero,
        )

    def test_previous_action_is_separate_and_rail_indices_are_not_normalized(self):
        builder = self.builder()
        previous = np.linspace(
            -0.5, 0.5, CONTROLLED_COUNT, dtype=np.float32
        )[:, None]
        _, pclient, first = self.build(builder=builder, previous=previous)
        np.testing.assert_array_equal(first.previous_applied_action, previous)
        self.assertEqual(builder.local_normalizer.dim, LOCAL_PHYSICAL_DIM)
        second = builder.build(
            pclient,
            next_10_route_oht_count=self.route_ahead,
            previous_applied_action=previous,
        )
        for name in (
            "center_rail_index",
            "incoming_rail_indices",
            "outgoing_rail_indices",
        ):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        with self.assertRaisesRegex(ObservationContractError, "previous_applied_action"):
            builder.build(
                pclient,
                next_10_route_oht_count=self.route_ahead,
                previous_applied_action=np.full(
                    (CONTROLLED_COUNT, 1), 1.5, np.float32
                ),
            )

    def test_controlled_and_neighbor_gathers_use_explicit_physical_indices(self):
        _, _, batch = self.build()
        id_to_physical = {
            int(rail_id): row
            for row, rail_id in enumerate(self.topology.all_rail_ids)
        }
        row = 123
        center_id = int(self.topology.controlled_rail_ids[row])
        incoming_id = int(self.topology.incoming_neighbor_ids[row, 4])
        outgoing_id = int(self.topology.outgoing_neighbor_ids[row, 6])
        self.assertEqual(batch.center_rail_index[row], id_to_physical[center_id])
        self.assertEqual(
            batch.incoming_rail_indices[row, 4], id_to_physical[incoming_id]
        )
        self.assertEqual(
            batch.outgoing_rail_indices[row, 6], id_to_physical[outgoing_id]
        )
        np.testing.assert_array_equal(
            batch.center_local[row],
            batch.physical_local_raw[id_to_physical[center_id]],
        )

    def test_boundary_is_not_center_but_can_be_context_neighbor(self):
        _, _, batch = self.build()
        self.assertTrue(
            set(BOUNDARY_IDS).isdisjoint(batch.controlled_rail_ids.tolist())
        )
        self.assertEqual(batch.incoming_rail_indices[0, 0], BOUNDARY_IDS[0])

    def test_dictionary_iteration_order_does_not_change_result(self):
        _, _, normal = self.build(reverse=False)
        _, _, reversed_result = self.build(reverse=True)
        for name in (
            "center_local",
            "incoming_local",
            "outgoing_local",
            "center_rail_index",
            "incoming_rail_indices",
            "outgoing_rail_indices",
            "global_state",
        ):
            np.testing.assert_array_equal(
                getattr(normal, name), getattr(reversed_result, name)
            )

    def test_local_normalizer_updates_each_physical_rail_exactly_once(self):
        builder, _, _ = self.build()
        self.assertEqual(builder.local_normalizer.update_calls, 1)
        self.assertEqual(builder.local_normalizer.count, PHYSICAL_COUNT)
        self.assertEqual(builder.global_normalizer.update_calls, 1)
        self.assertEqual(builder.global_normalizer.count, 1)

    def test_global_is_separate_from_local_physical_tokens(self):
        _, _, batch = self.build(
            builder=self.builder(
                normalizer_config=ObservationNormalizerConfig(clip=None)
            )
        )
        self.assertEqual(batch.center_local.shape[-1], LOCAL_PHYSICAL_DIM)
        self.assertEqual(batch.global_state.shape, (GLOBAL_DIM,))

    def test_relation_matches_static_topology_and_is_frozen(self):
        builder, _, batch = self.build()
        raw = np.stack(
            (self.topology.incoming_hops, self.topology.incoming_travel_time),
            axis=-1,
        )
        expected = builder.relation_normalizer.normalize(
            raw.reshape(-1, RELATION_DIM)
        ).reshape(raw.shape)
        np.testing.assert_allclose(batch.incoming_relation, expected)
        self.assertTrue(builder.relation_normalizer.frozen)
        self.assertEqual(builder.relation_normalizer.update_calls, 1)

    def test_unknown_oht_reference_fails_closed(self):
        pclient = self.pclient()
        pclient.RAILLINE_DIC[0].OhtList.append(999_999)
        with self.assertRaisesRegex(ObservationContractError, "unknown OHT"):
            self.builder().build_raw(
                pclient, next_10_route_oht_count=self.route_ahead
            )

    def test_null_oht_state_fails_closed(self):
        pclient = self.pclient()
        pclient.OHT_DIC[100].State = 6
        with self.assertRaisesRegex(ObservationContractError, "unsupported state 6"):
            self.builder().build_raw(
                pclient, next_10_route_oht_count=self.route_ahead
            )

    def test_duplicate_oht_on_one_rail_fails_closed(self):
        pclient = self.pclient()
        pclient.RAILLINE_DIC[10].OhtList.append(100)
        with self.assertRaisesRegex(
            ObservationContractError, "duplicate|multiple|more than once"
        ):
            self.builder().build_raw(
                pclient, next_10_route_oht_count=self.route_ahead
            )

    def test_oht_placed_on_multiple_rails_fails_closed(self):
        pclient = self.pclient()
        pclient.RAILLINE_DIC[11].OhtList.append(100)
        with self.assertRaisesRegex(
            ObservationContractError, "duplicate|multiple|more than once"
        ):
            self.builder().build_raw(
                pclient, next_10_route_oht_count=self.route_ahead
            )

    def test_oht_missing_from_all_rails_fails_closed(self):
        pclient = self.pclient()
        pclient.RAILLINE_DIC[10].OhtList.remove(100)
        with self.assertRaisesRegex(ObservationContractError, "missing|not placed"):
            self.builder().build_raw(
                pclient, next_10_route_oht_count=self.route_ahead
            )

    def test_negative_stop_time_and_nonfinite_feature_fail_closed(self):
        pclient = self.pclient()
        pclient.OHT_DIC[100].StopTime = -1
        with self.assertRaisesRegex(ObservationContractError, "StopTime"):
            self.builder().build_raw(
                pclient, next_10_route_oht_count=self.route_ahead
            )
        pclient = self.pclient()
        pclient.RAILLINE_DIC[10].PredictedOHTCount = np.inf
        with self.assertRaisesRegex(ObservationContractError, "Predicted|NaN|Inf"):
            self.builder().build_raw(
                pclient, next_10_route_oht_count=self.route_ahead
            )

    def test_normalizer_save_load_uses_v3_physical_snapshot_keys(self):
        first = self.builder(
            normalizer_config=ObservationNormalizerConfig(
                freeze_after_env_steps=1
            )
        )
        self.build(builder=first)
        path = Path(self.directory.name) / "normalizers.npz"
        first.save_normalizers(path, require_frozen=True)
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(str(saved["version"].item()), CONTEXTUAL_VERSION)
            self.assertEqual(
                tuple(str(value) for value in saved["local_physical_feature_names"]),
                LOCAL_PHYSICAL_FEATURE_NAMES,
            )
            self.assertEqual(
                int(saved["local_physical_dim"].item()), LOCAL_PHYSICAL_DIM
            )
            self.assertNotIn("local_feature_names", saved.files)
            self.assertNotIn("local_dim", saved.files)

        second = self.builder()
        second.load_normalizers(path, require_frozen=True)
        np.testing.assert_array_equal(
            second.local_normalizer.mean, first.local_normalizer.mean
        )
        np.testing.assert_array_equal(
            second.global_normalizer.mean, first.global_normalizer.mean
        )

    def test_state_normalizer_feature_order_mismatch_fails_fast(self):
        first = self.builder(
            normalizer_config=ObservationNormalizerConfig(
                freeze_after_env_steps=1
            )
        )
        self.build(builder=first)
        path = Path(self.directory.name) / "reordered_normalizers.npz"
        first.save_normalizers(path, require_frozen=True)
        with np.load(path, allow_pickle=False) as saved:
            payload = {key: saved[key].copy() for key in saved.files}
        payload["local_physical_feature_names"] = np.asarray(
            tuple(reversed(LOCAL_PHYSICAL_FEATURE_NAMES))
        )
        np.savez_compressed(path, **payload)
        with self.assertRaisesRegex(ObservationContractError, "order mismatch"):
            self.builder().load_normalizers(path, require_frozen=True)

    def _write_identity_cache(self, *, topology_hash=None, mapping_hash=None):
        path = Path(self.directory.name) / "mismatch.npz"
        np.savez_compressed(
            path,
            topology_version=np.asarray(TOPOLOGY_VERSION),
            topology_hash=np.asarray(topology_hash or self.topology.topology_hash),
            mapping_hash=np.asarray(mapping_hash or self.topology.mapping_hash),
        )
        return path

    def test_runtime_cache_identity_mismatch_fails(self):
        with self.assertRaisesRegex(ObservationContractError, "topology hash mismatch"):
            ContextualObservationBuilder(
                self.topology,
                cache_path=self._write_identity_cache(topology_hash="c" * 64),
            )
        with self.assertRaisesRegex(ObservationContractError, "mapping hash mismatch"):
            ContextualObservationBuilder(
                self.topology,
                cache_path=self._write_identity_cache(mapping_hash="d" * 64),
            )

    def test_builder_does_not_modify_simulator_objects(self):
        pclient = self.pclient()
        before_rails = {
            rail_id: tuple(rail.OhtList)
            for rail_id, rail in pclient.RAILLINE_DIC.items()
        }
        before_ohts = {
            oht_id: (oht.State, oht.StopTime)
            for oht_id, oht in pclient.OHT_DIC.items()
        }
        self.builder().build_raw(
            pclient, next_10_route_oht_count=self.route_ahead
        )
        self.assertEqual(
            before_rails,
            {
                rail_id: tuple(rail.OhtList)
                for rail_id, rail in pclient.RAILLINE_DIC.items()
            },
        )
        self.assertEqual(
            before_ohts,
            {
                oht_id: (oht.State, oht.StopTime)
                for oht_id, oht in pclient.OHT_DIC.items()
            },
        )

    def test_configured_warmup_freezes_only_state_normalizers(self):
        builder = self.builder(
            normalizer_config=ObservationNormalizerConfig(
                freeze_after_env_steps=2
            )
        )
        self.build(builder=builder)
        self.assertFalse(builder.local_normalizer.frozen)
        self.build(builder=builder)
        self.assertTrue(builder.local_normalizer.frozen)
        self.assertTrue(builder.global_normalizer.frozen)
        self.assertTrue(builder.relation_normalizer.frozen)
        count = builder.local_normalizer.count
        self.build(builder=builder)
        self.assertEqual(builder.local_normalizer.count, count)


if __name__ == "__main__":
    unittest.main()
