import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contextual_observation import (
    GLOBAL_DIM,
    LOCAL_DIM,
    RELATION_DIM,
    ContextualObservationBuilder,
    ObservationContractError,
    ObservationNormalizerConfig,
)
from contextual_topology import ContextualTopology


BOUNDARY_IDS = (3250, 3251, 3252)
PHYSICAL_COUNT = 4999
CONTROLLED_COUNT = 4996


@dataclass
class FakeRail:
    ID: int
    OhtList: list[int] = field(default_factory=list)
    IdleOHTCount: float = 0.0
    PredictedOHTCount: float = 0.0
    DistancePerVelocity: float = 1.0
    PortCount: float = 1.0
    DivergingLineCount: float = 1.0


@dataclass
class FakeJob:
    State: int
    ReAssignCount: float


class FakePClient:
    def __init__(self, rail_lines):
        self.RAILLINE_DIC = rail_lines
        self.TotalTat = 120.0
        self.TotalOhtOperationRate = 0.75
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

    offsets = np.arange(1, 11, dtype=np.int64)
    incoming = (
        controlled_to_physical[:, None] - offsets[None, :]
    ) % PHYSICAL_COUNT
    outgoing = (
        controlled_to_physical[:, None] + offsets[None, :]
    ) % PHYSICAL_COUNT
    # Prove a boundary rail can be a source for a controlled center.
    incoming[0, 0] = BOUNDARY_IDS[0]
    hops = np.broadcast_to(offsets, incoming.shape).astype(np.int32).copy()
    incoming_time = hops.astype(np.float64) * 0.5
    outgoing_time = hops.astype(np.float64) * 0.75
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
        incoming_travel_time=incoming_time,
        outgoing_travel_time=outgoing_time,
        topology_hash="a" * 64,
        mapping_hash="b" * 64,
        audit={},
    )


def make_rails(reverse=False):
    ids = list(range(PHYSICAL_COUNT))
    if reverse:
        ids.reverse()
    rails = {}
    for rail_id in ids:
        rails[rail_id] = FakeRail(
            ID=rail_id,
            OhtList=[0] * (rail_id % 5),
            IdleOHTCount=float(rail_id % 3),
            PredictedOHTCount=float(rail_id % 7),
            DistancePerVelocity=1.0 + (rail_id % 11) * 0.1,
            PortCount=float(1 + rail_id % 4),
            DivergingLineCount=float(1 + rail_id % 2),
        )
    return rails


class ContextualObservationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.topology = make_topology()
        self.cache_path = Path(self.directory.name) / "topology.npz"
        self.topology.save_cache(self.cache_path)
        self.parameter_dw = {
            rail_id: 1.0 + (rail_id % 13) * 0.01
            for rail_id in range(PHYSICAL_COUNT)
        }
        self.parameter_c = {
            rail_id: float(rail_id % 9) for rail_id in range(PHYSICAL_COUNT)
        }

    def builder(self, **kwargs):
        return ContextualObservationBuilder(
            self.topology, cache_path=self.cache_path, **kwargs
        )

    def build(self, builder=None, reverse=False):
        builder = builder or self.builder()
        pclient = FakePClient(make_rails(reverse=reverse))
        batch = builder.build(
            pclient,
            parameter_dw=self.parameter_dw,
            parameter_c=self.parameter_c,
        )
        return builder, pclient, batch

    def test_exact_output_shapes_finite_and_contiguous(self):
        _, _, batch = self.build()
        expected = {
            "center_local": (4996, 8),
            "incoming_local": (4996, 10, 8),
            "outgoing_local": (4996, 10, 8),
            "incoming_relation": (4996, 10, 2),
            "outgoing_relation": (4996, 10, 2),
            "global_state": (6,),
        }
        for name, shape in expected.items():
            array = getattr(batch, name)
            self.assertEqual(array.shape, shape)
            self.assertTrue(np.isfinite(array).all())
            self.assertTrue(array.flags.c_contiguous)

    def test_boundary_is_not_center_but_can_be_neighbor(self):
        _, _, batch = self.build()
        self.assertTrue(
            set(BOUNDARY_IDS).isdisjoint(batch.controlled_rail_ids.tolist())
        )
        boundary_row = BOUNDARY_IDS[0]
        self.assertEqual(self.topology.incoming_neighbor_ids[0, 0], boundary_row)
        expected_oht_count = float(boundary_row % 5)
        self.assertEqual(batch.incoming_local[0, 0, 0], expected_oht_count)

    def test_controlled_rows_and_neighbor_gathers_align_by_explicit_id_lookup(self):
        _, _, batch = self.build()
        row = 123
        rail_id = int(self.topology.controlled_rail_ids[row])
        self.assertEqual(batch.controlled_rail_ids[row], rail_id)
        self.assertEqual(batch.center_local[row, 0], float(rail_id % 5))
        incoming_id = int(self.topology.incoming_neighbor_ids[row, 4])
        outgoing_id = int(self.topology.outgoing_neighbor_ids[row, 6])
        self.assertEqual(batch.incoming_local[row, 4, 0], float(incoming_id % 5))
        self.assertEqual(batch.outgoing_local[row, 6, 0], float(outgoing_id % 5))

    def test_dictionary_and_iteration_order_do_not_change_result(self):
        _, _, normal = self.build(reverse=False)
        _, _, reversed_result = self.build(reverse=True)
        for name in (
            "center_local",
            "incoming_local",
            "outgoing_local",
            "global_state",
        ):
            np.testing.assert_array_equal(
                getattr(normal, name), getattr(reversed_result, name)
            )

    def test_local_normalizer_updates_once_with_each_physical_rail(self):
        builder, _, _ = self.build()
        self.assertEqual(builder.local_normalizer.update_calls, 1)
        self.assertEqual(builder.local_normalizer.count, PHYSICAL_COUNT)
        self.assertEqual(builder.global_normalizer.update_calls, 1)
        self.assertEqual(builder.global_normalizer.count, 1)

    def test_frequent_neighbor_does_not_receive_duplicate_statistical_weight(self):
        builder, pclient, _ = self.build()
        expected = np.mean(
            [len(pclient.RAILLINE_DIC[rail_id].OhtList) for rail_id in range(4999)]
        )
        self.assertAlmostEqual(builder.local_normalizer.mean[0], expected)
        self.assertEqual(builder.local_normalizer.count, 4999)

    def test_same_step_uses_preupdate_statistics_for_every_rail(self):
        builder = self.builder(
            normalizer_config=ObservationNormalizerConfig(clip=None)
        )
        _, _, first = self.build(builder=builder)
        physical_ids = self.topology.controlled_rail_ids
        np.testing.assert_array_equal(
            first.center_local[:, 0],
            (physical_ids % 5).astype(np.float32),
        )

    def test_global_is_separate_and_not_appended_to_local_tokens(self):
        builder = self.builder(
            normalizer_config=ObservationNormalizerConfig(clip=None)
        )
        _, _, batch = self.build(builder=builder)
        self.assertEqual(batch.center_local.shape[-1], LOCAL_DIM)
        self.assertEqual(batch.incoming_local.shape[-1], LOCAL_DIM)
        self.assertEqual(batch.global_state.shape, (GLOBAL_DIM,))
        np.testing.assert_array_equal(
            batch.global_state,
            np.asarray([120.0, 0.75, 1.0, 1.0, 1.0, 2.0], dtype=np.float32),
        )

    def test_relation_matches_static_topology_and_is_frozen(self):
        builder, _, batch = self.build()
        raw = np.stack(
            (
                self.topology.incoming_hops,
                self.topology.incoming_travel_time,
            ),
            axis=-1,
        )
        expected = builder.relation_normalizer.normalize(
            raw.reshape(-1, RELATION_DIM)
        ).reshape(raw.shape)
        np.testing.assert_allclose(batch.incoming_relation, expected)
        self.assertTrue(builder.relation_normalizer.frozen)
        self.assertEqual(builder.relation_normalizer.update_calls, 1)
        count = builder.relation_normalizer.count
        self.build(builder=builder)
        self.assertEqual(builder.relation_normalizer.count, count)

    def test_nan_and_inf_raw_input_fail_fast(self):
        for value in (np.nan, np.inf):
            pclient = FakePClient(make_rails())
            pclient.RAILLINE_DIC[10].PredictedOHTCount = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ObservationContractError, "NaN or Inf"):
                    self.builder().build(
                        pclient,
                        parameter_dw=self.parameter_dw,
                        parameter_c=self.parameter_c,
                    )

    def test_normalizer_save_load_reproduces_same_next_input(self):
        first = self.builder()
        self.build(builder=first)
        path = Path(self.directory.name) / "normalizers.npz"
        first.save_normalizers(path)
        second = self.builder()
        second.load_normalizers(path)
        _, _, first_result = self.build(builder=first)
        _, _, second_result = self.build(builder=second)
        for name in (
            "center_local",
            "incoming_local",
            "outgoing_local",
            "global_state",
        ):
            np.testing.assert_array_equal(
                getattr(first_result, name), getattr(second_result, name)
            )

    def _write_identity_cache(self, *, topology_hash=None, mapping_hash=None):
        path = Path(self.directory.name) / "mismatch.npz"
        np.savez_compressed(
            path,
            topology_version=np.asarray(
                "directed_10in_10out_controlled_centers_v2"
            ),
            topology_hash=np.asarray(topology_hash or self.topology.topology_hash),
            mapping_hash=np.asarray(mapping_hash or self.topology.mapping_hash),
        )
        return path

    def test_runtime_cache_topology_hash_mismatch_fails(self):
        path = self._write_identity_cache(topology_hash="c" * 64)
        with self.assertRaisesRegex(ObservationContractError, "topology hash mismatch"):
            ContextualObservationBuilder(self.topology, cache_path=path)

    def test_runtime_cache_mapping_hash_mismatch_fails(self):
        path = self._write_identity_cache(mapping_hash="d" * 64)
        with self.assertRaisesRegex(ObservationContractError, "mapping hash mismatch"):
            ContextualObservationBuilder(self.topology, cache_path=path)

    def test_builder_does_not_modify_input_rail_objects(self):
        pclient = FakePClient(make_rails())
        before = {
            rail_id: (
                list(rail.OhtList),
                rail.IdleOHTCount,
                rail.PredictedOHTCount,
                rail.DistancePerVelocity,
                rail.PortCount,
                rail.DivergingLineCount,
            )
            for rail_id, rail in pclient.RAILLINE_DIC.items()
        }
        self.builder().build(
            pclient,
            parameter_dw=self.parameter_dw,
            parameter_c=self.parameter_c,
        )
        after = {
            rail_id: (
                list(rail.OhtList),
                rail.IdleOHTCount,
                rail.PredictedOHTCount,
                rail.DistancePerVelocity,
                rail.PortCount,
                rail.DivergingLineCount,
            )
            for rail_id, rail in pclient.RAILLINE_DIC.items()
        }
        self.assertEqual(before, after)

    def test_configured_warmup_freezes_local_and_global_normalizers(self):
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
        count = builder.local_normalizer.count
        self.build(builder=builder)
        self.assertEqual(builder.local_normalizer.count, count)


if __name__ == "__main__":
    unittest.main()
