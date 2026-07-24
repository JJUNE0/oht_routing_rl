import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cocel_rl.algorithms.contextual_td7 import (
    ContextualActor,
    ContextualNetworkConfig,
    SALEOnline,
)
from ClientAlgorithm_contextual import (
    ClientAlgorithm,
    ContextualRuntimeConfig,
)
from contextual_action import EXP_RESIDUAL
from contextual_observation import ContextualObservationBatch
from test_contextual_observation import (
    CONTROLLED_COUNT,
    PHYSICAL_COUNT,
    FakePClient,
    make_rails,
    make_topology,
)


class CountingObservationBuilder:
    def __init__(self, topology):
        self.topology = topology
        self.calls = 0
        self.batch = ContextualObservationBatch(
            center_local=np.zeros((CONTROLLED_COUNT, 8), dtype=np.float32),
            incoming_local=np.zeros(
                (CONTROLLED_COUNT, 10, 8), dtype=np.float32
            ),
            outgoing_local=np.zeros(
                (CONTROLLED_COUNT, 10, 8), dtype=np.float32
            ),
            incoming_relation=np.zeros(
                (CONTROLLED_COUNT, 10, 2), dtype=np.float32
            ),
            outgoing_relation=np.zeros(
                (CONTROLLED_COUNT, 10, 2), dtype=np.float32
            ),
            global_state=np.zeros(6, dtype=np.float32),
            controlled_rail_ids=topology.controlled_rail_ids.copy(),
            topology_hash=topology.topology_hash,
            mapping_hash=topology.mapping_hash,
        )

    def build(self, pclient, *, parameter_dw, parameter_c):
        self.calls += 1
        return self.batch


def make_runtime_pclient(reverse=False):
    rails = make_rails(reverse=reverse)
    pclient = FakePClient(rails)
    pclient.RAILINE_COUNT = PHYSICAL_COUNT
    pclient.RAILLINECOST_DIC = {
        rail_id: SimpleNamespace(ID=rail_id, FRailLineCost=0.0)
        for rail_id in (reversed(range(PHYSICAL_COUNT)) if reverse else range(PHYSICAL_COUNT))
    }
    pclient.OHT_DIC = {}
    pclient.sent_is_end = []
    pclient.SendIsEnd = pclient.sent_is_end.append
    return pclient


class ContextualRuntimeTests(unittest.TestCase):
    def runtime(self, **overrides):
        values = {
            "mode": "baseline_only",
            "action_enabled": False,
            "action_scale": 0.0,
            "warmup_steps": 0,
            "normalizer_freeze_steps": 10,
            "device": "cpu",
            "seed": 7,
        }
        values.update(overrides)
        runtime = ClientAlgorithm(ContextualRuntimeConfig(**values))
        runtime.topology = make_topology()
        runtime.observation_builder = CountingObservationBuilder(runtime.topology)
        return runtime

    @staticmethod
    def costs_in_physical_order(runtime, pclient):
        return np.asarray(
            [
                pclient.RAILLINECOST_DIC[int(rail_id)].FRailLineCost
                for rail_id in runtime.topology.all_rail_ids
            ]
        )

    def test_baseline_tick_builds_observation_exactly_once_and_is_finite(self):
        runtime = self.runtime()
        pclient = make_runtime_pclient()
        result = runtime.Algorithm(pclient)
        self.assertEqual(pclient.sent_is_end, [0])
        self.assertEqual(runtime.observation_builder.calls, 1)
        self.assertEqual(
            runtime.last_diagnostics[
                "runtime/observation_build_calls_per_tick"
            ],
            1.0,
        )
        costs = self.costs_in_physical_order(runtime, pclient)
        self.assertTrue(np.isfinite(costs).all())
        np.testing.assert_array_equal(costs, result.final_cost)

    def test_rail_dictionary_order_does_not_change_cost_alignment(self):
        first = self.runtime()
        second = self.runtime()
        normal = make_runtime_pclient(False)
        reversed_client = make_runtime_pclient(True)
        first.Algorithm(normal)
        second.Algorithm(reversed_client)
        np.testing.assert_array_equal(
            self.costs_in_physical_order(first, normal),
            self.costs_in_physical_order(second, reversed_client),
        )

    def test_actor_runtime_uses_eval_inference_mode_and_no_attention_copy(self):
        runtime = self.runtime(mode="actor_inference", action_enabled=False)
        pclient = make_runtime_pclient()
        captured = {}
        original_actor_forward = runtime.actor.forward
        original_encoder_forward = runtime.encoder.forward

        def actor_forward(state):
            captured["inference_mode"] = torch.is_inference_mode_enabled()
            captured["actor_training"] = runtime.actor.training
            return original_actor_forward(state)

        def encoder_forward(*args, **kwargs):
            captured["return_attention"] = kwargs.get("return_attention")
            output = original_encoder_forward(*args, **kwargs)
            captured["attention_absent"] = (
                output.incoming_attention is None
                and output.outgoing_attention is None
            )
            return output

        with (
            patch.object(runtime.actor, "forward", side_effect=actor_forward),
            patch.object(runtime.encoder, "forward", side_effect=encoder_forward),
        ):
            runtime.Algorithm(pclient)
        self.assertTrue(captured["inference_mode"])
        self.assertFalse(captured["actor_training"])
        self.assertFalse(runtime.encoder.training)
        self.assertFalse(captured["return_attention"])
        self.assertTrue(captured["attention_absent"])
        self.assertFalse(hasattr(runtime, "critic"))

    def test_sale_runtime_expands_global_state_for_every_controlled_rail(self):
        runtime = self.runtime(mode="actor_inference", action_enabled=True)
        network_config = ContextualNetworkConfig()
        runtime.actor = ContextualActor(
            network_config, sale_embedding_dim=16, sale_feature_dim=16
        )
        sale_fixed = SALEOnline(network_config, embedding_dim=16)
        runtime.learner = SimpleNamespace(
            sale_fixed=sale_fixed,
            config=SimpleNamespace(sale_enabled=True),
        )
        captured = {}
        original = runtime.learner.sale_fixed.state

        def sale_state(observation):
            captured["global_shape"] = tuple(observation[-1].shape)
            return original(observation)

        with patch.object(
            runtime.learner.sale_fixed, "state", side_effect=sale_state
        ):
            action, _ = runtime._actor_inference(
                runtime.observation_builder.batch
            )
        self.assertEqual(captured["global_shape"], (4996, 6))
        self.assertEqual(action.shape, (4996,))
        self.assertTrue(np.isfinite(action).all())

    def test_warmup_keeps_all_costs_at_baseline_then_applies(self):
        runtime = self.runtime(
            mode="actor_inference",
            action_enabled=True,
            action_scale=0.2,
            action_mode=EXP_RESIDUAL,
            warmup_steps=1,
        )
        pclient = make_runtime_pclient()
        first = runtime.Algorithm(pclient)
        np.testing.assert_array_equal(first.final_cost, self.costs_in_physical_order(runtime, pclient))
        # Baseline contains only DistancePerVelocity in this empty-OHT fixture.
        baseline = np.asarray(
            [
                pclient.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
                for rail_id in runtime.topology.all_rail_ids
            ]
        )
        np.testing.assert_array_equal(first.final_cost, baseline)
        second = runtime.Algorithm(pclient)
        controlled_rows = runtime.topology.controlled_row_to_physical_index
        self.assertTrue(
            np.any(second.final_cost[controlled_rows] != baseline[controlled_rows])
        )

    def test_reset_clears_episode_runtime_state_without_training_state(self):
        runtime = self.runtime()
        pclient = make_runtime_pclient()
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.episode_steps, 1)
        runtime.Reset(pclient)
        self.assertEqual(runtime.episode_steps, 0)
        self.assertIsNone(runtime.last_observation)
        self.assertIsNone(runtime.last_controlled_action)
        self.assertEqual(runtime.last_diagnostics, {})
        self.assertEqual(runtime.total_steps, 1)

    def test_boundary_invariant_holds_for_100_consecutive_ticks(self):
        runtime = self.runtime(
            mode="actor_inference",
            action_enabled=True,
            action_scale=0.3,
        )
        pclient = make_runtime_pclient()
        boundary_rows = np.flatnonzero(
            runtime.topology.physical_index_to_controlled_row == -1
        )
        baseline = np.asarray(
            [
                pclient.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
                for rail_id in runtime.topology.all_rail_ids
            ]
        )
        for _ in range(100):
            result = runtime.Algorithm(pclient)
            np.testing.assert_array_equal(
                result.final_cost[boundary_rows], baseline[boundary_rows]
            )
            self.assertEqual(
                result.diagnostics[
                    "boundary/cost_baseline_abs_error_max"
                ],
                0.0,
            )
        self.assertEqual(runtime.observation_builder.calls, 100)

    def test_same_state_and_seed_produce_deterministic_actions(self):
        first = self.runtime(mode="actor_inference")
        second = self.runtime(mode="actor_inference")
        first.Algorithm(make_runtime_pclient())
        second.Algorithm(make_runtime_pclient())
        np.testing.assert_array_equal(
            first.last_controlled_action, second.last_controlled_action
        )

    def test_action_disabled_and_scale_zero_runtime_parity(self):
        disabled = self.runtime(
            mode="actor_inference", action_enabled=False, action_scale=1.0
        )
        zero = self.runtime(
            mode="actor_inference", action_enabled=True, action_scale=0.0
        )
        disabled_client = make_runtime_pclient()
        zero_client = make_runtime_pclient()
        disabled_result = disabled.Algorithm(disabled_client)
        zero_result = zero.Algorithm(zero_client)
        baseline = np.asarray(
            [
                disabled_client.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
                for rail_id in disabled.topology.all_rail_ids
            ]
        )
        np.testing.assert_array_equal(disabled_result.final_cost, baseline)
        np.testing.assert_array_equal(zero_result.final_cost, baseline)


if __name__ == "__main__":
    unittest.main()
