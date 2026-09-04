import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualActor,
    ContextualNetworkConfig,
    SALEOnline,
)
from oht_routing.runtime.client import (
    ClientAlgorithm,
    ContextualRuntimeConfig,
)
from oht_dispatching.config import DISPATCH_COST
from oht_routing.mdp.action import (
    EXP_RESIDUAL,
    FREE_FLOW_RESIDUAL,
    REGION_B_RL,
)
from oht_routing.mdp.observation import (
    CRITIC_EXTRA_DIM,
    GLOBAL_DIM,
    LOCAL_PHYSICAL_DIM,
    ContextualObservationBatch,
)
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
        neighbor_count = topology.incoming_neighbor_ids.shape[1]
        physical_by_id = {
            int(rail_id): row
            for row, rail_id in enumerate(topology.all_rail_ids)
        }
        incoming_indices = np.asarray([
            [physical_by_id[int(rail_id)] for rail_id in row]
            for row in topology.incoming_neighbor_ids
        ], dtype=np.int64)
        outgoing_indices = np.asarray([
            [physical_by_id[int(rail_id)] for rail_id in row]
            for row in topology.outgoing_neighbor_ids
        ], dtype=np.int64)
        self.batch = ContextualObservationBatch(
            center_local=np.zeros(
                (CONTROLLED_COUNT, LOCAL_PHYSICAL_DIM), dtype=np.float32
            ),
            incoming_local=np.zeros(
                (CONTROLLED_COUNT, neighbor_count, LOCAL_PHYSICAL_DIM),
                dtype=np.float32,
            ),
            outgoing_local=np.zeros(
                (CONTROLLED_COUNT, neighbor_count, LOCAL_PHYSICAL_DIM),
                dtype=np.float32,
            ),
            center_rail_index=np.ascontiguousarray(
                topology.controlled_row_to_physical_index.copy()
            ),
            incoming_rail_indices=np.ascontiguousarray(incoming_indices),
            outgoing_rail_indices=np.ascontiguousarray(outgoing_indices),
            incoming_relation=np.zeros(
                (CONTROLLED_COUNT, neighbor_count, 2), dtype=np.float32
            ),
            outgoing_relation=np.zeros(
                (CONTROLLED_COUNT, neighbor_count, 2), dtype=np.float32
            ),
            global_state=np.zeros(GLOBAL_DIM, dtype=np.float32),
            critic_total_tat=np.zeros(CRITIC_EXTRA_DIM, dtype=np.float32),
            previous_applied_action=np.zeros(
                (CONTROLLED_COUNT, 1), dtype=np.float32
            ),
            controlled_rail_ids=topology.controlled_rail_ids.copy(),
            topology_hash=topology.topology_hash,
            mapping_hash=topology.mapping_hash,
            critic_total_tat_raw=np.zeros(
                CRITIC_EXTRA_DIM, dtype=np.float32
            ),
        )

    def build(
        self, pclient, *, next_10_route_oht_count,
        recent_completed_tat_s=0.0,
        recent_completed_tat_available=False,
        previous_applied_action=None,
    ):
        self.calls += 1
        if previous_applied_action is None:
            return self.batch
        return replace(
            self.batch,
            previous_applied_action=np.ascontiguousarray(
                np.asarray(previous_applied_action, np.float32).reshape(-1, 1)
            ),
        )

    def diagnostics(self):
        return {}

    def reset_episode(self):
        pass


def make_runtime_pclient(reverse=False):
    rails = make_rails(reverse=reverse)
    pclient = FakePClient(rails)
    # Most runtime fixtures represent protocol clients without a monotonic
    # SimTime field. Stale-packet tests opt in explicitly.
    del pclient.SimTime
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
            "wandb_enabled": False,
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

    def test_default_runtime_artifact_paths(self):
        runtime = ClientAlgorithm(ContextualRuntimeConfig(device="cpu"))
        project_root = Path(__file__).resolve().parents[2]
        topology_cache_dir = (
            project_root / "PythonCode" / "oht_routing" / "topology" / "cache"
        )
        self.assertEqual(
            runtime.cache_path,
            topology_cache_dir / "contextual_topology_cache.npz",
        )
        self.assertEqual(
            runtime.audit_path,
            topology_cache_dir / "topology_neighbor_audit.json",
        )
        self.assertEqual(
            runtime.checkpoint_root,
            project_root / "checkpoints" / runtime.checkpoint_variant,
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

    def test_runtime_wires_locked_reward_q_profile(self):
        runtime = self.runtime()
        runtime._ensure_initialized(make_runtime_pclient())
        reward = runtime.reward_builder.config

        self.assertEqual(reward.reward_version, "Q")
        self.assertEqual(
            reward.contract.tat_signal_description,
            "one_sided_cumulative_total_tat_penalty",
        )
        self.assertEqual(reward.contract.tat_window_seconds, 0.0)
        self.assertEqual(reward.tat_reference, 165.0)
        self.assertEqual(reward.tat_weight, 4.0)
        self.assertEqual(reward.tat_window_seconds, 300.0)
        self.assertEqual(reward.op_weight, 0.0)
        self.assertFalse(reward.use_op)
        self.assertEqual(reward.backlog_weight, 0.0007)
        self.assertTrue(reward.backlog_growth_enabled)
        self.assertEqual(reward.backlog_growth_weight, 0.17)
        self.assertEqual(reward.idle_reserve_weight, 0.09)
        self.assertEqual(reward.local_oht_weight, 0.0)
        self.assertEqual(reward.local_predicted_oht_weight, 0.01)
        self.assertEqual(reward.local_stop_weight, 0.30)
        self.assertEqual(reward.local_density_weight, 5.5)
        self.assertEqual(reward.local_idle_weight, 0.0)
        self.assertEqual(reward.local_capacity_weight, 0.0)
        self.assertEqual(reward.rail_tat_weight, 660.0)
        self.assertEqual(reward.rail_tat_clip, 22.0)
        self.assertEqual(reward.rail_free_flow_neutral_ratio, 1.70)
        self.assertEqual(reward.smooth_b_rl_weight, 0.25)
        self.assertIsNone(runtime.reward_diagnostic_writer)
        self.assertIsNone(runtime.rail_tat_diagnostic_path)

    def test_reward_q_runtime_total_tat_dead_zone_and_penalty(self):
        for total_tat, expected_tat_raw in (
            (1.0, 0.0),
            (159.0, 0.0),
            (160.0, 0.0),
            (175.0, -4.0 * 15.0 / 165.0),
        ):
            with self.subTest(total_tat=total_tat):
                runtime = self.runtime()
                pclient = make_runtime_pclient()
                pclient.TotalTat = total_tat
                runtime.Algorithm(pclient)
                runtime.Algorithm(pclient)

                reward = runtime.transition_aligner.last_completed.reward
                self.assertEqual(reward.total_tat_level, total_tat)
                self.assertTrue(reward.tat_signal_available)
                self.assertAlmostEqual(reward.tat_raw, expected_tat_raw)
                if total_tat > 160.0:
                    self.assertLess(reward.tat_raw, 0.0)
                else:
                    self.assertEqual(reward.tat_raw, 0.0)

    def test_runtime_writes_bounded_reward_step_jsonl_only_when_opted_in(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(
                reward_diagnostic_dir=directory,
                reward_diagnostic_windows="0:10",
            )
            pclient = make_runtime_pclient()
            pclient.TotalTat = 175.0
            pclient.SimTime = 1.0
            runtime.Algorithm(pclient)
            pclient.SimTime = 2.0
            runtime.Algorithm(pclient)

            path = Path(directory) / "reward_step_window_00000_00010.jsonl"
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["global_step"], 1)
            self.assertEqual(record["episode_step"], 0)
            self.assertEqual(record["reward_version"], "Q")
            self.assertEqual(record["cumulative_total_tat"], 175.0)
            self.assertEqual(record["recent_completed_tat_300s_mean"], 0.0)
            self.assertAlmostEqual(record["tat_raw"], -4.0 * 15.0 / 165.0)
            completed_reward = runtime.transition_aligner.last_completed.reward
            self.assertEqual(completed_reward.total_tat_level, 175.0)
            self.assertEqual(
                completed_reward.cumulative_total_tat_level, 175.0
            )
            self.assertEqual(completed_reward.recent_completed_tat_mean, 0.0)
            self.assertEqual(record["tat_weight"], 4.0)
            self.assertEqual(record["backlog_weight"], 0.0007)
            self.assertEqual(record["local_predicted_oht_weight"], 0.01)
            self.assertEqual(record["local_reward_scale"], 2.0)
            self.assertEqual(record["rail_tat_weight"], 660.0)
            self.assertEqual(record["rail_tat_clip"], 22.0)
            for field in (
                "tat_raw_abs",
                "backlog_raw_abs",
                "global_component_abs",
                "tat_contribution_abs",
                "backlog_contribution_abs",
                "local_component_abs_mean",
                "rail_component_abs_mean",
                "smooth_component_abs_mean",
                "reward_budget_shares",
                "leading_indicator_snapshot",
            ):
                self.assertIn(field, record)
            self.assertEqual(
                set(record["reward_budget_shares"]),
                {"tat", "backlog", "local", "rail", "smooth"},
            )
            self.assertAlmostEqual(
                record["tat_contribution_abs"],
                0.5 * record["tat_raw_abs"],
            )
            self.assertAlmostEqual(
                record["backlog_contribution_abs"],
                0.5 * record["backlog_raw_abs"],
            )
            self.assertIn("rail_tat_top_abs", record)
            self.assertFalse(record["learner_available"])
            self.assertIsNone(record["q1_mean"])

    def test_cost_dispatch_snapshots_exact_command_zero_final_cost(self):
        runtime = self.runtime(dispatch_mode=DISPATCH_COST)
        pclient = make_runtime_pclient()

        result = runtime.Algorithm(pclient)

        expected = {
            int(rail_id): float(result.final_cost[row])
            for row, rail_id in enumerate(runtime.topology.all_rail_ids)
        }
        self.assertTrue(runtime.dispatcher.costs_ready)
        self.assertEqual(runtime.dispatcher.rail_costs, expected)

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

        def actor_forward(state, *args, **kwargs):
            captured["inference_mode"] = torch.is_inference_mode_enabled()
            captured["actor_training"] = runtime.actor.training
            return original_actor_forward(state, *args, **kwargs)

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

    def test_actor_uses_global_total_tat_not_direct_critic_copy(self):
        runtime = self.runtime(mode="actor_inference", action_enabled=True)
        actor_observation = runtime.observation_builder.batch
        low_global = actor_observation.global_state.copy()
        high_global = actor_observation.global_state.copy()
        low_global[0] = -3.0
        high_global[0] = 7.0
        low_actor_tat = replace(
            actor_observation,
            global_state=np.ascontiguousarray(low_global),
            critic_total_tat=np.asarray([-3.0], dtype=np.float32),
            critic_total_tat_raw=np.asarray([100.0], dtype=np.float32),
        )
        high_actor_tat = replace(
            actor_observation,
            global_state=np.ascontiguousarray(high_global),
            critic_total_tat=np.asarray([7.0], dtype=np.float32),
            critic_total_tat_raw=np.asarray([900.0], dtype=np.float32),
        )

        low_action, _ = runtime._actor_inference(low_actor_tat)
        high_action, _ = runtime._actor_inference(high_actor_tat)
        self.assertGreater(float(np.abs(low_action - high_action).sum()), 0.0)

        direct_low = replace(
            actor_observation,
            critic_total_tat=np.asarray([-3.0], dtype=np.float32),
        )
        direct_high = replace(
            actor_observation,
            critic_total_tat=np.asarray([7.0], dtype=np.float32),
        )
        direct_low_action, _ = runtime._actor_inference(direct_low)
        direct_high_action, _ = runtime._actor_inference(direct_high)
        np.testing.assert_array_equal(direct_low_action, direct_high_action)

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
        self.assertEqual(captured["global_shape"], (4996, GLOBAL_DIM))
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
        runtime.parameterDw = {0: 7.0}
        runtime.parameterPassTimes = {0: [3.0]}
        runtime.parameterC = {0: 5.0}
        self.assertEqual(runtime.episode_steps, 1)
        runtime.Reset(pclient)
        self.assertEqual(runtime.episode_steps, 0)
        self.assertIsNone(runtime.last_observation)
        self.assertIsNone(runtime.last_controlled_action)
        self.assertEqual(runtime.last_diagnostics, {})
        self.assertEqual(runtime.parameterDw, {})
        self.assertEqual(runtime.parameterPassTimes, {})
        self.assertEqual(runtime.parameterC, {})
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
        t_ff = np.asarray(
            [
                pclient.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
                for rail_id in runtime.topology.all_rail_ids
            ]
        )
        baseline = t_ff + 0.5
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

    def test_region_b_rl_uses_c_plus_one_at_zero_congestion(self):
        runtime = self.runtime(
            mode="actor_inference",
            action_enabled=True,
            action_mode=REGION_B_RL,
            curriculum_scale_start=1.0,
            curriculum_scale_end=1.0,
        )
        pclient = make_runtime_pclient()
        runtime._ensure_initialized(pclient)
        rows = runtime.topology.controlled_row_to_physical_index[:3]
        rail_ids = runtime.topology.all_rail_ids[rows]
        runtime.parameterDw.update({int(rail_id): 3.0 for rail_id in rail_ids})
        action = np.zeros(CONTROLLED_COUNT, dtype=np.float32)
        action[:3] = (-1.0, 0.0, 1.0)

        with patch.object(
            runtime, "_actor_inference", return_value=(action, {})
        ):
            result = runtime.Algorithm(pclient)

        t_ff = np.asarray([
            pclient.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
            for rail_id in rail_ids
        ])
        np.testing.assert_allclose(
            result.final_cost[rows],
            t_ff + 3.0 * np.asarray((0.0, 0.5, 1.0)),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            runtime._current_baseline[rows] - t_ff,
            np.full(3, 1.5),
            rtol=0.0,
            atol=1e-12,
        )

    def test_normalized_actor_action_reaches_free_flow_cost_without_scale(self):
        runtime = self.runtime(
            mode="actor_inference",
            action_enabled=True,
            action_mode=FREE_FLOW_RESIDUAL,
            action_scale=0.1,
            rl_cost_lambda=0.5,
        )
        pclient = make_runtime_pclient()
        action = np.zeros(CONTROLLED_COUNT, dtype=np.float32)
        action[:3] = (-1.0, 0.0, 1.0)
        with patch.object(
            runtime, "_actor_inference", return_value=(action, {})
        ):
            result = runtime.Algorithm(pclient)

        rows = runtime.topology.controlled_row_to_physical_index
        t_ff = np.asarray([
            pclient.RAILLINE_DIC[int(runtime.topology.all_rail_ids[row])]
            .DistancePerVelocity
            for row in rows[:3]
        ])
        np.testing.assert_allclose(
            result.final_cost[rows[:3]], t_ff * (0.5, 1.0, 1.5)
        )
        np.testing.assert_array_equal(
            result.applied_controlled_action, action
        )
        self.assertEqual(runtime._action_scale(), 1.0)

    def test_same_state_and_seed_produce_deterministic_actions(self):
        first = self.runtime(mode="actor_inference")
        second = self.runtime(mode="actor_inference")
        first.Algorithm(make_runtime_pclient())
        second.Algorithm(make_runtime_pclient())
        np.testing.assert_array_equal(
            first.last_controlled_action, second.last_controlled_action
        )

    def test_disabled_region_uses_strengthened_neutral_baseline(self):
        disabled = self.runtime(
            mode="actor_inference", action_enabled=False, action_scale=1.0
        )
        zero = self.runtime(
            mode="actor_inference",
            action_enabled=True,
            action_mode=EXP_RESIDUAL,
            action_scale=0.0,
        )
        disabled_client = make_runtime_pclient()
        zero_client = make_runtime_pclient()
        disabled_result = disabled.Algorithm(disabled_client)
        zero_result = zero.Algorithm(zero_client)
        t_ff = np.asarray(
            [
                disabled_client.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
                for rail_id in disabled.topology.all_rail_ids
            ]
        )
        np.testing.assert_array_equal(disabled_result.final_cost, t_ff + 0.5)
        np.testing.assert_array_equal(zero_result.final_cost, t_ff)


if __name__ == "__main__":
    unittest.main()
