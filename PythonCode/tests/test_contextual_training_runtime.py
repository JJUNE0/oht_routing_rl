import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ClientAlgorithm_contextual import ClientAlgorithm, ContextualRuntimeConfig
from ClientAlgorithm_contextual import ContextualTrainingFailure
from contextual_action import EXP_RESIDUAL, REGION_B_RL
from cocel_rl.algorithms.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualTD7Learner,
)
from cocel_rl.algorithms.contextual_td7.learner_types import (
    ContextualLearnerUpdate,
)
from contextual_observation import (
    ContextualObservationBatch,
    RunningFeatureNormalizer,
)
from test_contextual_observation import (
    CONTROLLED_COUNT,
    PHYSICAL_COUNT,
    make_topology,
)
from test_contextual_runtime import make_runtime_pclient
from contextual_wandb import (
    WANDB_METRIC_KEYS,
    ContextualWandbLogger,
    runtime_exp_meta,
)
from export_wandb_run import EXPORT_COLUMNS


class TrainingObservationBuilder:
    def __init__(self, topology, freeze_steps):
        self.topology = topology
        self.freeze_steps = freeze_steps
        self.calls = 0
        self.local_normalizer = RunningFeatureNormalizer(8)
        self.global_normalizer = RunningFeatureNormalizer(6)
        self._incoming_relation = np.zeros(
            (CONTROLLED_COUNT, 10, 2), np.float32
        )
        self._outgoing_relation = np.ones(
            (CONTROLLED_COUNT, 10, 2), np.float32
        )

    def build(self, pclient, *, parameter_dw, parameter_c):
        step = self.calls
        physical = np.full((PHYSICAL_COUNT, 8), step, np.float32)
        global_raw = np.full(6, step, np.float32)
        if not self.local_normalizer.frozen:
            self.local_normalizer.update(physical)
            self.global_normalizer.update(global_raw)
        self.calls += 1
        if self.calls >= self.freeze_steps:
            self.local_normalizer.freeze()
            self.global_normalizer.freeze()
        zeros = np.zeros((CONTROLLED_COUNT, 8), np.float32)
        return ContextualObservationBatch(
            center_local=zeros,
            incoming_local=np.zeros((CONTROLLED_COUNT, 10, 8), np.float32),
            outgoing_local=np.zeros((CONTROLLED_COUNT, 10, 8), np.float32),
            incoming_relation=self._incoming_relation,
            outgoing_relation=self._outgoing_relation,
            global_state=np.zeros(6, np.float32),
            controlled_rail_ids=self.topology.controlled_rail_ids.copy(),
            topology_hash=self.topology.topology_hash,
            mapping_hash=self.topology.mapping_hash,
            physical_local_raw=physical,
            global_raw=global_raw,
        )


class FakeLearner:
    def __init__(self):
        self.learner_update_count = 0
        self.actor_update_count = 0
        self.target_update_count = 0
        self.fail = False
        self.applied_action_scale = None

    def set_applied_action_scale(self, scale):
        self.applied_action_scale = float(scale)

    def update(self, batch):
        if self.fail:
            raise FloatingPointError("synthetic learner NaN")
        self.learner_update_count += 1
        actor = self.learner_update_count % 2 == 0
        target = self.learner_update_count % 250 == 0
        self.actor_update_count += int(actor)
        self.target_update_count += int(target)
        return ContextualLearnerUpdate(
            diagnostics={
                "learner/critic_loss": 0.1,
                "learner/actor_loss": 0.2 if actor else 0.0,
                "numeric/learner_finite_ratio": 1.0,
                "update/learner_count": float(self.learner_update_count),
                "update/actor_count": float(self.actor_update_count),
                "update/target_count": float(self.target_update_count),
                "learner/critic_action_matches_applied": 1.0,
            },
            actor_updated=actor,
            target_updated=target,
        )


def training_runtime(**overrides):
    values = {
        "mode": "training",
        "action_enabled": True,
        "action_scale": 0.05,
        "warmup_steps": 4,
        "normalizer_freeze_steps": 4,
        "device": "cpu",
        "seed": 12,
        "replay_capacity_env_steps": 64,
        "batch_size": 16,
        "minimum_replay_env_steps": 2,
        "minimum_action_enabled_env_steps": 2,
        "latest_checkpoint_interval": 10_000,
        "periodic_checkpoint_interval": 10_000,
        "wandb_enabled": False,
        "sale_enabled": False,
        "lap_enabled": False,
    }
    values.update(overrides)
    runtime = ClientAlgorithm(ContextualRuntimeConfig(**values))
    runtime.topology = make_topology()
    runtime.observation_builder = TrainingObservationBuilder(
        runtime.topology, runtime.config.normalizer_freeze_steps
    )
    pclient = make_runtime_pclient()
    runtime._ensure_initialized(pclient)
    runtime.learner = FakeLearner()
    return runtime, pclient


class ContextualTrainingRuntimeTests(unittest.TestCase):
    def test_default_checkpoint_root_isolated_by_runtime_variant(self):
        sale_lap_region = ClientAlgorithm(ContextualRuntimeConfig())
        uniform_region = ClientAlgorithm(
            ContextualRuntimeConfig(sale_enabled=False, lap_enabled=False)
        )
        sale_lap_residual = ClientAlgorithm(
            ContextualRuntimeConfig(action_mode=EXP_RESIDUAL)
        )

        variants = {
            sale_lap_region.runtime_variant,
            uniform_region.runtime_variant,
            sale_lap_residual.runtime_variant,
        }
        roots = {
            sale_lap_region.checkpoint_root,
            uniform_region.checkpoint_root,
            sale_lap_residual.checkpoint_root,
        }
        self.assertEqual(len(variants), 3)
        self.assertEqual(len(roots), 3)
        for runtime in (
            sale_lap_region, uniform_region, sale_lap_residual
        ):
            self.assertIn(runtime.algorithm_variant, runtime.runtime_variant)
            self.assertIn(runtime.action_version, runtime.runtime_variant)

    def test_wandb_tick_is_logged_only_after_send_timing_is_recorded(self):
        runtime, pclient = training_runtime(
            warmup_steps=0, normalizer_freeze_steps=1,
            wandb_log_interval=1,
        )
        captured = []
        runtime.wandb_logger.log = lambda diagnostics, step: captured.append(
            (dict(diagnostics), step)
        )

        runtime.Algorithm(pclient)
        self.assertEqual(captured, [])
        runtime.record_send_cost_ms(7.25)
        runtime.log_wandb_tick()

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0]["runtime/send_cost_ms"], 7.25)
        self.assertAlmostEqual(
            captured[0][0]["runtime/total_ms"],
            captured[0][0]["runtime/total_algorithm_ms"] + 7.25,
        )
        self.assertIn("runtime/checkpoint_ms", captured[0][0])
        self.assertEqual(captured[0][1], runtime.total_steps)

    def test_compatibility_diagnostics_reach_wandb_export_schema(self):
        runtime, pclient = training_runtime(
            warmup_steps=0, normalizer_freeze_steps=1
        )
        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)

        expected = {
            "reward/step_reward", "reward/global_norm",
            "reward/local_norm", "reward/rail_tat",
            "reward/rail_tat_event_count", "reward/rail_tat_sum",
            "reward/rail_tat_mean", "reward/rail_tat_max",
            "reward/alpha", "reward/local_normalized_std",
            "reward/marginal_tat_ema", "reward/tat_signal_available",
            "reward/marg_tat",
            "reward/backlog",
            "reward/smooth_mean", "curriculum/action_scale",
            "global/tat", "global/op_rate", "global/queued",
            "global/queued_jobs", "global/waiting", "global/transfer",
            "global/completed", "job/mean_wait_priority",
            "job/mean_reassign", "job/queued", "oht/idle_count",
            "oht/move_to_load", "oht/move_to_unload", "oht/loading",
            "oht/unloading",
        }
        self.assertTrue(expected <= runtime.last_diagnostics.keys())
        self.assertTrue(expected <= set(WANDB_METRIC_KEYS))
        self.assertTrue(expected <= set(EXPORT_COLUMNS))
        self.assertEqual(
            runtime.last_diagnostics["reward/step_reward"],
            runtime.last_diagnostics["reward/total_mean"],
        )
        self.assertEqual(
            runtime.last_diagnostics["reward/global_norm"],
            runtime.last_diagnostics["reward/global_normalized"],
        )
        self.assertNotIn(
            "reward/region_internal_std_mean", runtime.last_diagnostics
        )

    def test_training_requires_explicit_action_enabled(self):
        with self.assertRaises(ValueError):
            ContextualRuntimeConfig(
                mode="training", action_enabled=False, action_scale=0.05
            )

    def test_warmup_exploration_and_learning_gate(self):
        runtime, pclient = training_runtime(action_mode=EXP_RESIDUAL)
        baseline = np.asarray([
            pclient.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
            for rail_id in runtime.topology.all_rail_ids
        ])
        for _ in range(4):
            result = runtime.Algorithm(pclient)
            np.testing.assert_array_equal(result.final_cost, baseline)
            self.assertEqual(runtime.learner.learner_update_count, 0)
        # First enabled action has no completed enabled transition yet.
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.action_enabled_env_steps, 0)
        self.assertEqual(runtime.learner.learner_update_count, 0)
        self.assertGreater(runtime.last_diagnostics["action/exploratory_std"], 0)
        self.assertAlmostEqual(
            runtime.last_diagnostics["action/applied_std"],
            0.05 * runtime.last_diagnostics["action/exploratory_std"],
            places=6,
        )
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.action_enabled_env_steps, 1)
        self.assertEqual(runtime.learner.learner_update_count, 0)
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.action_enabled_env_steps, 2)
        self.assertEqual(runtime.learner.learner_update_count, 0)
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.learner.learner_update_count, 1)
        self.assertEqual(runtime.last_diagnostics["gate/open"], 1.0)

    def test_region_curriculum_is_exact_legacy_geometric_schedule(self):
        runtime, _ = training_runtime(
            action_mode=REGION_B_RL,
            warmup_steps=10_000,
            curriculum_end_step=40_000,
            curriculum_scale_start=0.05,
            curriculum_scale_end=1.0,
            curriculum_shape="geometric",
        )
        runtime.total_steps = 10_000
        self.assertAlmostEqual(runtime._action_scale(), 0.05)
        runtime.total_steps = 25_000
        self.assertAlmostEqual(runtime._action_scale(), np.sqrt(0.05))
        runtime.total_steps = 40_000
        self.assertAlmostEqual(runtime._action_scale(), 1.0)

    def test_one_replay_push_and_at_most_configured_updates_per_tick(self):
        runtime, pclient = training_runtime(
            warmup_steps=0, normalizer_freeze_steps=1,
            minimum_replay_env_steps=1,
            minimum_action_enabled_env_steps=1,
            updates_per_env_step=2,
        )
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.replay_buffer.push_count, 0)
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.replay_buffer.push_count, 1)
        self.assertEqual(runtime.learner.learner_update_count, 0)
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.replay_buffer.push_count, 2)
        self.assertEqual(runtime.learner.learner_update_count, 2)
        self.assertEqual(runtime.observation_builder.calls, 3)

    def test_replay_separates_policy_from_region_applied_action(self):
        runtime, pclient = training_runtime(
            action_mode=REGION_B_RL,
            warmup_steps=0,
            normalizer_freeze_steps=1,
            exploration_noise_std=0.0,
        )
        policy = np.full(CONTROLLED_COUNT, 0.4, dtype=np.float32)
        timing = {
            "runtime/tensor_conversion_ms": 0.0,
            "runtime/host_to_device_ms": 0.0,
            "runtime/encoder_actor_ms": 0.0,
            "runtime/device_to_host_ms": 0.0,
        }
        runtime._actor_inference = lambda *args, **kwargs: (
            policy.copy(), dict(timing)
        )
        runtime.Algorithm(pclient)
        pending = runtime.transition_aligner.pending
        np.testing.assert_array_equal(
            pending.controlled_action, policy[:, None]
        )
        np.testing.assert_allclose(
            pending.applied_action,
            (0.05 * policy)[:, None],
            rtol=0,
            atol=1e-7,
        )
        runtime.Algorithm(pclient)
        np.testing.assert_array_equal(
            runtime.replay_buffer._policy_action[0], policy
        )
        np.testing.assert_allclose(
            runtime.replay_buffer._applied_action[0],
            0.05 * policy,
            rtol=0,
            atol=1e-7,
        )

    def test_exploration_reproducible_and_boundary_always_baseline(self):
        first, p1 = training_runtime(warmup_steps=0)
        second, p2 = training_runtime(warmup_steps=0)
        r1 = first.Algorithm(p1)
        r2 = second.Algorithm(p2)
        np.testing.assert_array_equal(
            first.last_controlled_action, second.last_controlled_action
        )
        boundary = np.flatnonzero(
            first.topology.physical_index_to_controlled_row == -1
        )
        baseline = np.asarray([
            p1.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
            for rail_id in first.topology.all_rail_ids
        ])
        np.testing.assert_array_equal(r1.final_cost[boundary], baseline[boundary])
        np.testing.assert_array_equal(r2.final_cost[boundary], baseline[boundary])

    def test_numerical_failure_falls_back_then_raises_after_send_point(self):
        runtime, pclient = training_runtime(
            warmup_steps=0, normalizer_freeze_steps=1,
            minimum_replay_env_steps=1,
            minimum_action_enabled_env_steps=1,
        )
        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)
        pushes_before_failure = runtime.replay_buffer.push_count
        runtime.learner.fail = True
        crash_calls = []
        runtime._save_runtime_checkpoint = crash_calls.append
        result = runtime.Algorithm(pclient)
        baseline = np.asarray([
            pclient.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
            for rail_id in runtime.topology.all_rail_ids
        ])
        np.testing.assert_array_equal(result.final_cost, baseline)
        self.assertEqual(crash_calls, ["crash"])
        self.assertTrue(runtime.training_failed)
        self.assertIsNone(runtime.transition_aligner.pending)
        self.assertIsNone(runtime.transition_aligner.previous_applied_action)
        self.assertIsNone(runtime.transition_aligner.last_completed)
        # The only push completes the valid previous tick; no failed-tick
        # transition is staged for a later push.
        self.assertEqual(
            runtime.replay_buffer.push_count, pushes_before_failure
        )
        pushes_after_failure = runtime.replay_buffer.push_count
        with patch.object(
            runtime,
            "_actor_inference",
            side_effect=AssertionError("actor must not run after failure"),
        ):
            latched = runtime.Algorithm(pclient)
        np.testing.assert_array_equal(latched.final_cost, baseline)
        self.assertEqual(
            runtime.replay_buffer.push_count, pushes_after_failure
        )
        self.assertIsNone(runtime.transition_aligner.pending)
        with self.assertRaises(ContextualTrainingFailure):
            runtime.AlgorithmAfter(pclient)
        with self.assertRaisesRegex(
            ContextualTrainingFailure, "operator restart required"
        ):
            runtime.on_new_connection()

    def test_real_sale_lap_learner_crosses_runtime_gate(self):
        config = ContextualRuntimeConfig(
            mode="training",
            action_enabled=True,
            action_scale=0.05,
            warmup_steps=2,
            normalizer_freeze_steps=2,
            device="cpu",
            seed=33,
            replay_capacity_env_steps=8,
            batch_size=8,
            minimum_replay_env_steps=2,
            minimum_action_enabled_env_steps=2,
            latest_checkpoint_interval=10_000,
            periodic_checkpoint_interval=10_000,
            wandb_enabled=False,
            sale_enabled=True,
            lap_enabled=True,
        )
        runtime = ClientAlgorithm(config)
        runtime.topology = make_topology()
        runtime.observation_builder = TrainingObservationBuilder(
            runtime.topology, freeze_steps=2
        )
        pclient = make_runtime_pclient()
        runtime._ensure_initialized(pclient)
        self.assertIsInstance(runtime.learner, ContextualTD7Learner)

        result = None
        for _ in range(6):
            result = runtime.Algorithm(pclient)

        self.assertGreaterEqual(runtime.learner.learner_update_count, 1)
        self.assertTrue(np.isfinite(
            runtime.last_diagnostics["sale/loss"]
        ))
        self.assertTrue(np.isfinite(
            runtime.last_diagnostics["learner/critic_loss"]
        ))
        self.assertEqual(
            runtime.last_diagnostics[
                "learner/critic_action_matches_applied"
            ],
            1.0,
        )
        self.assertGreater(
            runtime.replay_buffer.priority_update_count, 0
        )
        self.assertTrue(np.isfinite(
            runtime.learner.current_target_q_min
        ))
        boundary = np.flatnonzero(
            runtime.topology.physical_index_to_controlled_row == -1
        )
        baseline = np.asarray([
            pclient.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
            for rail_id in runtime.topology.all_rail_ids
        ])
        np.testing.assert_array_equal(
            result.final_cost[boundary], baseline[boundary]
        )

    def test_wandb_disabled_does_not_import_or_initialize_run(self):
        with patch.dict("sys.modules", {"wandb": None}):
            runtime, _ = training_runtime(wandb_enabled=False)
        self.assertIsNone(runtime.wandb_logger.run)

    def test_latest_periodic_checkpoint_and_resume_refill_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, pclient = training_runtime(
                checkpoint_root=directory,
                latest_checkpoint_interval=2,
                periodic_checkpoint_interval=2,
                warmup_steps=0,
                normalizer_freeze_steps=1,
            )
            # Use the real learner for checkpoint state.
            runtime.learner = ContextualTD7Learner(
                runtime.replay_buffer,
                network_config=ContextualNetworkConfig(),
                config=ContextualLearnerConfig(
                    action_scale=0.05,
                    batch_size=16,
                    minimum_replay_env_steps=2,
                    minimum_action_enabled_env_steps=2,
                    sale_enabled=False,
                    lap_enabled=False,
                ),
                device="cpu",
                seed=12,
            )
            runtime.encoder = runtime.learner.encoder
            runtime.actor = runtime.learner.actor
            runtime.Algorithm(pclient)
            runtime.Algorithm(pclient)
            latest = Path(directory) / "latest" / "checkpoint.pt"
            periodic = Path(directory) / "periodic" / "step_00002.pt"
            self.assertTrue(latest.is_file())
            self.assertTrue(periodic.is_file())

            resumed, resumed_client = training_runtime(
                checkpoint_root=directory,
                resume_checkpoint_path=str(latest),
                latest_checkpoint_interval=10_000,
                periodic_checkpoint_interval=10_000,
                warmup_steps=0,
                normalizer_freeze_steps=1,
            )
            # training_runtime replaces the loaded learner with FakeLearner,
            # but initialization already restored runtime metadata and reset
            # refill counters.
            self.assertEqual(resumed.total_steps, 2)
            self.assertEqual(resumed.replay_buffer.size_env_steps, 0)
            self.assertEqual(resumed.action_enabled_env_steps, 0)
            self.assertFalse(resumed._training_gate()[1])
            self.assertTrue(resumed._resume_requires_refill)

            resumed.Algorithm(resumed_client)
            resumed.Algorithm(resumed_client)
            self.assertEqual(resumed.replay_buffer.size_env_steps, 1)
            self.assertEqual(resumed.action_enabled_env_steps, 1)
            self.assertFalse(resumed._training_gate()[1])

            resumed.Algorithm(resumed_client)
            self.assertEqual(resumed.replay_buffer.size_env_steps, 2)
            self.assertEqual(resumed.action_enabled_env_steps, 2)
            self.assertTrue(resumed._training_gate()[1])
            self.assertFalse(resumed._resume_requires_refill)

            # The gate is evaluated before the current transition is committed,
            # so learning starts on the tick after the refill threshold.
            resumed.Algorithm(resumed_client)
            self.assertEqual(resumed.learner.learner_update_count, 1)

    def test_job_priority_uses_real_priority_field(self):
        runtime, pclient = training_runtime()
        pclient.JOB_DIC = {
            1: SimpleNamespace(Priority=2, ReAssignCount=1),
            2: SimpleNamespace(Priority=4, ReAssignCount=3),
        }
        diagnostics = runtime._job_diagnostics(pclient)
        self.assertEqual(diagnostics["job/mean_wait_priority"], 3.0)
        self.assertEqual(diagnostics["job/mean_reassign"], 2.0)

    def test_early_stop_sends_end_and_does_not_stage_terminal_action(self):
        runtime, pclient = training_runtime(
            early_stop_queued_threshold=500,
        )
        pclient.QueuedCommandCount = 501
        runtime.Algorithm(pclient)
        self.assertEqual(pclient.sent_is_end, [1])
        self.assertEqual(runtime.last_diagnostics["termination/done"], 1.0)
        self.assertEqual(runtime.last_diagnostics["termination/by_queue"], 1.0)
        self.assertEqual(runtime.last_diagnostics["env/termination_reason"], 1.0)
        self.assertIsNone(runtime.transition_aligner.pending)

    def test_frozen_sim_time_fails_closed_before_unbounded_training(self):
        runtime, pclient = training_runtime(max_stale_sim_time_ticks=2)
        runtime._save_runtime_checkpoint = lambda kind: None
        pclient.SimTime = 10.0
        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)
        self.assertTrue(runtime.training_failed)
        self.assertEqual(pclient.sent_is_end, [0, 0, 1])
        with self.assertRaises(ContextualTrainingFailure):
            runtime.AlgorithmAfter(pclient)

    def test_wandb_schema_matches_export_and_runtime_config(self):
        self.assertEqual(set(WANDB_METRIC_KEYS), set(EXPORT_COLUMNS) - {"_step"})
        twin_metrics = {
            "critic/q_abs_diff_mean",
            "critic/q_abs_diff_max",
            "critic/parameter_l2_distance",
            "critic/parameter_max_abs_diff",
            "critic/q1_loss",
            "critic/q2_loss",
            "critic/q1_grad_norm",
            "critic/q2_grad_norm",
            "learner/actor_loss_last",
            "learner/actor_grad_norm_last",
            "learner/actor_last_update_step",
            "learner/actor_updates_total",
            "learner/actor_updated_this_step",
        }
        self.assertTrue(twin_metrics <= set(WANDB_METRIC_KEYS))
        self.assertNotIn("learner/actor_loss", WANDB_METRIC_KEYS)
        self.assertNotIn("grad/actor_norm", WANDB_METRIC_KEYS)
        config = ContextualRuntimeConfig(
            mode="training", action_enabled=True, action_scale=0.05,
            wandb_enabled=True, device="cpu",
        )
        captured = {}

        class Run:
            def __init__(self):
                self.summary = {}
                self.finish_count = 0

            def log(self, payload, step):
                captured["payload"] = payload
                captured["step"] = step

            def finish(self):
                self.finish_count += 1

        fake_wandb = SimpleNamespace(
            init=lambda **kwargs: captured.update(kwargs) or Run()
        )
        with patch.dict("sys.modules", {"wandb": fake_wandb}):
            logger = ContextualWandbLogger(config)
        meta = runtime_exp_meta(config)
        self.assertEqual(captured["config"]["EXP_META"], meta)
        self.assertEqual(captured["notes"], meta["description"])
        self.assertEqual(
            meta["algorithm_version"],
            "contextual_directional_td7_independent_twin_critic_v1",
        )
        self.assertEqual(meta["critic_initialization"], "independent")
        logger.log({"env/step": 3.0, "not/exported": 9.0}, 3)
        self.assertEqual(captured["payload"], {"env/step": 3.0})
        logger.finish_failed("boom", 3)
        logger.finish_failed("again", 4)
        self.assertEqual(logger.run.finish_count, 1)
        self.assertEqual(logger.run.summary["run/status"], "failed")
        self.assertEqual(logger.run.summary["run/failure_reason"], "boom")
        self.assertEqual(logger.run.summary["run/failure_step"], 3)


if __name__ == "__main__":
    unittest.main()
