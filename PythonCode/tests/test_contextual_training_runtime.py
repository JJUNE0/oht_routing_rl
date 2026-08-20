import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from oht_routing.runtime.client import ClientAlgorithm, ContextualRuntimeConfig
from oht_routing.runtime.client import ContextualTrainingFailure
from oht_routing.mdp.action import EXP_RESIDUAL, REGION_B_RL
from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualTD7Learner,
)
from oht_routing.algorithms.rl.contextual_td7.learner_types import (
    ContextualLearnerUpdate,
)
from oht_routing.mdp.observation import (
    ContextualObservationBatch,
    RunningFeatureNormalizer,
)
from test_contextual_observation import (
    CONTROLLED_COUNT,
    PHYSICAL_COUNT,
    make_topology,
)
from test_contextual_runtime import make_runtime_pclient
from oht_routing.utils.wandb_logging import (
    WANDB_METRIC_KEYS,
    ContextualWandbLogger,
    runtime_exp_meta,
)
from oht_routing.utils.export_wandb_run import EXPORT_COLUMNS
from oht_routing.version import CONTEXTUAL_VERSION


class TrainingObservationBuilder:
    def __init__(self, topology, freeze_steps):
        self.topology = topology
        self.freeze_steps = freeze_steps
        self.calls = 0
        self.env_steps = 0
        self.load_calls = []
        self.save_calls = []
        self.local_normalizer = RunningFeatureNormalizer(8)
        self.global_normalizer = RunningFeatureNormalizer(6)
        self._incoming_relation = np.zeros(
            (CONTROLLED_COUNT, 10, 2), np.float32
        )
        self._outgoing_relation = np.ones(
            (CONTROLLED_COUNT, 10, 2), np.float32
        )

    def build(
        self, pclient, *, parameter_dw, parameter_c,
        previous_applied_action=None,
    ):
        step = self.calls
        physical = np.full((PHYSICAL_COUNT, 8), step, np.float32)
        global_raw = np.full(6, step, np.float32)
        if not self.local_normalizer.frozen:
            self.local_normalizer.update(physical)
            self.global_normalizer.update(global_raw)
        self.calls += 1
        self.env_steps += 1
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
            previous_applied_action=np.ascontiguousarray(
                np.zeros((CONTROLLED_COUNT, 1), np.float32)
                if previous_applied_action is None
                else np.asarray(previous_applied_action, np.float32).reshape(-1, 1)
            ),
            controlled_rail_ids=self.topology.controlled_rail_ids.copy(),
            topology_hash=self.topology.topology_hash,
            mapping_hash=self.topology.mapping_hash,
            physical_local_raw=physical,
            global_raw=global_raw,
        )

    def load_normalizers(self, path, *, require_frozen=False):
        self.load_calls.append((str(path), bool(require_frozen)))
        if self.local_normalizer.count == 0:
            self.local_normalizer.update(np.zeros((1, 8), np.float32))
            self.global_normalizer.update(np.zeros((1, 6), np.float32))
        self.local_normalizer.freeze()
        self.global_normalizer.freeze()
        self.env_steps = 10_000

    def save_normalizers(self, path, *, require_frozen=False):
        self.save_calls.append((str(path), bool(require_frozen)))
        return Path(path)


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
        "terminate_on_warmup_complete": False,
        "episode_burnin_steps": 0,
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
    def test_runtime_observation_carries_previous_executed_action(self):
        runtime, pclient = training_runtime(
            warmup_steps=0,
            normalizer_freeze_steps=1,
            exploration_noise_std=0.0,
            exploration_noise_final_std=0.0,
        )
        policy = np.full(CONTROLLED_COUNT, 0.5, np.float32)
        runtime._actor_inference = lambda *args, **kwargs: (
            policy.copy(),
            {
                "runtime/tensor_conversion_ms": 0.0,
                "runtime/host_to_device_ms": 0.0,
                "runtime/encoder_actor_ms": 0.0,
                "runtime/device_to_host_ms": 0.0,
            },
        )

        runtime.Algorithm(pclient)
        np.testing.assert_array_equal(
            runtime.last_observation.previous_applied_action,
            np.zeros((CONTROLLED_COUNT, 1), np.float32),
        )
        first_applied = runtime.last_applied_action.copy()
        self.assertGreater(float(np.abs(first_applied).max()), 0.0)

        runtime.Algorithm(pclient)
        np.testing.assert_allclose(
            runtime.last_observation.previous_applied_action[:, 0],
            first_applied,
        )

        runtime.Reset(pclient)
        self.assertIsNone(runtime.last_applied_action)

    def test_default_checkpoint_root_isolated_by_runtime_variant(self):
        sale_lap_region = ClientAlgorithm(ContextualRuntimeConfig())
        uniform_region = ClientAlgorithm(
            ContextualRuntimeConfig(sale_enabled=False, lap_enabled=False)
        )
        sale_lap_residual = ClientAlgorithm(
            ContextualRuntimeConfig(action_mode=EXP_RESIDUAL)
        )
        sale_uniform_random_rail = ClientAlgorithm(
            ContextualRuntimeConfig(
                sale_enabled=True,
                lap_enabled=False,
                replay_sampling_mode="random_rail",
            )
        )

        variants = {
            sale_lap_region.runtime_variant,
            uniform_region.runtime_variant,
            sale_lap_residual.runtime_variant,
            sale_uniform_random_rail.runtime_variant,
        }
        roots = {
            sale_lap_region.checkpoint_root,
            uniform_region.checkpoint_root,
            sale_lap_residual.checkpoint_root,
            sale_uniform_random_rail.checkpoint_root,
        }
        self.assertEqual(len(variants), 4)
        self.assertEqual(len(roots), 4)
        for runtime in (
            sale_lap_region,
            uniform_region,
            sale_lap_residual,
            sale_uniform_random_rail,
        ):
            self.assertIn(runtime.algorithm_variant, runtime.runtime_variant)
            self.assertIn(runtime.config.action_mode, runtime.runtime_variant)
            self.assertIn(CONTEXTUAL_VERSION, runtime.runtime_variant)
            self.assertLess(len(runtime.checkpoint_root.name), 80)
            self.assertEqual(
                runtime.checkpoint_root.name, runtime.checkpoint_variant
            )

        scheduled_n = ClientAlgorithm(ContextualRuntimeConfig(
            curriculum_scale_start=0.05,
            curriculum_scale_end=1.0,
        ))
        fixed_scale_n = ClientAlgorithm(ContextualRuntimeConfig(
            curriculum_scale_start=1.0,
            curriculum_scale_end=1.0,
        ))
        self.assertNotEqual(
            scheduled_n.checkpoint_root, fixed_scale_n.checkpoint_root
        )

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

    def test_wandb_logging_uses_the_configured_interval(self):
        runtime, _ = training_runtime(wandb_log_interval=10)
        captured = []
        runtime.wandb_logger.log = lambda diagnostics, step: captured.append(
            (dict(diagnostics), step)
        )
        runtime.total_steps = 7
        runtime.last_diagnostics = {"reward/total_mean": 1.0}

        runtime.log_wandb_tick()
        self.assertEqual(captured, [])

        runtime.total_steps = 10
        runtime.log_wandb_tick()

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0]["reward/total_mean"], 1.0)
        self.assertEqual(captured[0][1], 10)

    def test_rich_compatibility_diagnostics_stay_out_of_compact_wandb(self):
        runtime, pclient = training_runtime(
            warmup_steps=0, normalizer_freeze_steps=1
        )
        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)

        expected = {
            "reward/step_reward", "reward/global_norm",
            "reward/local_norm", "reward/rail_tat",
            "reward/rail_tat_event_count", "reward/rail_tat_sum",
            "reward/rail_tat_mean", "reward/rail_tat_vector_mean",
            "reward/rail_tat_event_mean", "reward/rail_tat_max",
            "reward/alpha", "reward/local_scaled_std",
            "reward/global_raw", "reward/global_component",
            "reward/local_raw_mean", "reward/local_scaled_mean",
            "reward/rail_tat_raw_mean",
            "reward/rail_tat_weighted_preclip_mean",
            "reward/rail_tat_penalty_mean",
            "reward/total_tat_level", "reward/global/total_tat",
            "reward/global/tat_reference",
            "reward/global/tat_signal_available",
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
        compact_retained = {
            "curriculum/action_scale",
            "oht/idle_count",
            "reward/rail_tat_mean",
            "reward/global/component",
            "reward/local/normalized_mean",
        }
        compact_removed = {
            "reward/step_reward",
            "reward/global_norm",
            "reward/local_norm",
            "reward/rail_tat_vector_mean",
            "reward/rail_tat_event_mean",
            "reward/rail_tat_penalty_mean",
            "reward/global_raw",
            "reward/local_raw_mean",
            "global/tat",
            "global/op_rate",
            "oht/move_to_load",
        }
        self.assertTrue(compact_retained <= set(WANDB_METRIC_KEYS))
        self.assertTrue(compact_retained <= set(EXPORT_COLUMNS))
        self.assertTrue(compact_removed.isdisjoint(WANDB_METRIC_KEYS))
        self.assertTrue(compact_removed.isdisjoint(EXPORT_COLUMNS))
        self.assertEqual(
            runtime.last_diagnostics["reward/step_reward"],
            runtime.last_diagnostics["reward/total_mean"],
        )
        self.assertEqual(
            runtime.last_diagnostics["reward/global_norm"],
            runtime.last_diagnostics["reward/global/normalized"],
        )
        self.assertNotIn(
            "reward/region_internal_std_mean", runtime.last_diagnostics
        )

    def test_training_requires_explicit_action_enabled(self):
        with self.assertRaises(ValueError):
            ContextualRuntimeConfig(
                mode="training", action_enabled=False, action_scale=0.05
            )
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            ContextualRuntimeConfig(episode_burnin_steps=1.5)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ContextualRuntimeConfig(episode_burnin_steps=-1)
        with self.assertRaisesRegex(ValueError, "locked full reward profile"):
            ContextualRuntimeConfig(early_stop_tat_threshold=0.0)
        with self.assertRaisesRegex(ValueError, "locked full reward profile"):
            ContextualRuntimeConfig(tat_termination_grace_steps=True)
        with self.assertRaisesRegex(ValueError, "locked full reward profile"):
            ContextualRuntimeConfig(tat_above_threshold_patience=0)
        with self.assertRaisesRegex(ValueError, "locked full reward profile"):
            ContextualRuntimeConfig(terminal_tat_penalty=1.0)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ContextualRuntimeConfig(
                load_state_normalizer_path="state.npz",
                resume_checkpoint_path="checkpoint.pt",
            )
        with self.assertRaisesRegex(ValueError, "requires"):
            ContextualRuntimeConfig(state_normalizer_warmup_bypass=True)
        with self.assertRaisesRegex(ValueError, "requires checkpoint resume"):
            ContextualRuntimeConfig(
                resume_inference_until_replay_full=True
            )
        with self.assertRaisesRegex(ValueError, "requires checkpoint resume"):
            ContextualRuntimeConfig(
                resume_deterministic_first_episode=True
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

    def test_warmup_completion_ends_collection_episode_before_actor(self):
        runtime, pclient = training_runtime(
            warmup_steps=2,
            normalizer_freeze_steps=2,
            terminate_on_warmup_complete=True,
            exploration_noise_std=0.0,
            exploration_noise_final_std=0.0,
        )
        actor_steps = []
        runtime._actor_inference = lambda *args, **kwargs: (
            actor_steps.append(runtime.total_steps)
            or np.zeros(CONTROLLED_COUNT, dtype=np.float32),
            {
                "runtime/tensor_conversion_ms": 0.0,
                "runtime/host_to_device_ms": 0.0,
                "runtime/encoder_actor_ms": 0.0,
                "runtime/device_to_host_ms": 0.0,
            },
        )
        checkpoint_kinds = []
        runtime._save_runtime_checkpoint = checkpoint_kinds.append

        runtime.Reset(pclient)
        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)
        self.assertEqual(pclient.sent_is_end, [0, 0])
        self.assertEqual(actor_steps, [])
        self.assertEqual(runtime.reward_builder.reward_steps, 1)

        runtime.Algorithm(pclient)
        self.assertEqual(pclient.sent_is_end, [0, 0, 1])
        self.assertEqual(actor_steps, [])
        self.assertEqual(runtime.reward_builder.reward_steps, 2)
        self.assertEqual(
            runtime.last_diagnostics["termination/by_warmup"], 1.0
        )
        self.assertEqual(
            runtime.last_diagnostics["env/termination_reason"], 4.0
        )
        self.assertIsNone(runtime.transition_aligner.pending)
        self.assertTrue(runtime.warmup_episode_boundary_sent)
        self.assertEqual(checkpoint_kinds, ["latest"])

        runtime.Reset(pclient)
        runtime.Algorithm(pclient)
        self.assertEqual(pclient.sent_is_end, [0, 0, 1, 0])
        self.assertEqual(actor_steps, [3])

    def test_episode_burnin_boundary_excludes_replay_and_stale_action(self):
        runtime, pclient = training_runtime(
            episode_burnin_steps=3,
            warmup_steps=0,
            normalizer_freeze_steps=1,
            exploration_noise_std=0.0,
        )
        runtime.checkpoint_loaded = True
        timing = {
            "runtime/tensor_conversion_ms": 0.0,
            "runtime/host_to_device_ms": 0.0,
            "runtime/encoder_actor_ms": 0.0,
            "runtime/device_to_host_ms": 0.0,
        }
        actions = {
            0: 0.1, 1: 0.2, 2: 0.3, 3: 0.7, 4: 0.8,
        }
        runtime._actor_inference = lambda *args, **kwargs: (
            np.full(
                CONTROLLED_COUNT,
                actions[runtime.episode_steps],
                dtype=np.float32,
            ),
            dict(timing),
        )

        for step in range(3):
            runtime.Algorithm(pclient)
            self.assertEqual(runtime.last_diagnostics["episode/step"], step)
            self.assertEqual(runtime.last_diagnostics["burnin/active"], 1.0)
            self.assertEqual(runtime.last_diagnostics["burnin/action_source"], 1.0)
            self.assertEqual(runtime.last_diagnostics["action/exploration_noise_std"], 0.0)
            self.assertEqual(runtime.replay_buffer.push_count, 0)
            self.assertIsNone(runtime.transition_aligner.pending)

        reward_steps_before = runtime.reward_builder.reward_steps
        last_burnin_applied = (
            runtime.transition_aligner.previous_applied_action.copy()
        )
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.last_diagnostics["burnin/active"], 0.0)
        self.assertEqual(runtime.replay_buffer.push_count, 0)
        self.assertFalse(hasattr(runtime.reward_builder, "_tat_ema"))
        self.assertEqual(
            runtime.reward_builder.reward_steps, reward_steps_before + 1
        )
        np.testing.assert_array_equal(
            runtime.transition_aligner.previous_applied_action,
            last_burnin_applied,
        )
        np.testing.assert_array_equal(
            runtime.transition_aligner.pending.controlled_action,
            np.full((CONTROLLED_COUNT, 1), 0.7, dtype=np.float32),
        )
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.replay_buffer.push_count, 1)
        np.testing.assert_array_equal(
            runtime.replay_buffer._policy_action[0],
            np.full(CONTROLLED_COUNT, 0.7, dtype=np.float32),
        )

    def test_episode_burnin_untrained_fallback_skips_actor(self):
        runtime, pclient = training_runtime(
            episode_burnin_steps=3,
            warmup_steps=0,
            normalizer_freeze_steps=1,
        )
        runtime._actor_inference = lambda *args, **kwargs: (
            (_ for _ in ()).throw(AssertionError("untrained actor used"))
        )
        baseline = np.asarray([
            pclient.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
            for rail_id in runtime.topology.all_rail_ids
        ])
        for _ in range(3):
            result = runtime.Algorithm(pclient)
            np.testing.assert_array_equal(result.final_cost, baseline)
            self.assertEqual(runtime.last_diagnostics["burnin/action_source"], 2.0)
            self.assertEqual(runtime.learner.learner_update_count, 0)
            self.assertEqual(runtime.replay_buffer.push_count, 0)

    def test_episode_reset_preserves_replay_and_pauses_existing_learner(self):
        runtime, pclient = training_runtime(
            episode_burnin_steps=3,
            warmup_steps=0,
            normalizer_freeze_steps=1,
            minimum_replay_env_steps=1,
            minimum_action_enabled_env_steps=1,
        )
        for _ in range(6):
            runtime.Algorithm(pclient)
        replay_size = runtime.replay_buffer.size_env_steps
        replay_pushes = runtime.replay_buffer.push_count
        updates = runtime.learner.learner_update_count
        actor_updates = runtime.learner.actor_update_count
        target_updates = runtime.learner.target_update_count
        self.assertGreater(replay_size, 0)
        self.assertGreater(updates, 0)

        runtime.Reset(pclient)
        self.assertEqual(runtime.replay_buffer.size_env_steps, replay_size)
        sample = runtime.replay_buffer.sample(1, device="cpu")
        self.assertTrue(torch.isfinite(sample.reward).all())
        for _ in range(3):
            runtime.Algorithm(pclient)
        self.assertEqual(runtime.replay_buffer.size_env_steps, replay_size)
        self.assertEqual(runtime.replay_buffer.push_count, replay_pushes)
        self.assertEqual(runtime.learner.learner_update_count, updates)
        self.assertEqual(runtime.learner.actor_update_count, actor_updates)
        self.assertEqual(runtime.learner.target_update_count, target_updates)

    def test_zero_episode_burnin_is_exactly_disabled(self):
        runtime, pclient = training_runtime(
            episode_burnin_steps=0,
            warmup_steps=0,
            normalizer_freeze_steps=1,
        )
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.last_diagnostics["burnin/active"], 0.0)
        self.assertEqual(runtime.last_diagnostics["burnin/action_source"], 0.0)
        self.assertIsNotNone(runtime.transition_aligner.pending)

    def test_global_warmup_skips_actor_until_action_boundary(self):
        runtime, pclient = training_runtime(
            episode_burnin_steps=0,
            warmup_steps=2,
            normalizer_freeze_steps=1,
        )
        actor_steps = []
        timing = {
            "runtime/tensor_conversion_ms": 0.0,
            "runtime/host_to_device_ms": 0.0,
            "runtime/encoder_actor_ms": 0.0,
            "runtime/device_to_host_ms": 0.0,
        }

        def actor_inference(*args, **kwargs):
            actor_steps.append(runtime.total_steps)
            return (
                np.full(CONTROLLED_COUNT, 0.5, dtype=np.float32),
                dict(timing),
            )

        runtime._actor_inference = actor_inference

        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)
        self.assertEqual(actor_steps, [])
        np.testing.assert_array_equal(
            runtime.transition_aligner.pending.controlled_action,
            np.zeros((CONTROLLED_COUNT, 1), dtype=np.float32),
        )

        runtime.Algorithm(pclient)
        self.assertEqual(actor_steps, [2])
        np.testing.assert_array_equal(
            runtime.transition_aligner.pending.controlled_action,
            np.full((CONTROLLED_COUNT, 1), 0.5, dtype=np.float32),
        )

    def test_loaded_state_normalizer_bypasses_action_warmup_only(self):
        runtime, pclient = training_runtime(
            warmup_steps=10_000,
            normalizer_freeze_steps=10_000,
            terminate_on_warmup_complete=True,
            load_state_normalizer_path="state_normalizer.npz",
            exploration_noise_std=0.0,
            exploration_noise_final_std=0.0,
        )
        actor_steps = []
        runtime._actor_inference = lambda *args, **kwargs: (
            actor_steps.append(runtime.total_steps)
            or np.full(CONTROLLED_COUNT, 0.25, dtype=np.float32),
            {
                "runtime/tensor_conversion_ms": 0.0,
                "runtime/host_to_device_ms": 0.0,
                "runtime/encoder_actor_ms": 0.0,
                "runtime/device_to_host_ms": 0.0,
            },
        )

        self.assertTrue(runtime.config.state_normalizer_warmup_bypass)
        self.assertEqual(runtime.config.warmup_steps, 10_000)
        self.assertEqual(runtime.config.effective_warmup_steps, 0)
        self.assertTrue(runtime.state_normalizer_loaded)
        self.assertEqual(
            runtime.observation_builder.load_calls,
            [("state_normalizer.npz", True)],
        )

        runtime.Algorithm(pclient)
        self.assertEqual(pclient.sent_is_end, [0])
        self.assertEqual(actor_steps, [0])
        np.testing.assert_array_equal(
            runtime.transition_aligner.pending.controlled_action,
            np.full((CONTROLLED_COUNT, 1), 0.25, dtype=np.float32),
        )
        self.assertEqual(runtime.replay_buffer.size_env_steps, 0)
        self.assertEqual(runtime.learner.learner_update_count, 0)

    def test_state_normalizer_is_saved_once_at_freeze_boundary(self):
        runtime, pclient = training_runtime(
            warmup_steps=4,
            normalizer_freeze_steps=2,
            save_state_normalizer_path="state_normalizer.npz",
        )

        runtime.Algorithm(pclient)
        self.assertEqual(runtime.observation_builder.save_calls, [])
        runtime.Algorithm(pclient)
        self.assertEqual(
            runtime.observation_builder.save_calls,
            [("state_normalizer.npz", True)],
        )
        runtime.Algorithm(pclient)
        self.assertEqual(len(runtime.observation_builder.save_calls), 1)
        self.assertTrue(runtime.state_normalizer_saved)

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

    def test_exploration_noise_anneals_to_final_std_by_100k(self):
        runtime, _ = training_runtime(
            warmup_steps=10_000,
            exploration_noise_std=0.10,
            exploration_noise_final_std=0.02,
            exploration_noise_anneal_steps=100_000,
        )
        expected = (
            (0, 0.10),
            (9_999, 0.10),
            (10_000, 0.10),
            (60_000, 0.06),
            (110_000, 0.02),
            (150_000, 0.02),
        )
        for step, noise_std in expected:
            runtime.total_steps = step
            self.assertAlmostEqual(runtime._exploration_noise_std(), noise_std)

        deterministic, _ = training_runtime(exploration_noise_std=0.0)
        deterministic.total_steps = 100_000
        self.assertEqual(deterministic._exploration_noise_std(), 0.0)

    def test_policy_and_exploratory_temporal_deltas_are_separate(self):
        runtime, pclient = training_runtime(
            warmup_steps=0,
            exploration_noise_std=0.0,
        )
        first = np.zeros(CONTROLLED_COUNT, dtype=np.float32)
        second = np.linspace(
            -0.5, 0.5, CONTROLLED_COUNT, dtype=np.float32
        )
        actions = iter((first, second))
        timing = {
            "runtime/tensor_conversion_ms": 0.0,
            "runtime/host_to_device_ms": 0.0,
            "runtime/encoder_actor_ms": 0.0,
            "runtime/device_to_host_ms": 0.0,
        }
        runtime._actor_inference = lambda *args, **kwargs: (
            next(actions).copy(), dict(timing)
        )

        runtime.Algorithm(pclient)
        self.assertEqual(
            runtime.last_diagnostics["action/policy_temporal_delta_std"], 0.0
        )
        runtime.Algorithm(pclient)
        diagnostics = runtime.last_diagnostics
        self.assertAlmostEqual(
            diagnostics["action/policy_temporal_delta_mean"],
            float(second.mean()),
        )
        self.assertAlmostEqual(
            diagnostics["action/policy_temporal_delta_std"],
            float(second.std()),
        )
        self.assertAlmostEqual(
            diagnostics["action/exploratory_temporal_delta_mean"],
            float(second.mean()),
        )
        self.assertAlmostEqual(
            diagnostics["action/exploratory_temporal_delta_std"],
            float(second.std()),
        )
        self.assertEqual(diagnostics["action/exploration_noise_std"], 0.0)

        runtime.Reset(pclient)
        self.assertIsNone(runtime.last_policy_action)
        self.assertIsNone(runtime.last_exploratory_action)

    def test_same_snapshot_action_variance_is_decomposed_before_scaling(self):
        runtime, pclient = training_runtime(
            warmup_steps=0,
            exploration_noise_std=0.1,
            exploration_noise_clip=0.2,
        )
        policy = np.linspace(
            -0.9, 0.9, CONTROLLED_COUNT, dtype=np.float32
        )
        raw_noise = np.linspace(
            -0.2, 0.2, CONTROLLED_COUNT, dtype=np.float32
        )
        timing = {
            "runtime/tensor_conversion_ms": 0.0,
            "runtime/host_to_device_ms": 0.0,
            "runtime/encoder_actor_ms": 0.0,
            "runtime/device_to_host_ms": 0.0,
        }
        runtime._actor_inference = lambda *args, **kwargs: (
            policy.copy(), dict(timing)
        )
        runtime.exploration_rng = SimpleNamespace(
            normal=lambda *args, **kwargs: raw_noise.copy()
        )

        runtime.Algorithm(pclient)
        diagnostics = runtime.last_diagnostics
        preclip = policy + raw_noise
        postclip = np.clip(preclip, -1.0, 1.0)
        effective_noise = postclip - policy
        self.assertAlmostEqual(
            diagnostics["action/cross_rail_policy_std"], policy.std()
        )
        self.assertAlmostEqual(
            diagnostics["action/cross_rail_applied_std"], postclip.std()
        )
        self.assertAlmostEqual(
            diagnostics["action/cross_rail_noise_std"], raw_noise.std()
        )
        self.assertAlmostEqual(
            diagnostics["action/cross_rail_noise_residual_std"],
            effective_noise.std(),
        )
        self.assertAlmostEqual(
            diagnostics["action/clipped_fraction"],
            np.mean(np.abs(preclip) > 1.0),
        )
        self.assertGreater(
            diagnostics["action/noise_suppressed_by_clip_mean"], 0.0
        )
        self.assertAlmostEqual(
            diagnostics["action/applied_std"], 0.05 * postclip.std(),
            places=7,
        )

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
            terminate_on_warmup_complete=False,
            episode_burnin_steps=0,
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

    def test_actor_inference_restores_checkpoint_without_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = training_runtime(
                checkpoint_root=directory,
                warmup_steps=4,
                normalizer_freeze_steps=1,
            )
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
            runtime.observation_builder.load_normalizers(
                "synthetic-normalizer.npz", require_frozen=True
            )
            runtime.total_steps = 400_000
            runtime.episode_id = 8
            runtime.transition_aligner.episode_id = 8
            checkpoint = runtime._save_runtime_checkpoint("latest")
            expected_actor = {
                key: value.detach().clone()
                for key, value in runtime.actor.state_dict().items()
            }
            expected_updates = runtime.learner.learner_update_count

            inference = ClientAlgorithm(ContextualRuntimeConfig(
                mode="actor_inference",
                action_enabled=True,
                action_scale=0.05,
                warmup_steps=4,
                terminate_on_warmup_complete=False,
                normalizer_freeze_steps=1,
                state_normalizer_warmup_bypass=True,
                device="cpu",
                seed=12,
                replay_capacity_env_steps=64,
                batch_size=16,
                minimum_replay_env_steps=2,
                minimum_action_enabled_env_steps=2,
                checkpoint_root=directory,
                resume_checkpoint_path=str(checkpoint),
                sale_enabled=False,
                lap_enabled=False,
            ))
            inference.topology = make_topology()
            inference.observation_builder = TrainingObservationBuilder(
                inference.topology, freeze_steps=1
            )
            pclient = make_runtime_pclient()
            inference._ensure_initialized(pclient)

            self.assertTrue(inference.checkpoint_loaded)
            self.assertEqual(inference.total_steps, 400_000)
            self.assertEqual(inference.episode_id, 8)
            self.assertEqual(inference._action_scale(), 1.0)
            self.assertTrue(inference._state_normalizers_ready_for_bypass())
            self.assertIsNone(inference.replay_buffer)
            self.assertIsNone(inference.transition_aligner.callback)
            for key, expected in expected_actor.items():
                torch.testing.assert_close(
                    inference.actor.state_dict()[key], expected
                )

            inference.Algorithm(pclient)
            inference.Algorithm(pclient)
            self.assertEqual(
                inference.learner.learner_update_count, expected_updates
            )
            self.assertIsNone(inference.replay_buffer)
            self.assertEqual(
                inference.last_diagnostics["action/cross_rail_noise_std"],
                0.0,
            )

    def test_resume_can_collect_full_replay_before_any_learner_update(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, pclient = training_runtime(
                checkpoint_root=directory,
                replay_capacity_env_steps=4,
                latest_checkpoint_interval=2,
                periodic_checkpoint_interval=10_000,
                warmup_steps=0,
                normalizer_freeze_steps=1,
            )
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

            resumed, resumed_client = training_runtime(
                checkpoint_root=directory,
                resume_checkpoint_path=str(latest),
                resume_inference_until_replay_full=True,
                replay_capacity_env_steps=4,
                latest_checkpoint_interval=10_000,
                periodic_checkpoint_interval=10_000,
                warmup_steps=0,
                normalizer_freeze_steps=1,
            )
            self.assertEqual(resumed.config.resume_refill_target_env_steps, 4)
            self.assertTrue(resumed._resume_requires_refill)
            for _ in range(5):
                resumed.Algorithm(resumed_client)
            self.assertEqual(resumed.replay_buffer.size_env_steps, 4)
            self.assertFalse(resumed._resume_requires_refill)
            self.assertEqual(resumed.learner.learner_update_count, 0)

            resumed.Algorithm(resumed_client)
            self.assertEqual(resumed.learner.learner_update_count, 1)

    def test_resume_deterministic_first_episode_then_trains_from_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, pclient = training_runtime(
                checkpoint_root=directory,
                latest_checkpoint_interval=2,
                periodic_checkpoint_interval=10_000,
                warmup_steps=0,
                normalizer_freeze_steps=1,
            )
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

            resumed, resumed_client = training_runtime(
                checkpoint_root=directory,
                resume_checkpoint_path=str(latest),
                resume_deterministic_first_episode=True,
                latest_checkpoint_interval=10_000,
                periodic_checkpoint_interval=10_000,
                warmup_steps=0,
                normalizer_freeze_steps=1,
            )
            self.assertTrue(resumed._resume_deterministic_episode_active)
            for _ in range(3):
                resumed.Algorithm(resumed_client)
            self.assertGreaterEqual(resumed.replay_buffer.size_env_steps, 2)
            self.assertEqual(resumed.learner.learner_update_count, 0)
            self.assertEqual(
                resumed.last_diagnostics["action/exploration_noise_std"],
                0.0,
            )
            self.assertEqual(
                resumed.last_diagnostics["action/cross_rail_noise_std"],
                0.0,
            )

            resumed.Reset(resumed_client)
            self.assertFalse(resumed._resume_deterministic_episode_active)
            resumed.Algorithm(resumed_client)
            self.assertEqual(resumed.learner.learner_update_count, 1)
            self.assertGreater(
                resumed.last_diagnostics["action/exploration_noise_std"],
                0.0,
            )

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

    def test_reward_n_tat_patience_adds_terminal_penalty_once(self):
        runtime, pclient = training_runtime(
            replay_capacity_env_steps=512,
            minimum_replay_env_steps=512,
            minimum_action_enabled_env_steps=512,
        )
        runtime._actor_inference = lambda observation, **kwargs: (
            np.zeros(CONTROLLED_COUNT, dtype=np.float32), {}
        )
        pclient.TotalTat = 250.0
        runtime.episode_steps = 10_000
        runtime.tat_above_threshold_count = 296
        for tick in range(1, 5):
            pclient.SimTime = float(tick)
            runtime.Algorithm(pclient)

        self.assertEqual(pclient.sent_is_end, [0, 0, 0, 1])
        self.assertEqual(runtime.last_diagnostics["termination/done"], 1.0)
        self.assertEqual(runtime.last_diagnostics["termination/by_tat"], 1.0)
        self.assertEqual(runtime.last_diagnostics["env/termination_reason"], 2.0)
        self.assertEqual(runtime.replay_buffer.push_count, 3)
        self.assertEqual(runtime.replay_buffer._done[2], 1.0)
        terminal = runtime.transition_aligner.last_completed.reward
        expected_without_terminal = (
            terminal.global_component
            + terminal.local_component
            + terminal.rail_reward_postclip
            - terminal.smooth_penalty
        )
        np.testing.assert_allclose(
            terminal.total, expected_without_terminal - 20.0
        )
        self.assertEqual(terminal.terminal_penalty, -20.0)
        self.assertEqual(
            runtime.last_diagnostics["reward/terminal_penalty"], -20.0
        )
        self.assertTrue(runtime._tat_terminal_penalty_applied)
        self.assertIsNone(runtime.transition_aligner.pending)

        runtime.Reset(pclient)
        self.assertEqual(runtime.tat_above_threshold_count, 0)
        self.assertFalse(runtime._tat_terminal_penalty_applied)

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
        expected = {
            "env/step", "env/episode", "episode/step", "env/tat",
            "env/operation_rate", "env/queued", "env/waiting",
            "env/transferring", "oht/idle_count", "termination/done",
            "termination/by_tat", "termination/by_warmup",
            "warmup/episode_boundary_sent", "env/termination_reason",
            "action/policy_mean", "action/policy_std",
            "action/cross_rail_policy_std",
            "action/policy_saturation_ratio", "action/clipped_fraction",
            "action/exploration_noise_std", "action/applied_mean",
            "action/applied_std", "curriculum/action_scale", "b_rl/mean",
            "b_rl/std", "cost/all_baseline_abs_error_max",
            "reward/total_mean", "reward/total_std",
            "reward/terminal_penalty",
            "reward/global/tat_component_raw",
            "reward/global/backlog_component_raw",
            "reward/global/raw", "reward/global/component",
            "reward/local/raw_mean",
            "reward/local/raw_std", "reward/local/normalized_mean",
            "reward/local/normalized_std", "reward/local/component_mean",
            "reward/local/component_std", "reward/rail_tat_mean",
            "local/oht_abs_mean", "local/oht_std",
            "local/pred_abs_mean", "local/pred_std",
            "local/stop_abs_mean", "local/stop_std",
            "local/idle_abs_mean", "local/idle_std",
            "local/capacity_abs_mean", "local/capacity_std",
            "reward/smooth_penalty_mean",
            "reward/contribution/tat_abs",
            "reward/contribution/backlog_abs",
            "reward/contribution/local_abs",
            "reward/budget/rail_abs", "reward/budget/smooth_abs",
            "reward/budget/tat_share", "reward/budget/backlog_share",
            "reward/budget/local_share",
            "reward/budget/rail_share", "reward/budget/smooth_share",
            "reward/budget/share_sum_error", "reward/rail/route_ratio_mean",
            "reward/rail/route_ratio_p95",
            "reward/rail/positive_cycle_ratio",
            "reward/rail/negative_cycle_ratio",
            "reward/rail/clip_assignment_ratio",
            "reward/rail/clip_removed_ratio",
            "reward/scale/rail_active_representative",
            "learner/critic_loss", "learner/actor_loss_last",
            "learner/actor_grad_norm_last", "critic/q1_mean",
            "critic/q2_mean", "critic/target_q_mean",
            "critic/q_abs_diff_mean", "critic/td_error_mean",
            "critic/td_error_max", "grad/encoder_norm",
            "grad/critic_norm", "learner/actor_updates_total",
            "learner/updates", "replay/size_env_steps",
            "replay/reward_mean", "replay/reward_std",
            "numeric/learner_finite_ratio",
            "runtime/observation_build_calls_per_tick",
            "runtime/nonfinite_count", "sale/loss",
            "sale/prediction_error_mean", "sale/online_grad_norm",
            "sale/fixed_online_distance", "lead/backlog/value",
            "lead/backlog/delta_300", "lead/idle/value",
            "lead/idle/delta_300", "lead/idle/reserve_signal",
            "lead/op/value", "lead/predicted_oht/mean",
            "lead/predicted_oht/p95", "lead/route_ratio/p95",
            "lead/flow/arrival_available",
            "lead/flow/imbalance_count_300", "lead/composite_pressure",
            "leadlag/predicted_p95_vs_future_tat_500",
            "leadlag/backlog_delta_300_vs_future_tat_500",
            "leadlag/idle_delta_300_vs_future_tat_500",
            "leadlag/flow_imbalance_vs_future_tat_500",
            "leadlag/composite_vs_future_tat_500",
            "leadlag/sample_count_500",
        }
        self.assertEqual(set(WANDB_METRIC_KEYS), expected)
        self.assertEqual(len(WANDB_METRIC_KEYS), len(expected))
        removed_prefixes = (
            "lap/", "dispatch/", "burnin/", "attention/", "boundary/",
            "protocol/", "value/",
        )
        self.assertFalse(any(
            key.startswith(removed_prefixes) for key in WANDB_METRIC_KEYS
        ))
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
        self.assertEqual(meta["version"], CONTEXTUAL_VERSION)
        self.assertEqual(meta["reward_version"], "N")
        self.assertEqual(
            meta["tat_signal"],
            "one_sided_total_tat_level",
        )
        self.assertNotIn("marginal_tat_enabled", meta)
        self.assertNotIn("command_trace_selection", meta)
        self.assertNotIn("reward_normalizer_freeze_steps", meta)
        self.assertEqual(meta["op_weight"], 4.0)
        self.assertTrue(meta["use_op"])
        self.assertEqual(meta["dispatch_mode"], "first-match")
        self.assertIn("dispatch_first_match", meta["note"])
        self.assertEqual(meta["num_stacks"], 1)
        self.assertEqual(meta["stack_interval"], 1)
        self.assertNotIn("observation_version", meta)
        self.assertNotIn("sale_version", meta)
        self.assertNotIn("lap_version", meta)
        self.assertEqual(
            meta["previous_action_input"],
            "actor_and_critic_previous_applied_action_separate_from_encoder",
        )
        self.assertNotIn("critic_initialization", meta)
        experiment_meta = runtime_exp_meta(ContextualRuntimeConfig(
            curriculum_scale_start=1.0,
            curriculum_scale_end=1.0,
            replay_capacity_env_steps=100_000,
            batch_size=1_024,
            warmup_steps=10_000,
            seed=0,
        ))
        self.assertEqual(experiment_meta["reward_version"], "N")
        self.assertEqual(experiment_meta["action_scale"], 1.0)
        self.assertFalse(experiment_meta["curriculum_enabled"])
        self.assertEqual(
            experiment_meta["tat_termination_policy"], "reward_profile"
        )
        self.assertEqual(experiment_meta["tat_termination_start_episode"], 1)
        self.assertEqual(experiment_meta["early_stop_tat_threshold"], 200.0)
        self.assertEqual(experiment_meta["replay_capacity_env_steps"], 100_000)
        self.assertEqual(experiment_meta["batch_size"], 1_024)
        self.assertEqual(experiment_meta["configured_warmup_steps"], 10_000)
        self.assertEqual(experiment_meta["effective_warmup_steps"], 10_000)
        reused_meta = runtime_exp_meta(ContextualRuntimeConfig(
            load_state_normalizer_path="state_normalizer.npz",
        ))
        self.assertEqual(reused_meta["configured_warmup_steps"], 10_000)
        self.assertEqual(reused_meta["effective_warmup_steps"], 0)
        self.assertTrue(reused_meta["state_normalizer_load_requested"])
        self.assertTrue(reused_meta["state_normalizer_warmup_bypass"])
        self.assertEqual(
            reused_meta["tat_termination_configured_start_episode"], 1
        )
        self.assertEqual(reused_meta["tat_termination_start_episode"], 1)
        self.assertIn("normreuse1", reused_meta["note"])
        full_refill_meta = runtime_exp_meta(ContextualRuntimeConfig(
            resume_checkpoint_path="checkpoint.pt",
            resume_inference_until_replay_full=True,
            replay_capacity_env_steps=100_000,
            batch_size=2_048,
        ))
        self.assertTrue(
            full_refill_meta["resume_inference_until_replay_full"]
        )
        self.assertEqual(
            full_refill_meta["resume_refill_target_env_steps"], 100_000
        )
        self.assertEqual(full_refill_meta["batch_size"], 2_048)
        self.assertIn("fullrefill1", full_refill_meta["note"])
        deterministic_meta = runtime_exp_meta(ContextualRuntimeConfig(
            resume_checkpoint_path="checkpoint.pt",
            resume_deterministic_first_episode=True,
            replay_capacity_env_steps=100_000,
            batch_size=2_048,
        ))
        self.assertTrue(
            deterministic_meta["resume_deterministic_first_episode"]
        )
        self.assertIn("detfirst1", deterministic_meta["note"])
        logger.log({
            "env/step": 3.0,
            "lap/enabled": 1.0,
            "dispatch/cost_mode_active": 1.0,
            "attention/entropy": 9.0,
            "not/exported": 9.0,
        }, 3)
        self.assertEqual(captured["payload"], {"env/step": 3.0})
        logger.finish_failed("boom", 3)
        logger.finish_failed("again", 4)
        self.assertEqual(logger.run.finish_count, 1)
        self.assertEqual(logger.run.summary["run/status"], "failed")
        self.assertEqual(logger.run.summary["run/failure_reason"], "boom")
        self.assertEqual(logger.run.summary["run/failure_step"], 3)


if __name__ == "__main__":
    unittest.main()
