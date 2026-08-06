import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ClientAlgorithm_contextual import ClientAlgorithm, ContextualRuntimeConfig
from cocel_rl.algorithms.contextual_sac import (
    ALGORITHM_VERSION,
    CHECKPOINT_VERSION,
    ContextualSACLearner,
    ContextualSACLearnerConfig,
    load_contextual_sac_checkpoint,
    save_contextual_sac_checkpoint,
)
from cocel_rl.algorithms.contextual_td7 import REPLAY_SAMPLING_RAIL
from contextual_wandb import ContextualWandbLogger, WANDB_METRIC_KEYS, runtime_exp_meta
from test_contextual_learner import SMALL_NETWORK, changed, make_replay, parameters
from test_contextual_observation import make_topology
from test_contextual_runtime import make_runtime_pclient
from test_contextual_training_runtime import TrainingObservationBuilder


def sac_config():
    return ContextualSACLearnerConfig(
        batch_size=16,
        action_scale=0.05,
        minimum_replay_env_steps=1,
        minimum_action_enabled_env_steps=1,
        require_normalizer_frozen=False,
    )


class StateBox:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class ContextualSACTests(unittest.TestCase):
    def test_gaussian_policy_and_full_update_are_finite(self):
        learner = ContextualSACLearner(
            make_replay(), network_config=SMALL_NETWORK,
            config=sac_config(), seed=11,
        )
        batch = learner.replay.sample(16)
        with torch.no_grad():
            state = learner.encoder(
                batch.center_local,
                batch.incoming_local,
                batch.outgoing_local,
                batch.incoming_relation,
                batch.outgoing_relation,
                batch.global_state,
            ).state
            sample = learner.actor.sample(state)
        self.assertEqual(sample.action.shape, (16, 1))
        self.assertEqual(sample.log_prob.shape, (16, 1))
        self.assertTrue(bool((sample.action.abs() <= 1.0).all()))
        self.assertTrue(torch.isfinite(sample.log_prob).all())

        actor_before = parameters(learner.actor)
        critic_before = parameters(learner.critic)
        encoder_before = parameters(learner.encoder)
        target_before = parameters(learner.target_critic)
        alpha_before = float(learner.alpha.detach())
        result = learner.update(batch)
        self.assertTrue(result.actor_updated)
        self.assertTrue(result.target_updated)
        self.assertTrue(changed(actor_before, learner.actor))
        self.assertTrue(changed(critic_before, learner.critic))
        self.assertTrue(changed(encoder_before, learner.encoder))
        self.assertTrue(changed(target_before, learner.target_critic))
        self.assertNotEqual(alpha_before, float(learner.alpha.detach()))
        self.assertTrue(all(
            math.isfinite(value) for value in result.diagnostics.values()
        ))
        self.assertEqual(result.diagnostics["sale/enabled"], 0.0)
        self.assertEqual(result.diagnostics["lap/enabled"], 0.0)

    def test_checkpoint_round_trip_restores_entropy_and_models(self):
        replay = make_replay()
        learner = ContextualSACLearner(
            replay, network_config=SMALL_NETWORK,
            config=sac_config(), seed=5,
        )
        learner.update()
        replay.observation_builder.local_normalizer = StateBox(1)
        replay.observation_builder.global_normalizer = StateBox(2)
        reward_builder = SimpleNamespace(
            local_normalizer=StateBox(3), global_normalizer=StateBox(4)
        )
        metadata = {"algorithm_version": "runtime-sac", "runtime_env_step": 9}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_contextual_sac_checkpoint(
                path,
                learner,
                observation_builder=replay.observation_builder,
                reward_builder=reward_builder,
                runtime_metadata=metadata,
                runtime_config={"algorithm": "sac"},
            )
            restored_replay = make_replay()
            restored_replay.observation_builder.local_normalizer = StateBox(0)
            restored_replay.observation_builder.global_normalizer = StateBox(0)
            restored_reward = SimpleNamespace(
                local_normalizer=StateBox(0), global_normalizer=StateBox(0)
            )
            restored = ContextualSACLearner(
                restored_replay, network_config=SMALL_NETWORK,
                config=sac_config(), seed=99,
            )
            loaded = load_contextual_sac_checkpoint(
                path,
                restored,
                observation_builder=restored_replay.observation_builder,
                reward_builder=restored_reward,
                expected_runtime_metadata={"algorithm_version": "runtime-sac"},
            )
        self.assertEqual(loaded["runtime_env_step"], 9)
        self.assertEqual(restored.learner_update_count, 1)
        torch.testing.assert_close(restored.log_alpha, learner.log_alpha)
        for source, target in zip(learner.actor.parameters(), restored.actor.parameters()):
            torch.testing.assert_close(source, target)
        self.assertEqual(restored_reward.global_normalizer.value, 4)

    def test_runtime_config_and_wandb_metadata_are_sac_specific(self):
        with self.assertRaises(ValueError):
            ContextualRuntimeConfig(algorithm="sac")
        config = ContextualRuntimeConfig(
            algorithm="sac", sale_enabled=False, lap_enabled=False,
            critic_loss_mode="mse", wandb_enabled=True, device="cpu",
        )
        captured = {}

        class Run:
            summary = {}

            def log(self, payload, step):
                captured["payload"] = payload

            def finish(self):
                pass

        fake_wandb = SimpleNamespace(
            init=lambda **kwargs: captured.update(kwargs) or Run()
        )
        with patch.dict("sys.modules", {"wandb": fake_wandb}):
            ContextualWandbLogger(config)
        meta = runtime_exp_meta(config)
        self.assertEqual(meta["algorithm_version"], ALGORITHM_VERSION)
        self.assertEqual(meta["checkpoint_version"], CHECKPOINT_VERSION)
        self.assertEqual(meta["algorithm"], "sac")
        self.assertEqual(meta["exploration_noise_std"], 0.0)
        self.assertEqual(captured["project"], "oht-routing-contextual-sac")
        self.assertEqual(captured["config"]["EXP_META"], meta)
        self.assertTrue({"sac/alpha", "sac/entropy"} <= set(WANDB_METRIC_KEYS))

    def test_real_contextual_sac_crosses_runtime_training_gate(self):
        config = ContextualRuntimeConfig(
            algorithm="sac",
            mode="training",
            action_enabled=True,
            action_scale=0.05,
            warmup_steps=2,
            episode_burnin_steps=0,
            normalizer_freeze_steps=2,
            device="cpu",
            seed=17,
            replay_capacity_env_steps=8,
            replay_sampling_mode=REPLAY_SAMPLING_RAIL,
            batch_size=8,
            minimum_replay_env_steps=2,
            minimum_action_enabled_env_steps=2,
            latest_checkpoint_interval=10_000,
            periodic_checkpoint_interval=10_000,
            wandb_enabled=False,
            sale_enabled=False,
            lap_enabled=False,
            critic_loss_mode="mse",
        )
        runtime = ClientAlgorithm(config)
        runtime.topology = make_topology()
        runtime.observation_builder = TrainingObservationBuilder(
            runtime.topology, freeze_steps=2
        )
        pclient = make_runtime_pclient()
        runtime._ensure_initialized(pclient)
        self.assertIsInstance(runtime.learner, ContextualSACLearner)

        for _ in range(7):
            runtime.Algorithm(pclient)

        self.assertGreaterEqual(runtime.learner.learner_update_count, 1)
        self.assertTrue(math.isfinite(runtime.last_diagnostics["sac/alpha"]))
        self.assertEqual(
            runtime.last_diagnostics["action/exploration_noise_std"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
