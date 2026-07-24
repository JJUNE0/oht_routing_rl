import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cocel_rl.algorithms.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
)
from cocel_rl.algorithms.contextual_td7.checkpoint import (
    ContextualCheckpointError,
    load_contextual_checkpoint,
    save_contextual_checkpoint,
)
from contextual_observation import RunningFeatureNormalizer
from contextual_reward import ContextualRewardBuilder
from test_contextual_learner import SMALL_NETWORK
from test_contextual_observation import make_topology
from test_contextual_replay import FakeObservationBuilder, make_snapshot


class CheckpointObservationBuilder(FakeObservationBuilder):
    def __init__(self, topology):
        super().__init__(topology)
        self.local_normalizer = RunningFeatureNormalizer(8)
        self.global_normalizer = RunningFeatureNormalizer(6)
        self.local_normalizer.update(
            np.ones((4999, 8)), name="checkpoint_local"
        )
        self.global_normalizer.update(
            np.ones((1, 6)), name="checkpoint_global"
        )
        self.local_normalizer.freeze()
        self.global_normalizer.freeze()


def components(seed=17, *, sale=False, lap=False):
    topology = make_topology()
    builder = CheckpointObservationBuilder(topology)
    replay = ContextualStepReplayBuffer(
        topology, builder, capacity_env_steps=8, seed=seed,
        lap_enabled=lap,
    )
    for step in range(8):
        snapshot = make_snapshot(topology, step)
        policy = np.sin(
            np.arange(4996, dtype=np.float32) + step
        )[:, None]
        replay.push(replace(
            snapshot,
            policy_action=policy,
            applied_action=0.1 * policy,
            reward=np.tanh(snapshot.reward / 1000),
        ))
    reward_builder = ContextualRewardBuilder(topology)
    config = ContextualLearnerConfig(
        batch_size=16, target_noise=0.01,
        target_update_interval=4, policy_update_delay=2,
        minimum_replay_env_steps=1,
        minimum_action_enabled_env_steps=1,
        sale_enabled=sale,
        lap_enabled=lap,
        sale_embedding_dim=16,
        sale_feature_dim=16,
    )
    learner = ContextualTD7Learner(
        replay, network_config=SMALL_NETWORK, config=config,
        device="cpu", seed=seed,
    )
    return learner, builder, reward_builder


class ContextualCheckpointTests(unittest.TestCase):
    def test_exploration_rng_round_trip_and_legacy_fallback(self):
        learner, obs, reward = components()
        source_rng = np.random.default_rng(123)
        source_rng.normal(size=17)
        expected_next = source_rng.normal(size=8)

        # Recreate the state immediately before expected_next.
        source_rng = np.random.default_rng(123)
        source_rng.normal(size=17)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "checkpoint.pt", learner,
                observation_builder=obs, reward_builder=reward,
                runtime_metadata={"runtime_env_step": 92_000, "episode_id": 1},
                exploration_rng=source_rng,
            )
            restored_rng = np.random.default_rng(999)
            load_contextual_checkpoint(
                path, learner, observation_builder=obs,
                reward_builder=reward, exploration_rng=restored_rng,
                exploration_seed=123,
            )
            np.testing.assert_array_equal(
                restored_rng.normal(size=8), expected_next
            )

            legacy = torch.load(path, weights_only=False)
            legacy.pop("exploration_rng_state")
            legacy_path = Path(directory) / "legacy.pt"
            torch.save(legacy, legacy_path)
            first = np.random.default_rng(123)
            second = np.random.default_rng(123)
            load_contextual_checkpoint(
                legacy_path, learner, observation_builder=obs,
                reward_builder=reward, exploration_rng=first,
                exploration_seed=123,
            )
            load_contextual_checkpoint(
                legacy_path, learner, observation_builder=obs,
                reward_builder=reward, exploration_rng=second,
                exploration_seed=123,
            )
            np.testing.assert_array_equal(
                first.normal(size=8), second.normal(size=8)
            )
            self.assertFalse(np.array_equal(
                first.normal(size=8),
                np.random.default_rng(123).normal(size=8),
            ))

    def test_round_trip_forward_and_next_update_reproducible(self):
        source, source_obs, source_reward = components()
        source.update()
        source.update()
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "checkpoint.pt", source,
                observation_builder=source_obs,
                reward_builder=source_reward,
            )
            restored, restored_obs, restored_reward = components(seed=99)
            load_contextual_checkpoint(
                path, restored, observation_builder=restored_obs,
                reward_builder=restored_reward,
            )
            fixed = source.replay.sample(8)
            with torch.no_grad():
                source_z = source.encoder(
                    fixed.center_local, fixed.incoming_local,
                    fixed.outgoing_local, fixed.incoming_relation,
                    fixed.outgoing_relation, fixed.global_state,
                ).state
                restored_z = restored.encoder(
                    fixed.center_local, fixed.incoming_local,
                    fixed.outgoing_local, fixed.incoming_relation,
                    fixed.outgoing_relation, fixed.global_state,
                ).state
                torch.testing.assert_close(source_z, restored_z, rtol=0, atol=0)
                torch.testing.assert_close(
                    source.actor(source_z).action,
                    restored.actor(restored_z).action, rtol=0, atol=0,
                )

            # Restore once more because the fixed sample advanced source replay
            # RNG after the checkpoint. Both next updates now start identically.
            load_contextual_checkpoint(
                path, source, observation_builder=source_obs,
                reward_builder=source_reward,
            )
            source_result = source.update()
            load_contextual_checkpoint(
                path, restored, observation_builder=restored_obs,
                reward_builder=restored_reward,
            )
            restored_result = restored.update()
            self.assertEqual(
                source_result.diagnostics["learner/critic_loss"],
                restored_result.diagnostics["learner/critic_loss"],
            )
            for left, right in zip(
                source.critic.parameters(), restored.critic.parameters()
            ):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_hash_version_and_config_mismatch_rejected(self):
        learner, obs, reward = components()
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "checkpoint.pt", learner,
                observation_builder=obs, reward_builder=reward,
            )
            payload = torch.load(path, weights_only=False)
            for key, value in (
                ("topology_hash", "wrong"),
                ("mapping_hash", "wrong"),
                ("learner_version", "wrong"),
                ("action_version", "wrong"),
                ("action_scale", 0.9),
            ):
                changed = dict(payload)
                changed[key] = value
                bad = Path(directory) / f"{key}.pt"
                torch.save(changed, bad)
                target, target_obs, target_reward = components()
                with self.assertRaises(ContextualCheckpointError):
                    load_contextual_checkpoint(
                        bad, target, observation_builder=target_obs,
                        reward_builder=target_reward,
                    )

    def test_checkpoint_explicitly_excludes_replay_payload(self):
        learner, obs, reward = components()
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "checkpoint.pt", learner,
                observation_builder=obs, reward_builder=reward,
            )
            payload = torch.load(path, weights_only=False)
            self.assertFalse(payload["replay_saved"])
            self.assertNotIn("replay", payload)

    def test_crash_checkpoint_has_failure_metadata_and_resume_is_refused(self):
        learner, obs, reward = components()
        metadata = {
            "checkpoint_kind": "crash",
            "training_failed": True,
            "failure_type": "FloatingPointError",
            "failure_message": "synthetic NaN",
            "failure_traceback": "traceback text",
            "failure_env_step": 17,
            "failure_episode_id": 2,
            "algorithm_variant": learner.config.algorithm_variant,
            "sale_enabled": learner.config.sale_enabled,
            "lap_enabled": learner.config.lap_enabled,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "crash.pt", learner,
                observation_builder=obs,
                reward_builder=reward,
                runtime_metadata=metadata,
            )
            payload = torch.load(path, weights_only=False)
            for key, value in metadata.items():
                self.assertEqual(payload["runtime_metadata"][key], value)
            target, target_obs, target_reward = components(seed=99)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "Crash checkpoint resume refused"
            ):
                load_contextual_checkpoint(
                    path,
                    target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                )

    def test_sale_lap_metadata_mismatch_is_rejected(self):
        learner, obs, reward = components()
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "checkpoint.pt", learner,
                observation_builder=obs, reward_builder=reward,
            )
            payload = torch.load(path, weights_only=False)
            for key, value in (
                ("sale_enabled", True),
                ("lap_enabled", True),
                ("algorithm_variant", "wrong"),
                ("sale_version", "wrong"),
                ("lap_version", "wrong"),
            ):
                changed = dict(payload)
                changed[key] = value
                bad = Path(directory) / f"mismatch_{key}.pt"
                torch.save(changed, bad)
                target, target_obs, target_reward = components()
                with self.assertRaises(ContextualCheckpointError):
                    load_contextual_checkpoint(
                        bad, target, observation_builder=target_obs,
                        reward_builder=target_reward,
                    )

    def test_sale_checkpoint_restores_all_generations_and_output(self):
        source, obs, reward = components(sale=True, lap=True)
        source.update()
        source._hard_update_targets()
        batch = source.replay.sample(4)
        observation = (
            batch.center_local, batch.incoming_local, batch.outgoing_local,
            batch.incoming_relation, batch.outgoing_relation,
            batch.global_state,
        )
        with torch.no_grad():
            expected = source.sale_fixed.state(observation)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "sale_lap.pt", source,
                observation_builder=obs, reward_builder=reward,
            )
            restored, restored_obs, restored_reward = components(
                seed=99, sale=True, lap=True
            )
            load_contextual_checkpoint(
                path, restored, observation_builder=restored_obs,
                reward_builder=restored_reward,
            )
            with torch.no_grad():
                actual = restored.sale_fixed.state(observation)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertEqual(
                restored.sale_fixed_generation,
                source.sale_fixed_generation,
            )
            self.assertEqual(
                restored.current_target_q_min,
                source.current_target_q_min,
            )
            self.assertEqual(
                restored.current_target_q_max,
                source.current_target_q_max,
            )
            self.assertEqual(
                restored.fixed_target_q_min,
                source.fixed_target_q_min,
            )
            self.assertEqual(
                restored.fixed_target_q_max,
                source.fixed_target_q_max,
            )


if __name__ == "__main__":
    unittest.main()
