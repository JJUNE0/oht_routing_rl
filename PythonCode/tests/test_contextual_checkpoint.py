import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
)
from oht_routing.algorithms.rl.contextual_td7.checkpoint import (
    CHECKPOINT_VERSION,
    ContextualCheckpointError,
    load_contextual_checkpoint,
    read_contextual_runtime_config,
    save_contextual_checkpoint,
)
from oht_routing.mdp.observation import RunningFeatureNormalizer
from oht_routing.mdp.reward.builder import ContextualRewardBuilder, ContextualRewardConfig
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


def components(
    seed=17, *, sale=False, lap=False, reward_version="N", batch_size=16
):
    topology = make_topology()
    builder = CheckpointObservationBuilder(topology)
    replay = ContextualStepReplayBuffer(
        topology, builder, capacity_env_steps=8, seed=seed,
        lap_enabled=lap,
        reward_version=reward_version,
    )
    for step in range(8):
        snapshot = make_snapshot(topology, step)
        policy = np.sin(
            np.arange(4996, dtype=np.float32) + step
        )[:, None]
        applied = 0.1 * policy
        previous_applied = 0.1 * np.sin(
            np.arange(4996, dtype=np.float32) + max(step - 1, 0)
        )[:, None]
        replay.push(replace(
            snapshot,
            reward_version=reward_version,
            previous_applied_action=previous_applied,
            policy_action=policy,
            applied_action=applied,
            next_previous_applied_action=applied,
            reward=np.tanh(snapshot.reward / 1000),
        ))
    reward_builder = ContextualRewardBuilder(
        topology,
        ContextualRewardConfig.for_version(reward_version),
    )
    config = ContextualLearnerConfig(
        batch_size=batch_size, target_noise=0.01,
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
    def test_reward_step_counter_round_trips(self):
        learner, obs, reward = components()
        reward.reward_steps = 12_345
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "reward_steps.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
            )
            target, target_obs, target_reward = components(seed=99)
            self.assertEqual(target_reward.reward_steps, 0)
            load_contextual_checkpoint(
                path,
                target,
                observation_builder=target_obs,
                reward_builder=target_reward,
            )
            self.assertEqual(target_reward.reward_steps, 12_345)

    def test_checkpoint_rejects_non_n_reward_identity(self):
        learner, obs, reward = components()
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "reward_n.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
            )
            payload = torch.load(path, weights_only=False)
            payload["reward_version"] = "U"
            incompatible = Path(directory) / "reward_u.pt"
            torch.save(payload, incompatible)
            target, target_obs, target_reward = components(seed=99)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "reward_version mismatch"
            ):
                load_contextual_checkpoint(
                    incompatible,
                    target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                )

    def test_resume_only_runtime_metadata_upgrade_is_narrow(self):
        learner, obs, reward = components()
        legacy_metadata = {
            "algorithm_version": "algorithm_core",
            "state_normalizer_warmup_bypass": False,
            "effective_warmup_steps": 10_000,
            "reward_normalizer_reuse": False,
            "tat_termination_start_episode": 2,
            "early_stop_tat_threshold": 180.0,
        }
        expected = {
            "algorithm_version": (
                "algorithm_core_action_scale_schedule_v1_"
                "rewardnormreuse1_ep1_normreuse1"
            ),
            "action_scale_schedule_version": "action_scale_schedule_v1",
            "state_normalizer_snapshot_version": "state_snapshot_v1",
            "state_normalizer_warmup_bypass": True,
            "effective_warmup_steps": 0,
            "warmup_episode_transition_version": "warmup_boundary_v1",
            "terminate_on_warmup_complete": True,
            "reward_normalizer_snapshot_version": "reward_snapshot_v1",
            "reward_normalizer_reuse": True,
            "tat_termination_policy_version": "tat_policy_v1",
            "tat_termination_policy": "episode2_tat180",
            "tat_termination_configured_start_episode": 2,
            "tat_termination_start_episode": 1,
            "early_stop_tat_threshold": 180.0,
            "resume_inference_until_replay_full": False,
            "resume_refill_target_env_steps": 100,
            "resume_deterministic_first_episode": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "legacy_lifecycle.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
                runtime_metadata=legacy_metadata,
            )
            load_contextual_checkpoint(
                path,
                learner,
                observation_builder=obs,
                reward_builder=reward,
                expected_runtime_metadata=expected,
                allow_resume_metadata_upgrade=True,
            )
            incompatible = dict(expected)
            incompatible["early_stop_tat_threshold"] = 175.0
            with self.assertRaisesRegex(
                ContextualCheckpointError,
                "early_stop_tat_threshold mismatch",
            ):
                load_contextual_checkpoint(
                    path,
                    learner,
                    observation_builder=obs,
                    reward_builder=reward,
                    expected_runtime_metadata=incompatible,
                    allow_resume_metadata_upgrade=True,
                )

    def test_resume_can_override_only_batch_size_in_learner_config(self):
        learner, obs, reward = components(batch_size=16)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "batch16.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
            )
            target, target_obs, target_reward = components(
                seed=99, batch_size=32
            )
            with self.assertRaisesRegex(
                ContextualCheckpointError, "learner_config mismatch"
            ):
                load_contextual_checkpoint(
                    path,
                    target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                )
            load_contextual_checkpoint(
                path,
                target,
                observation_builder=target_obs,
                reward_builder=target_reward,
                allowed_learner_config_overrides={"batch_size"},
            )

    def test_legacy_tat_metadata_may_be_proven_by_runtime_config(self):
        learner, obs, reward = components()
        tat_config = {
            "tat_termination_enabled": True,
            "tat_termination_grace_steps": 10_000,
            "early_stop_tat_threshold": 200.0,
            "tat_above_threshold_patience": 300,
            "tat_termination_inclusive": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "legacy_tat_metadata.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
                runtime_config=tat_config,
                runtime_metadata={},
            )
            load_contextual_checkpoint(
                path,
                learner,
                observation_builder=obs,
                reward_builder=reward,
                expected_runtime_metadata=tat_config,
                allow_resume_metadata_upgrade=True,
            )
            incompatible = dict(tat_config)
            incompatible["early_stop_tat_threshold"] = 180.0
            with self.assertRaisesRegex(
                ContextualCheckpointError,
                "early_stop_tat_threshold mismatch",
            ):
                load_contextual_checkpoint(
                    path,
                    learner,
                    observation_builder=obs,
                    reward_builder=reward,
                    expected_runtime_metadata=incompatible,
                    allow_resume_metadata_upgrade=True,
                )

    def test_full_runtime_config_round_trip_and_v3_partial_compatibility(self):
        learner, obs, reward = components()
        runtime_config = {
            "warmup_steps": 12_345,
            "exploration_noise_std": 0.17,
            "exploration_noise_final_std": 0.03,
            "exploration_noise_anneal_steps": 87_654,
            "exploration_noise_clip": 0.19,
            "smooth_b_rl_weight": 0.07,
            "updates_per_env_step": 3,
        }
        metadata = {
            "action_mode": "region_b_rl",
            "action_scale": 0.05,
            "exploration_noise_std": 0.17,
            "exploration_noise_final_std": 0.03,
            "exploration_noise_anneal_steps": 87_654,
            "exploration_noise_anneal_start_step": 12_345,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "checkpoint.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
                runtime_metadata=metadata,
                runtime_config=runtime_config,
            )
            restored, complete = read_contextual_runtime_config(path)
            self.assertTrue(complete)
            self.assertEqual(restored, runtime_config)

            payload = torch.load(path, weights_only=False)
            payload["checkpoint_version"] = (
                "contextual_td7_checkpoint_v3_independent_twin_critic"
            )
            payload.pop("runtime_config")
            legacy_path = Path(directory) / "v3.pt"
            torch.save(payload, legacy_path)
            restored, complete = read_contextual_runtime_config(legacy_path)
            self.assertFalse(complete)
            self.assertEqual(restored["action_mode"], "region_b_rl")
            self.assertEqual(restored["action_scale"], 0.05)
            self.assertEqual(restored["warmup_steps"], 12_345)
            self.assertNotIn("exploration_noise_clip", restored)
            load_contextual_checkpoint(
                legacy_path,
                learner,
                observation_builder=obs,
                reward_builder=reward,
            )

            payload = torch.load(path, weights_only=False)
            for reward_version in (
                "contextual_controlled_reward_v4_balanced_global_local",
                "contextual_controlled_reward_v5_completion_dedup",
                "contextual_controlled_reward_v6_oht_state_cycle",
                "contextual_controlled_reward_v7_oht_cycle_segments_boundary_delta",
            ):
                payload["reward_version"] = reward_version
                torch.save(payload, legacy_path)
                with self.assertRaisesRegex(
                    ContextualCheckpointError, "reward_version mismatch"
                ):
                    load_contextual_checkpoint(
                        legacy_path,
                        learner,
                        observation_builder=obs,
                        reward_builder=reward,
                    )

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
                    source.actor(
                        source_z,
                        previous_action=fixed.previous_applied_action,
                    ).action,
                    restored.actor(
                        restored_z,
                        previous_action=fixed.previous_applied_action,
                    ).action,
                    rtol=0,
                    atol=0,
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
            source_q1 = dict(source.critic.q1.named_parameters())
            source_q2 = dict(source.critic.q2.named_parameters())
            restored_q1 = dict(restored.critic.q1.named_parameters())
            restored_q2 = dict(restored.critic.q2.named_parameters())
            for name in source_q1:
                torch.testing.assert_close(
                    source_q1[name], restored_q1[name], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    source_q2[name], restored_q2[name], rtol=0, atol=0
                )
            self.assertGreater(
                restored.twin_parameter_diagnostics()[
                    "critic/parameter_max_abs_diff"
                ],
                0.0,
            )

    def test_legacy_and_symmetric_checkpoint_resume_are_rejected(self):
        learner, obs, reward = components()
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "checkpoint.pt", learner,
                observation_builder=obs, reward_builder=reward,
            )
            payload = torch.load(path, weights_only=False)
            legacy = dict(payload)
            legacy["checkpoint_version"] = "contextual_td7_checkpoint_v2"
            legacy_path = Path(directory) / "legacy.pt"
            torch.save(legacy, legacy_path)
            target, target_obs, target_reward = components(seed=99)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "fresh independent twin-critic"
            ):
                load_contextual_checkpoint(
                    legacy_path, target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                )

            symmetric = dict(payload)
            symmetric["checkpoint_version"] = CHECKPOINT_VERSION
            symmetric["online_critic"] = {
                key: value.clone()
                for key, value in payload["online_critic"].items()
            }
            for key in list(symmetric["online_critic"]):
                if key.startswith("q2."):
                    symmetric["online_critic"][key] = symmetric[
                        "online_critic"
                    ]["q1." + key[3:]].clone()
            symmetric_path = Path(directory) / "symmetric.pt"
            torch.save(symmetric, symmetric_path)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "symmetric twin-critic"
            ):
                load_contextual_checkpoint(
                    symmetric_path, target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                )

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
                ("reward_version", "wrong"),
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
