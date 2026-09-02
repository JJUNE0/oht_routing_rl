import copy
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
)
from oht_routing.algorithms.rl.contextual_td7.checkpoint import (
    PROMOTED_CHECKPOINTS,
    ContextualCheckpointError,
    load_frozen_contextual_policy,
    load_contextual_checkpoint,
    read_contextual_runtime_config,
    save_contextual_checkpoint,
)
from oht_routing.version import CONTEXTUAL_VERSION
from oht_routing.mdp.observation import (
    CRITIC_EXTRA_DIM,
    GLOBAL_DIM,
    LOCAL_PHYSICAL_DIM,
    RunningFeatureNormalizer,
)
from oht_routing.mdp.reward.builder import ContextualRewardBuilder, ContextualRewardConfig
from oht_routing.mdp.reward.config import REWARD_VERSION
from oht_routing.runtime.config import restore_checkpoint_runtime_config
from test_contextual_learner import SMALL_NETWORK
from test_contextual_observation import make_topology
from test_contextual_replay import FakeObservationBuilder, make_snapshot


class CheckpointObservationBuilder(FakeObservationBuilder):
    def __init__(self, topology):
        super().__init__(topology)
        self.local_normalizer = RunningFeatureNormalizer(LOCAL_PHYSICAL_DIM)
        self.global_normalizer = RunningFeatureNormalizer(GLOBAL_DIM)
        self.critic_normalizer = RunningFeatureNormalizer(CRITIC_EXTRA_DIM)
        self.local_normalizer.update(
            np.ones((4999, LOCAL_PHYSICAL_DIM)), name="checkpoint_local"
        )
        self.global_normalizer.update(
            np.ones((1, GLOBAL_DIM)), name="checkpoint_global"
        )
        self.critic_normalizer.update(
            np.ones((1, CRITIC_EXTRA_DIM)), name="checkpoint_critic_total_tat"
        )
        self.local_normalizer.freeze()
        self.global_normalizer.freeze()
        self.critic_normalizer.freeze()


def components(
    seed=17,
    *,
    sale=False,
    lap=False,
    reward_version=REWARD_VERSION,
    batch_size=16,
    use_attention=False,
    populate_replay=True,
):
    topology = make_topology()
    builder = CheckpointObservationBuilder(topology)
    replay = ContextualStepReplayBuffer(
        topology, builder, capacity_env_steps=8, seed=seed,
        lap_enabled=lap,
        reward_version=reward_version,
    )
    if populate_replay:
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
        replay,
        network_config=replace(
            SMALL_NETWORK, use_attention=bool(use_attention)
        ),
        config=config,
        device="cpu", seed=seed,
    )
    return learner, builder, reward_builder


class ContextualCheckpointTests(unittest.TestCase):
    def test_v6_stage1_policy_is_rejected_by_v7_action_contract(self):
        source, source_obs, reward = components(seed=17, sale=True)
        source.set_applied_action_scale(1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "stage1_v6_1.pt",
                source,
                observation_builder=source_obs,
                reward_builder=reward,
                runtime_metadata={"stage": 1},
            )
            payload = torch.load(path, weights_only=False)
            payload["version"] = "v6.1.0"
            payload["network_config"] = dict(payload["network_config"])
            payload["network_config"]["global_dim"] = 5
            torch.save(payload, path)

            stage2, stage2_obs, _ = components(seed=99, sale=True)
            with self.assertRaisesRegex(
                ContextualCheckpointError,
                "checkpoint version mismatch",
            ):
                load_frozen_contextual_policy(
                    path,
                    stage2,
                    observation_builder=stage2_obs,
                    expected_reward_version=REWARD_VERSION,
                )

    def test_v6_resume_is_rejected_by_v7_action_contract(self):
        source, source_obs, reward = components(seed=17, sale=True)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "resume_v6_1.pt",
                source,
                observation_builder=source_obs,
                reward_builder=reward,
            )
            payload = torch.load(path, weights_only=False)
            payload["version"] = "v6.1.0"
            payload["network_config"] = dict(payload["network_config"])
            payload["network_config"]["global_dim"] = 5
            torch.save(payload, path)

            target, target_obs, target_reward = components(seed=99, sale=True)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "checkpoint version mismatch"
            ):
                load_contextual_checkpoint(
                    path,
                    target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                )

    def test_stage1_policy_only_load_is_frozen_and_does_not_mutate_stage2(self):
        source, source_obs, reward = components(seed=17, sale=True)
        source.set_applied_action_scale(0.73)
        source_obs.local_normalizer = RunningFeatureNormalizer(
            LOCAL_PHYSICAL_DIM
        )
        source_obs.global_normalizer = RunningFeatureNormalizer(GLOBAL_DIM)
        source_obs.critic_normalizer = RunningFeatureNormalizer(
            CRITIC_EXTRA_DIM
        )
        source_obs.local_normalizer.update(
            np.full((3, LOCAL_PHYSICAL_DIM), 2.0), name="stage1_local"
        )
        source_obs.global_normalizer.update(
            np.full((3, GLOBAL_DIM), 3.0), name="stage1_global"
        )
        source_obs.critic_normalizer.update(
            np.full((3, CRITIC_EXTRA_DIM), 4.0), name="stage1_critic"
        )
        for normalizer in (
            source_obs.local_normalizer,
            source_obs.global_normalizer,
            source_obs.critic_normalizer,
        ):
            normalizer.freeze()

        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "stage1.pt",
                source,
                observation_builder=source_obs,
                reward_builder=reward,
                runtime_metadata={
                    "stage": 1,
                    "runtime_env_step": 34_000,
                },
            )
            stage2, stage2_obs, _ = components(seed=99, sale=True)
            stage2_modules_before = {
                name: {
                    key: value.detach().clone()
                    for key, value in module.state_dict().items()
                }
                for name, module in (
                    ("encoder", stage2.encoder),
                    ("actor", stage2.actor),
                    ("critic", stage2.critic),
                )
            }
            optimizer_before = {
                "encoder": copy.deepcopy(stage2.encoder_optimizer.state_dict()),
                "actor": copy.deepcopy(stage2.actor_optimizer.state_dict()),
                "critic": copy.deepcopy(stage2.critic_optimizer.state_dict()),
            }
            replay_rng_before = copy.deepcopy(
                stage2.replay.rng.bit_generator.state
            )
            python_rng_before = random.getstate()
            numpy_rng_before = np.random.get_state()
            torch_rng_before = torch.get_rng_state().clone()

            frozen = load_frozen_contextual_policy(
                path,
                stage2,
                observation_builder=stage2_obs,
                expected_reward_version=REWARD_VERSION,
            )

            self.assertEqual(frozen.applied_action_scale, 0.73)
            self.assertEqual(frozen.runtime_metadata["runtime_env_step"], 34_000)
            self.assertFalse(frozen.encoder.training)
            self.assertFalse(frozen.actor.training)
            self.assertFalse(frozen.sale_fixed.training)
            self.assertTrue(all(
                not parameter.requires_grad
                for module in (
                    frozen.encoder, frozen.actor, frozen.sale_fixed
                )
                for parameter in module.parameters()
            ))
            for frozen_module, source_module in (
                (frozen.encoder, source.encoder),
                (frozen.actor, source.actor),
                (frozen.sale_fixed, source.sale_fixed),
            ):
                for key, expected in source_module.state_dict().items():
                    torch.testing.assert_close(
                        frozen_module.state_dict()[key], expected,
                        rtol=0, atol=0,
                    )

            frozen_ids = {
                id(parameter)
                for module in (
                    frozen.encoder, frozen.actor, frozen.sale_fixed
                )
                for parameter in module.parameters()
            }
            stage2_ids = {
                id(parameter)
                for module in (
                    stage2.encoder, stage2.actor, stage2.sale_fixed
                )
                for parameter in module.parameters()
            }
            self.assertFalse(frozen_ids & stage2_ids)
            for name, module in (
                ("encoder", stage2.encoder),
                ("actor", stage2.actor),
                ("critic", stage2.critic),
            ):
                for key, expected in stage2_modules_before[name].items():
                    torch.testing.assert_close(
                        module.state_dict()[key], expected, rtol=0, atol=0
                    )
            self.assertEqual(
                stage2.encoder_optimizer.state_dict(), optimizer_before["encoder"]
            )
            self.assertEqual(
                stage2.actor_optimizer.state_dict(), optimizer_before["actor"]
            )
            self.assertEqual(
                stage2.critic_optimizer.state_dict(), optimizer_before["critic"]
            )
            self.assertEqual(
                stage2.replay.rng.bit_generator.state, replay_rng_before
            )
            self.assertEqual(random.getstate(), python_rng_before)
            numpy_rng_after = np.random.get_state()
            self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
            np.testing.assert_array_equal(
                numpy_rng_after[1], numpy_rng_before[1]
            )
            self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])
            torch.testing.assert_close(
                torch.get_rng_state(), torch_rng_before, rtol=0, atol=0
            )
            np.testing.assert_array_equal(
                stage2_obs.local_normalizer.mean,
                source_obs.local_normalizer.mean,
            )
            np.testing.assert_array_equal(
                stage2_obs.global_normalizer.mean,
                source_obs.global_normalizer.mean,
            )
            np.testing.assert_array_equal(
                stage2_obs.critic_normalizer.mean,
                source_obs.critic_normalizer.mean,
            )

    def test_stage1_policy_warm_starts_only_a_pristine_stage2_policy_path(self):
        source, source_obs, reward = components(seed=17, sale=True)
        source.set_applied_action_scale(0.73)
        # Give the Stage 1 checkpoint non-zero optimizer/counter state so this
        # test can detect an accidental full training-state restore.
        source.update()
        source.update()
        source_obs.local_normalizer = RunningFeatureNormalizer(
            LOCAL_PHYSICAL_DIM
        )
        source_obs.global_normalizer = RunningFeatureNormalizer(GLOBAL_DIM)
        source_obs.critic_normalizer = RunningFeatureNormalizer(
            CRITIC_EXTRA_DIM
        )
        source_obs.local_normalizer.update(
            np.full((3, LOCAL_PHYSICAL_DIM), 2.0), name="stage1_local"
        )
        source_obs.global_normalizer.update(
            np.full((3, GLOBAL_DIM), 3.0), name="stage1_global"
        )
        source_obs.critic_normalizer.update(
            np.full((3, CRITIC_EXTRA_DIM), 4.0), name="stage1_critic"
        )
        for normalizer in (
            source_obs.local_normalizer,
            source_obs.global_normalizer,
            source_obs.critic_normalizer,
        ):
            normalizer.freeze()
        # Make every lagged policy representation observably different. The
        # fresh Stage 2 targets must sync to online encoder/actor, while all
        # SALE roles must use the behaviorally active saved sale_fixed basis.
        with torch.no_grad():
            for parameter in source.target_encoder.parameters():
                parameter.fill_(0.91)
            for parameter in source.target_actor.parameters():
                parameter.fill_(0.92)
            for value, module in (
                (0.31, source.sale_online),
                (0.32, source.sale_fixed),
                (0.33, source.sale_target_fixed),
            ):
                for parameter in module.parameters():
                    parameter.fill_(value)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "stage1.pt",
                source,
                observation_builder=source_obs,
                reward_builder=reward,
                runtime_metadata={"stage": 1},
            )
            stage2, stage2_obs, _ = components(
                seed=99, sale=True, populate_replay=False
            )

            def clone_state(module):
                return {
                    key: value.detach().clone()
                    for key, value in module.state_dict().items()
                }

            critic_before = clone_state(stage2.critic)
            target_critic_before = clone_state(stage2.target_critic)
            optimizer_before = {
                name: copy.deepcopy(optimizer.state_dict())
                for name, optimizer in (
                    ("encoder", stage2.encoder_optimizer),
                    ("actor", stage2.actor_optimizer),
                    ("critic", stage2.critic_optimizer),
                    ("sale", stage2.sale_optimizer),
                )
            }
            module_ids_before = {
                name: id(module)
                for name, module in (
                    ("encoder", stage2.encoder),
                    ("actor", stage2.actor),
                    ("target_encoder", stage2.target_encoder),
                    ("target_actor", stage2.target_actor),
                    ("sale_online", stage2.sale_online),
                    ("sale_fixed", stage2.sale_fixed),
                    ("sale_target_fixed", stage2.sale_target_fixed),
                )
            }
            replay_rng_before = copy.deepcopy(
                stage2.replay.rng.bit_generator.state
            )
            python_rng_before = random.getstate()
            numpy_rng_before = np.random.get_state()
            torch_rng_before = torch.get_rng_state().clone()
            applied_scale_before = stage2.applied_action_scale

            frozen = load_frozen_contextual_policy(
                path,
                stage2,
                observation_builder=stage2_obs,
                expected_reward_version=REWARD_VERSION,
                initialize_fresh_learner_policy=True,
            )

            for actual in (stage2.encoder, stage2.target_encoder):
                for key, expected in source.encoder.state_dict().items():
                    torch.testing.assert_close(
                        actual.state_dict()[key], expected, rtol=0, atol=0
                    )
            for actual in (stage2.actor, stage2.target_actor):
                for key, expected in source.actor.state_dict().items():
                    torch.testing.assert_close(
                        actual.state_dict()[key], expected, rtol=0, atol=0
                    )
            for actual in (
                stage2.sale_online,
                stage2.sale_fixed,
                stage2.sale_target_fixed,
            ):
                for key, expected in source.sale_fixed.state_dict().items():
                    torch.testing.assert_close(
                        actual.state_dict()[key], expected, rtol=0, atol=0
                    )
            self.assertTrue(any(
                not torch.equal(
                    stage2.target_encoder.state_dict()[key], expected
                )
                for key, expected in source.target_encoder.state_dict().items()
            ))
            self.assertTrue(any(
                not torch.equal(
                    stage2.target_actor.state_dict()[key], expected
                )
                for key, expected in source.target_actor.state_dict().items()
            ))
            self.assertTrue(any(
                not torch.equal(
                    stage2.sale_online.state_dict()[key], expected
                )
                for key, expected in source.sale_online.state_dict().items()
            ))

            self.assertEqual(
                module_ids_before,
                {
                    name: id(module)
                    for name, module in (
                        ("encoder", stage2.encoder),
                        ("actor", stage2.actor),
                        ("target_encoder", stage2.target_encoder),
                        ("target_actor", stage2.target_actor),
                        ("sale_online", stage2.sale_online),
                        ("sale_fixed", stage2.sale_fixed),
                        ("sale_target_fixed", stage2.sale_target_fixed),
                    )
                },
            )
            self.assertTrue(all(
                parameter.requires_grad
                for module in (
                    stage2.encoder, stage2.actor, stage2.sale_online
                )
                for parameter in module.parameters()
            ))
            self.assertTrue(all(
                not parameter.requires_grad
                for module in (
                    stage2.target_encoder,
                    stage2.target_actor,
                    stage2.sale_fixed,
                    stage2.sale_target_fixed,
                )
                for parameter in module.parameters()
            ))
            for module, expected_state in (
                (stage2.critic, critic_before),
                (stage2.target_critic, target_critic_before),
            ):
                for key, expected in expected_state.items():
                    torch.testing.assert_close(
                        module.state_dict()[key], expected, rtol=0, atol=0
                    )
            for name, optimizer in (
                ("encoder", stage2.encoder_optimizer),
                ("actor", stage2.actor_optimizer),
                ("critic", stage2.critic_optimizer),
                ("sale", stage2.sale_optimizer),
            ):
                self.assertEqual(
                    optimizer.state_dict(), optimizer_before[name]
                )
            self.assertEqual(stage2.replay.size_env_steps, 0)
            self.assertEqual(stage2.learner_update_count, 0)
            self.assertEqual(stage2.actor_update_count, 0)
            self.assertEqual(stage2.target_update_count, 0)
            self.assertEqual(stage2.sale_update_count, 0)
            self.assertEqual(stage2.sale_fixed_generation, 0)
            self.assertEqual(stage2.applied_action_scale, applied_scale_before)
            self.assertEqual(frozen.applied_action_scale, 0.73)
            for source_normalizer, stage2_normalizer in (
                (source_obs.local_normalizer, stage2_obs.local_normalizer),
                (source_obs.global_normalizer, stage2_obs.global_normalizer),
                (source_obs.critic_normalizer, stage2_obs.critic_normalizer),
            ):
                source_state = source_normalizer.state_dict()
                stage2_state = stage2_normalizer.state_dict()
                self.assertEqual(source_state.keys(), stage2_state.keys())
                for key, expected in source_state.items():
                    actual = stage2_state[key]
                    if isinstance(expected, np.ndarray):
                        np.testing.assert_array_equal(actual, expected)
                    else:
                        self.assertEqual(actual, expected)
            self.assertEqual(
                stage2.replay.rng.bit_generator.state, replay_rng_before
            )
            self.assertEqual(random.getstate(), python_rng_before)
            numpy_rng_after = np.random.get_state()
            self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
            np.testing.assert_array_equal(
                numpy_rng_after[1], numpy_rng_before[1]
            )
            self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])
            torch.testing.assert_close(
                torch.get_rng_state(), torch_rng_before, rtol=0, atol=0
            )

            with self.assertRaisesRegex(
                ContextualCheckpointError, "requires a pristine learner"
            ):
                load_frozen_contextual_policy(
                    path,
                    stage2,
                    observation_builder=stage2_obs,
                    expected_reward_version=REWARD_VERSION,
                    initialize_fresh_learner_policy=True,
                )

    def test_stage1_policy_warm_start_supports_sale_disabled(self):
        source, source_obs, reward = components(seed=17, sale=False)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "stage1_no_sale.pt",
                source,
                observation_builder=source_obs,
                reward_builder=reward,
                runtime_metadata={"stage": 1},
            )
            stage2, stage2_obs, _ = components(
                seed=99, sale=False, populate_replay=False
            )
            frozen = load_frozen_contextual_policy(
                path,
                stage2,
                observation_builder=stage2_obs,
                expected_reward_version=REWARD_VERSION,
                initialize_fresh_learner_policy=True,
            )
            self.assertIsNone(frozen.sale_fixed)
            self.assertIsNone(stage2.sale_online)
            self.assertIsNone(stage2.sale_fixed)
            self.assertIsNone(stage2.sale_target_fixed)
            for actual, expected_module in (
                (stage2.encoder, source.encoder),
                (stage2.target_encoder, source.encoder),
                (stage2.actor, source.actor),
                (stage2.target_actor, source.actor),
            ):
                for key, expected in expected_module.state_dict().items():
                    torch.testing.assert_close(
                        actual.state_dict()[key], expected, rtol=0, atol=0
                    )

    def test_stage1_policy_only_load_rejects_crash_and_unfrozen_normalizer(self):
        source, source_obs, reward = components(seed=17, sale=True)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "stage1.pt",
                source,
                observation_builder=source_obs,
                reward_builder=reward,
            )
            target, target_obs, _ = components(seed=99, sale=True)
            payload = torch.load(path, weights_only=False)
            payload["runtime_metadata"] = {"checkpoint_kind": "crash"}
            crash = Path(directory) / "crash.pt"
            torch.save(payload, crash)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "Crash checkpoint"
            ):
                load_frozen_contextual_policy(
                    crash,
                    target,
                    observation_builder=target_obs,
                    expected_reward_version=REWARD_VERSION,
                )

            payload["runtime_metadata"] = {"stage": 2}
            stage2 = Path(directory) / "stage2.pt"
            torch.save(payload, stage2)
            with self.assertRaisesRegex(
                ContextualCheckpointError,
                "requires a Stage 1 checkpoint",
            ):
                load_frozen_contextual_policy(
                    stage2,
                    target,
                    observation_builder=target_obs,
                    expected_reward_version=REWARD_VERSION,
                )

            payload["runtime_metadata"] = {"stage": 1}
            payload["observation_local_normalizer"]["frozen"] = False
            unfrozen = Path(directory) / "unfrozen.pt"
            torch.save(payload, unfrozen)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "local normalizer must be frozen"
            ):
                load_frozen_contextual_policy(
                    unfrozen,
                    target,
                    observation_builder=target_obs,
                    expected_reward_version=REWARD_VERSION,
                )

            payload["observation_local_normalizer"]["frozen"] = True
            payload["applied_action_scale"] = 0.0
            zero_scale = Path(directory) / "zero_scale.pt"
            torch.save(payload, zero_scale)
            with self.assertRaisesRegex(
                ContextualCheckpointError, r"finite and in \(0, 1\]"
            ):
                load_frozen_contextual_policy(
                    zero_scale,
                    target,
                    observation_builder=target_obs,
                    expected_reward_version=REWARD_VERSION,
                )

    def test_distributed_full_resume_is_rejected_before_state_restore(self):
        source, source_obs, reward = components(seed=17, sale=True)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "distributed.pt",
                source,
                observation_builder=source_obs,
                reward_builder=reward,
                runtime_metadata={
                    "distributed": True,
                    "distributed_full_resume_supported": False,
                },
            )
            target, target_obs, target_reward = components(
                seed=99, sale=True
            )
            encoder_before = {
                key: value.detach().clone()
                for key, value in target.encoder.state_dict().items()
            }
            update_count_before = target.learner_update_count

            with self.assertRaisesRegex(
                ContextualCheckpointError,
                "distributed checkpoint full-state training resume",
            ):
                load_contextual_checkpoint(
                    path,
                    target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                    reject_distributed_full_resume=True,
                )

            self.assertEqual(
                target.learner_update_count, update_count_before
            )
            for key, expected in encoder_before.items():
                torch.testing.assert_close(
                    target.encoder.state_dict()[key],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_critic_total_tat_normalizer_round_trips(self):
        learner, obs, reward = components()
        obs.critic_normalizer = RunningFeatureNormalizer(CRITIC_EXTRA_DIM)
        obs.critic_normalizer.update(
            np.asarray([[111.0], [222.0]], dtype=np.float64),
            name="checkpoint_critic_total_tat",
        )
        obs.critic_normalizer.freeze()
        expected = obs.critic_normalizer.state_dict()

        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "critic_normalizer.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
            )
            payload = torch.load(path, weights_only=False)
            saved = payload["observation_critic_total_tat_normalizer"]
            np.testing.assert_array_equal(saved["mean"], expected["mean"])
            np.testing.assert_array_equal(saved["m2"], expected["m2"])
            self.assertEqual(saved["count"], expected["count"])
            self.assertTrue(saved["frozen"])

            target, target_obs, target_reward = components(seed=99)
            load_contextual_checkpoint(
                path,
                target,
                observation_builder=target_obs,
                reward_builder=target_reward,
            )
            restored = target_obs.critic_normalizer.state_dict()
            np.testing.assert_array_equal(restored["mean"], expected["mean"])
            np.testing.assert_array_equal(restored["m2"], expected["m2"])
            self.assertEqual(restored["count"], expected["count"])
            self.assertEqual(
                restored["update_calls"], expected["update_calls"]
            )
            self.assertTrue(restored["frozen"])

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

    def test_v5_checkpoint_is_rejected_by_v6_encoder_contract(self):
        learner, obs, reward = components()
        self.assertEqual(CONTEXTUAL_VERSION, "v7.1.0")
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "v5_0.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
                runtime_config={"reward_version": REWARD_VERSION},
            )
            payload = torch.load(path, weights_only=False)
            payload["version"] = "v5.0.0"
            torch.save(payload, path)

            target, target_obs, target_reward = components(seed=99)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "checkpoint version mismatch"
            ):
                load_contextual_checkpoint(
                    path,
                    target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                )

    def test_same_v6_reward_o_checkpoint_is_rejected_before_restore(self):
        learner, obs, reward = components()
        self.assertEqual(CONTEXTUAL_VERSION, "v7.1.0")
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "reward_p.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
                runtime_config={"reward_version": REWARD_VERSION},
            )
            payload = torch.load(path, weights_only=False)
            payload["version"] = CONTEXTUAL_VERSION
            payload["reward_version"] = "O"
            payload["runtime_config"]["reward_version"] = "O"
            incompatible = Path(directory) / "reward_o.pt"
            torch.save(payload, incompatible)
            with self.assertRaisesRegex(
                ContextualCheckpointError,
                "reward_version mismatch: saved='O', runtime='P'",
            ):
                read_contextual_runtime_config(
                    incompatible, expected_reward_version=REWARD_VERSION
                )
            with self.assertRaisesRegex(
                ContextualCheckpointError,
                "reward_version mismatch: saved='O', runtime='P'",
            ):
                restore_checkpoint_runtime_config(
                    {"reward_version": REWARD_VERSION}, incompatible
                )
            target, target_obs, target_reward = components(seed=99)
            with self.assertRaisesRegex(
                ContextualCheckpointError,
                "reward_version mismatch: saved='O', runtime='P'",
            ):
                load_contextual_checkpoint(
                    incompatible,
                    target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                )

    def test_save_rejects_runtime_config_reward_identity_mismatch(self):
        learner, obs, reward = components()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ContextualCheckpointError,
                "payload='P', runtime_config='O'",
            ):
                save_contextual_checkpoint(
                    Path(directory) / "inconsistent.pt",
                    learner,
                    observation_builder=obs,
                    reward_builder=reward,
                    runtime_config={"reward_version": "O"},
                )

    def test_runtime_metadata_and_learner_config_do_not_gate_loading(self):
        learner, obs, reward = components(batch_size=16)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "batch16.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
                runtime_metadata={"algorithm_version": "retired-name"},
            )
            target, target_obs, target_reward = components(
                seed=99, batch_size=32
            )
            load_contextual_checkpoint(
                path,
                target,
                observation_builder=target_obs,
                reward_builder=target_reward,
            )

    def test_exact_v2_legacy_artifact_fingerprint_is_rejected(self):
        learner, obs, reward = components()
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "promoted.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
                runtime_config={"reward_version": REWARD_VERSION},
            )
            payload = torch.load(path, weights_only=False)
            payload.pop("version")
            payload["checkpoint_version"] = (
                "contextual_td7_checkpoint_v7_locked_reward_profile"
            )
            torch.save(payload, path)
            old_v2_fingerprint = (
                "7c2d6a19184060584efeb69f52be57c3e7fffad33da52c8c1c879ed39db8bc2b"
            )
            self.assertNotIn(old_v2_fingerprint, PROMOTED_CHECKPOINTS)
            with patch(
                "oht_routing.algorithms.rl.contextual_td7.checkpoint."
                "_checkpoint_sha256",
                return_value=old_v2_fingerprint,
            ):
                with self.assertRaisesRegex(
                    ContextualCheckpointError,
                    "checkpoint version mismatch",
                ):
                    read_contextual_runtime_config(path)
                with self.assertRaisesRegex(
                    ContextualCheckpointError,
                    "checkpoint version mismatch",
                ):
                    load_contextual_checkpoint(
                        path,
                        learner,
                        observation_builder=obs,
                        reward_builder=reward,
                    )

    def test_v4_semver_checkpoint_is_rejected(self):
        learner, obs, reward = components()
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "v4_1.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
                runtime_config={"reward_version": REWARD_VERSION},
            )
            payload = torch.load(path, weights_only=False)
            payload["version"] = "v4.1.0"
            torch.save(payload, path)

            with self.assertRaisesRegex(
                ContextualCheckpointError, "checkpoint version mismatch"
            ):
                read_contextual_runtime_config(path)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "checkpoint version mismatch"
            ):
                load_contextual_checkpoint(
                    path,
                    learner,
                    observation_builder=obs,
                    reward_builder=reward,
                )

    def test_full_runtime_config_round_trip(self):
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
            self.assertEqual(payload["version"], CONTEXTUAL_VERSION)
            self.assertNotIn("checkpoint_version", payload)
            for retired_key in (
                "learner_version",
                "observation_version",
                "action_version",
                "replay_version",
                "stack_version",
                "sale_version",
                "lap_version",
            ):
                self.assertNotIn(retired_key, payload)

    def test_replay_eviction_rng_round_trip_and_legacy_fallback(self):
        source, source_obs, source_reward = components(
            populate_replay=False
        )
        source.replay.eviction_rng.integers(10_000, size=17)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "checkpoint.pt",
                source,
                observation_builder=source_obs,
                reward_builder=source_reward,
            )
            expected_next = source.replay.eviction_rng.integers(
                10_000, size=8
            )

            target, target_obs, target_reward = components(
                seed=99, populate_replay=False
            )
            load_contextual_checkpoint(
                path,
                target,
                observation_builder=target_obs,
                reward_builder=target_reward,
            )
            np.testing.assert_array_equal(
                target.replay.eviction_rng.integers(10_000, size=8),
                expected_next,
            )

            legacy = torch.load(path, weights_only=False)
            legacy.pop("replay_eviction_rng_state")
            legacy["version"] = "v7.0.0"
            legacy_path = Path(directory) / "legacy.pt"
            torch.save(legacy, legacy_path)
            legacy_target, legacy_obs, legacy_reward = components(
                seed=101, populate_replay=False
            )
            fallback = np.random.default_rng()
            fallback.bit_generator.state = copy.deepcopy(
                legacy_target.replay.eviction_rng.bit_generator.state
            )
            expected_fallback = fallback.integers(10_000, size=8)
            load_contextual_checkpoint(
                legacy_path,
                legacy_target,
                observation_builder=legacy_obs,
                reward_builder=legacy_reward,
            )
            np.testing.assert_array_equal(
                legacy_target.replay.eviction_rng.integers(10_000, size=8),
                expected_fallback,
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
                    fixed.outgoing_local, fixed.center_rail_index,
                    fixed.incoming_rail_indices,
                    fixed.outgoing_rail_indices,
                    fixed.incoming_relation, fixed.outgoing_relation,
                    fixed.global_state,
                ).state
                restored_z = restored.encoder(
                    fixed.center_local, fixed.incoming_local,
                    fixed.outgoing_local, fixed.center_rail_index,
                    fixed.incoming_rail_indices,
                    fixed.outgoing_rail_indices,
                    fixed.incoming_relation, fixed.outgoing_relation,
                    fixed.global_state,
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
            legacy["version"] = "v1.0.0"
            legacy_path = Path(directory) / "legacy.pt"
            torch.save(legacy, legacy_path)
            target, target_obs, target_reward = components(seed=99)
            with self.assertRaisesRegex(
                ContextualCheckpointError, "checkpoint version mismatch"
            ):
                load_contextual_checkpoint(
                    legacy_path, target,
                    observation_builder=target_obs,
                    reward_builder=target_reward,
                )

            symmetric = dict(payload)
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
                ("version", "v9.9.9"),
                ("topology_hash", "wrong"),
                ("mapping_hash", "wrong"),
                ("network_config", {"wrong": True}),
                ("action_mode", "wrong"),
                ("reward_version", "wrong"),
                ("sale_enabled", True),
                ("lap_enabled", True),
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

    def test_checkpoint_rejects_attention_mode_mismatch(self):
        learner, obs, reward = components(use_attention=False)
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "flat.pt",
                learner,
                observation_builder=obs,
                reward_builder=reward,
            )
            target, target_obs, target_reward = components(
                use_attention=True
            )
            with self.assertRaisesRegex(
                ContextualCheckpointError, "network_config mismatch"
            ):
                load_contextual_checkpoint(
                    path,
                    target,
                    observation_builder=target_obs,
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

    def test_retired_component_versions_do_not_gate_loading(self):
        learner, obs, reward = components()
        with tempfile.TemporaryDirectory() as directory:
            path = save_contextual_checkpoint(
                Path(directory) / "checkpoint.pt", learner,
                observation_builder=obs, reward_builder=reward,
            )
            payload = torch.load(path, weights_only=False)
            payload.update({
                "learner_version": "wrong",
                "observation_version": "wrong",
                "action_version": "wrong",
                "replay_version": "wrong",
                "stack_version": "wrong",
                "sale_version": "wrong",
                "lap_version": "wrong",
                "algorithm_variant": "wrong",
                "action_scale": 0.9,
            })
            changed = Path(directory) / "retired_versions.pt"
            torch.save(payload, changed)
            target, target_obs, target_reward = components()
            load_contextual_checkpoint(
                changed,
                target,
                observation_builder=target_obs,
                reward_builder=target_reward,
            )

    def test_sale_checkpoint_restores_all_generations_and_output(self):
        source, obs, reward = components(sale=True, lap=True)
        source.update()
        source._hard_update_targets()
        batch = source.replay.sample(4)
        observation = (
            batch.center_local, batch.incoming_local, batch.outgoing_local,
            batch.center_rail_index, batch.incoming_rail_indices,
            batch.outgoing_rail_indices,
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
