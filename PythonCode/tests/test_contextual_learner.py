import unittest
from dataclasses import replace

import numpy as np
import torch

from cocel_rl.algorithms.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
)
from test_contextual_observation import make_topology
from test_contextual_replay import FakeObservationBuilder, make_snapshot


SMALL_NETWORK = ContextualNetworkConfig(
    d_model=16, num_heads=4, global_emb_dim=8,
    context_dim=32, hidden_dim=64,
)


def make_replay(count=12, seed=3):
    topology = make_topology()
    builder = FakeObservationBuilder(topology)
    replay = ContextualStepReplayBuffer(
        topology, builder, capacity_env_steps=count, seed=seed
    )
    for step in range(count):
        snapshot = make_snapshot(topology, step, done=(step % 11 == 10))
        policy = np.sin(
            np.arange(len(topology.controlled_rail_ids), dtype=np.float32)
            + step
        )[:, None]
        applied = 0.1 * policy
        reward = np.tanh(
            np.arange(len(topology.controlled_rail_ids), dtype=np.float32)
            / 100 + step
        )
        replay.push(replace(
            snapshot, policy_action=policy, applied_action=applied,
            reward=reward,
        ))
    return replay


def parameters(module):
    return [value.detach().clone() for value in module.parameters()]


def changed(before, module):
    return any(
        not torch.equal(left, right.detach())
        for left, right in zip(before, module.parameters())
    )


class ContextualLearnerTests(unittest.TestCase):
    def config(self, **kwargs):
        return ContextualLearnerConfig(
            batch_size=32, target_noise=0.0,
            target_update_interval=4, policy_update_delay=2,
            minimum_replay_env_steps=1,
            minimum_action_enabled_env_steps=1,
            require_normalizer_frozen=False,
            sale_enabled=False,
            lap_enabled=False,
            **kwargs,
        )

    def test_initial_targets_identical_and_not_shared(self):
        learner = ContextualTD7Learner(
            make_replay(), network_config=SMALL_NETWORK,
            config=self.config(), seed=7,
        )
        for online, target in (
            (learner.encoder, learner.target_encoder),
            (learner.actor, learner.target_actor),
            (learner.critic, learner.target_critic),
        ):
            for a, b in zip(online.parameters(), target.parameters()):
                self.assertTrue(torch.equal(a, b))
                self.assertIsNot(a, b)

    def test_critic_encoder_actor_delay_and_hard_target_updates(self):
        learner = ContextualTD7Learner(
            make_replay(), network_config=SMALL_NETWORK,
            config=self.config(), seed=9,
        )
        encoder_before = parameters(learner.encoder)
        critic_before = parameters(learner.critic)
        actor_before = parameters(learner.actor)
        first = learner.update()
        self.assertFalse(first.actor_updated)
        self.assertFalse(changed(actor_before, learner.actor))
        self.assertTrue(changed(encoder_before, learner.encoder))
        self.assertTrue(changed(critic_before, learner.critic))
        second = learner.update()
        self.assertTrue(second.actor_updated)
        self.assertTrue(changed(actor_before, learner.actor))
        self.assertFalse(second.target_updated)
        learner.update()
        fourth = learner.update()
        self.assertTrue(fourth.target_updated)
        self.assertEqual(
            fourth.diagnostics["target/encoder_distance"], 0.0
        )

    def test_online_critic_receives_replay_applied_action(self):
        learner = ContextualTD7Learner(
            make_replay(), network_config=SMALL_NETWORK,
            config=self.config(), seed=4,
        )
        batch = learner.replay.sample(16)
        captured = []
        hook = learner.critic.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[1].detach().clone())
        )
        try:
            learner.update(batch)
        finally:
            hook.remove()
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(captured[0], batch.applied_action)
        self.assertFalse(torch.equal(captured[0], batch.policy_action))

    def test_target_critic_receives_scaled_policy_action(self):
        learner = ContextualTD7Learner(
            make_replay(), network_config=SMALL_NETWORK,
            config=self.config(), seed=11,
        )
        captured = []
        hook = learner.target_critic.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[1].detach().clone())
        )
        try:
            learner.update()
        finally:
            hook.remove()
        self.assertEqual(len(captured), 1)
        self.assertLessEqual(
            float(captured[0].abs().max()), learner.config.action_scale + 1e-7
        )

    def test_baseline_batch_critic_action_is_zero(self):
        learner = ContextualTD7Learner(
            make_replay(), network_config=SMALL_NETWORK,
            config=self.config(), seed=14,
        )
        batch = learner.replay.sample(16)
        baseline_batch = replace(
            batch, applied_action=torch.zeros_like(batch.applied_action)
        )
        self.assertGreater(float(batch.policy_action.abs().max()), 0)
        captured = []
        hook = learner.critic.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[1].detach().clone())
        )
        try:
            learner.update(baseline_batch)
        finally:
            hook.remove()
        torch.testing.assert_close(
            captured[0], torch.zeros_like(captured[0])
        )

    def test_finite_diagnostics_and_gate(self):
        replay = make_replay()
        learner = ContextualTD7Learner(
            replay, network_config=SMALL_NETWORK,
            config=self.config(), seed=2,
        )
        self.assertTrue(learner.can_learn(action_enabled_env_steps=1))
        for _ in range(8):
            result = learner.update()
            self.assertTrue(all(
                np.isfinite(value)
                for value in result.diagnostics.values()
            ))
            self.assertEqual(
                result.diagnostics["numeric/learner_finite_ratio"], 1.0
            )
        self.assertLess(result.diagnostics["critic/q_max"], 1e4)
        self.assertIn(
            "learner/encoder_joint_critic_loss", result.diagnostics
        )
        self.assertNotIn("learner/encoder_loss", result.diagnostics)

    def test_optimizer_parameter_ownership_is_disjoint(self):
        learner = ContextualTD7Learner(
            make_replay(), network_config=SMALL_NETWORK,
            config=self.config(),
        )
        groups = []
        for optimizer in (
            learner.encoder_optimizer,
            learner.actor_optimizer,
            learner.critic_optimizer,
        ):
            groups.append({
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            })
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_target_q_bounds_roll_with_hard_target_generation(self):
        learner = ContextualTD7Learner(
            make_replay(), network_config=SMALL_NETWORK,
            config=replace(
                self.config(), target_update_interval=2
            ),
            seed=21,
        )
        first = learner.update()
        self.assertFalse(first.target_updated)
        self.assertTrue(np.isfinite(learner.current_target_q_min))
        self.assertTrue(np.isfinite(learner.current_target_q_max))
        self.assertFalse(np.isfinite(learner.fixed_target_q_min))
        observed_min = learner.current_target_q_min
        observed_max = learner.current_target_q_max

        second = learner.update()
        self.assertTrue(second.target_updated)
        self.assertLessEqual(learner.fixed_target_q_min, observed_min)
        self.assertGreaterEqual(learner.fixed_target_q_max, observed_max)
        self.assertFalse(np.isfinite(learner.current_target_q_min))
        self.assertFalse(np.isfinite(learner.current_target_q_max))
        self.assertIn(
            "value/fixed_target_q_min", second.diagnostics
        )
        self.assertIn(
            "value/fixed_target_q_max", second.diagnostics
        )


if __name__ == "__main__":
    unittest.main()
