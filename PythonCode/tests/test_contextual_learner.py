import unittest
from dataclasses import replace

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
)
from test_contextual_observation import make_topology
from test_contextual_replay import FakeObservationBuilder, make_snapshot


SMALL_NETWORK = ContextualNetworkConfig(
    local_physical_dim=14,
    rail_embedding_dim=8,
    num_rails=4_999,
    global_dim=5,
    critic_extra_dim=1,
    neighbor_count=15,
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
        previous_policy = np.sin(
            np.arange(len(topology.controlled_rail_ids), dtype=np.float32)
            + max(step - 1, 0)
        )[:, None]
        reward = np.tanh(
            np.arange(len(topology.controlled_rail_ids), dtype=np.float32)
            / 100 + step
        )
        replay.push(replace(
            snapshot,
            previous_applied_action=0.1 * previous_policy,
            policy_action=policy,
            applied_action=applied,
            next_previous_applied_action=applied,
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
        for online_head, target_head in (
            (learner.critic.q1, learner.target_critic.q1),
            (learner.critic.q2, learner.target_critic.q2),
        ):
            for online, target in zip(
                online_head.parameters(), target_head.parameters()
            ):
                torch.testing.assert_close(online, target, rtol=0, atol=0)
                self.assertIsNot(online, target)
        self.assertGreater(
            learner.twin_parameter_diagnostics()[
                "critic/parameter_max_abs_diff"
            ],
            0.0,
        )

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

    def test_rail_embedding_is_critic_trained_and_hard_targeted(self):
        learner = ContextualTD7Learner(
            make_replay(),
            network_config=SMALL_NETWORK,
            config=replace(self.config(), target_update_interval=2),
            seed=29,
        )
        online_before = learner.encoder.rail_embedding.weight.detach().clone()
        target_before = (
            learner.target_encoder.rail_embedding.weight.detach().clone()
        )

        first = learner.update()
        self.assertFalse(first.target_updated)
        self.assertFalse(torch.equal(
            learner.encoder.rail_embedding.weight.detach(), online_before
        ))
        torch.testing.assert_close(
            learner.target_encoder.rail_embedding.weight,
            target_before,
            rtol=0,
            atol=0,
        )

        second = learner.update()
        self.assertTrue(second.target_updated)
        torch.testing.assert_close(
            learner.target_encoder.rail_embedding.weight,
            learner.encoder.rail_embedding.weight,
            rtol=0,
            atol=0,
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

    def test_actor_and_critics_receive_aligned_previous_applied_action(self):
        learner = ContextualTD7Learner(
            make_replay(),
            network_config=SMALL_NETWORK,
            config=replace(self.config(), policy_update_delay=1),
            seed=41,
        )
        batch = learner.replay.sample(16)
        online_actor_previous = []
        online_critic_previous = []
        target_actor_previous = []

        def capture(target):
            def hook(module, args, kwargs):
                target.append(kwargs["previous_action"].detach().clone())
            return hook

        handles = (
            learner.actor.register_forward_pre_hook(
                capture(online_actor_previous), with_kwargs=True
            ),
            learner.critic.register_forward_pre_hook(
                capture(online_critic_previous), with_kwargs=True
            ),
            learner.target_actor.register_forward_pre_hook(
                capture(target_actor_previous), with_kwargs=True
            ),
        )
        try:
            learner.update(batch)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(online_actor_previous), 1)
        self.assertGreaterEqual(len(online_critic_previous), 2)
        self.assertEqual(len(target_actor_previous), 1)
        torch.testing.assert_close(
            online_actor_previous[0], batch.previous_applied_action
        )
        for value in online_critic_previous:
            torch.testing.assert_close(value, batch.previous_applied_action)
        torch.testing.assert_close(
            target_actor_previous[0], batch.next_previous_applied_action
        )

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

    def test_current_and_next_total_tat_reach_only_matching_critics(self):
        learner = ContextualTD7Learner(
            make_replay(),
            network_config=SMALL_NETWORK,
            config=replace(self.config(), policy_update_delay=1),
            seed=53,
        )
        batch = learner.replay.sample(16)
        current_tat = torch.full_like(batch.critic_total_tat, -3.25)
        next_tat = torch.full_like(batch.next_critic_total_tat, 4.75)
        routed_batch = replace(
            batch,
            critic_total_tat=current_tat,
            next_critic_total_tat=next_tat,
        )
        online_seen = []
        target_seen = []

        def capture(target):
            def hook(module, args, kwargs):
                target.append(
                    kwargs["critic_total_tat"].detach().clone()
                )
            return hook

        handles = (
            learner.critic.register_forward_pre_hook(
                capture(online_seen), with_kwargs=True
            ),
            learner.target_critic.register_forward_pre_hook(
                capture(target_seen), with_kwargs=True
            ),
        )
        try:
            result = learner.update(routed_batch)
        finally:
            for handle in handles:
                handle.remove()

        self.assertTrue(result.actor_updated)
        # Online critic supervision plus actor-loss Q1 both use current state.
        self.assertEqual(len(online_seen), 2)
        self.assertEqual(len(target_seen), 1)
        for observed in online_seen:
            torch.testing.assert_close(
                observed, current_tat, rtol=0, atol=0
            )
            self.assertFalse(torch.equal(observed, next_tat))
        torch.testing.assert_close(
            target_seen[0], next_tat, rtol=0, atol=0
        )
        self.assertFalse(torch.equal(target_seen[0], current_tat))

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
            config=replace(
                self.config(),
                sale_enabled=True,
                sale_embedding_dim=16,
                sale_feature_dim=16,
            ),
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
        self.assertIn(id(learner.encoder.rail_embedding.weight), groups[0])
        self.assertNotIn(id(learner.encoder.rail_embedding.weight), groups[1])
        self.assertNotIn(id(learner.encoder.rail_embedding.weight), groups[2])
        ownership = learner.optimizer_parameter_ownership()
        self.assertFalse(ownership["q1"] & ownership["q2"])
        self.assertEqual(
            ownership["critic_optimizer"],
            ownership["q1"]
            | ownership["q2"]
            | ownership["critic_shared"],
        )
        self.assertFalse(
            ownership["critic_optimizer"] & ownership["target_critic"]
        )
        self.assertFalse(
            ownership["critic_optimizer"] & ownership["shared_encoder"]
        )
        self.assertGreater(len(ownership["critic_shared"]), 0)
        for online_head, target_head in (
            (learner.critic.q1, learner.target_critic.q1),
            (learner.critic.q2, learner.target_critic.q2),
        ):
            for online, target in zip(
                online_head.parameters(), target_head.parameters()
            ):
                torch.testing.assert_close(online, target, rtol=0, atol=0)
                self.assertIsNot(online, target)
        target_q1 = dict(learner.target_critic.q1.named_parameters())
        target_q2 = dict(learner.target_critic.q2.named_parameters())
        self.assertGreater(max(
            float(
                (target_q1[name] - target_q2[name])
                .detach().abs().max()
            )
            for name in target_q1
        ), 0.0)

    def test_twin_gradients_outputs_and_parameters_remain_diverse(self):
        learner = ContextualTD7Learner(
            make_replay(), network_config=SMALL_NETWORK,
            config=self.config(), seed=41,
        )
        initial = learner.twin_parameter_diagnostics()
        self.assertGreater(initial["critic/parameter_l2_distance"], 0.0)
        self.assertGreater(
            initial["critic/parameter_max_abs_diff"], 0.0
        )
        result = learner.update()
        diagnostics = result.diagnostics
        for key in (
            "critic/q_abs_diff_mean",
            "critic/q_abs_diff_max",
            "critic/q1_loss",
            "critic/q2_loss",
            "critic/q1_grad_norm",
            "critic/q2_grad_norm",
            "critic/parameter_l2_distance",
            "critic/parameter_max_abs_diff",
        ):
            self.assertTrue(np.isfinite(diagnostics[key]), key)
        self.assertGreater(diagnostics["critic/q_abs_diff_mean"], 0.0)
        self.assertGreater(diagnostics["critic/q1_grad_norm"], 0.0)
        self.assertGreater(diagnostics["critic/q2_grad_norm"], 0.0)
        self.assertGreater(
            diagnostics["critic/parameter_l2_distance"], 0.0
        )

    def test_actor_last_update_diagnostics_survive_non_actor_steps(self):
        learner = ContextualTD7Learner(
            make_replay(), network_config=SMALL_NETWORK,
            config=self.config(), seed=43,
        )
        first = learner.update().diagnostics
        self.assertEqual(
            first["learner/actor_updated_this_step"], 0.0
        )
        self.assertEqual(first["learner/actor_updates_total"], 0.0)
        second = learner.update().diagnostics
        self.assertEqual(
            second["learner/actor_updated_this_step"], 1.0
        )
        self.assertEqual(second["learner/actor_updates_total"], 1.0)
        self.assertEqual(
            second["learner/actor_last_update_step"], 2.0
        )
        third = learner.update().diagnostics
        self.assertEqual(
            third["learner/actor_updated_this_step"], 0.0
        )
        for key in (
            "learner/actor_loss_last",
            "learner/actor_grad_norm_last",
            "learner/actor_last_update_step",
            "learner/actor_updates_total",
        ):
            self.assertEqual(third[key], second[key])

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
