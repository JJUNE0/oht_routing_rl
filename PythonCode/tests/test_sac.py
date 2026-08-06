import math
import unittest

import numpy as np
import torch

from cocel_rl.algorithms import SAC
from cocel_rl.core.learner import Learner


class DummyEnv:
    obs_dim = 6
    act_dim = 2
    action_bound = [-1.0, 1.0]


def config():
    return {
        "device": "cpu",
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "adam_eps": 1e-8,
        "buffer_capacity": 256,
        "batch_size": 16,
        "gamma": 0.99,
        "tau": 0.01,
        "max_rollout": 1,
        "actor_hidden_dims": [32, 32],
        "critic_hidden_dims": [32, 32],
        "activation_fc": "relu",
        "algorithm": {
            "name": "SAC",
            "log_std_bound": [-20.0, 2.0],
            "temperature_lr": 3e-4,
            "initial_alpha": 0.2,
            "target_entropy": None,
        },
    }


class SACTests(unittest.TestCase):
    def test_policy_sampling_is_bounded_and_differentiable(self):
        learner = Learner(SAC(DummyEnv(), config()), config())
        states = torch.randn(9, DummyEnv.obs_dim)
        action, log_prob, mean = learner.actor.sample(states)
        self.assertEqual(action.shape, (9, DummyEnv.act_dim))
        self.assertEqual(log_prob.shape, (9, 1))
        self.assertEqual(mean.shape, action.shape)
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(bool((action.abs() <= 1.0).all()))
        self.assertTrue(action.requires_grad)

        batched, _ = learner.actor.get_action(
            np.zeros((5, DummyEnv.obs_dim), np.float32), eval=False
        )
        deterministic, _ = learner.actor.get_action(
            np.zeros(DummyEnv.obs_dim, np.float32), eval=True
        )
        self.assertEqual(batched.shape, (5, DummyEnv.act_dim))
        self.assertEqual(deterministic.shape, (DummyEnv.act_dim,))

    def test_learner_updates_actor_critic_target_and_temperature(self):
        cfg = config()
        learner = Learner(SAC(DummyEnv(), cfg), cfg)
        rng = np.random.default_rng(7)
        for index in range(80):
            state = rng.normal(size=DummyEnv.obs_dim).astype(np.float32)
            action = np.tanh(rng.normal(size=DummyEnv.act_dim)).astype(np.float32)
            reward = np.float32(rng.normal())
            next_state = (state + 0.1).astype(np.float32)
            learner.buffer.push(state, action, reward, next_state, index % 17 == 0)

        actor_before = [p.detach().clone() for p in learner.actor.parameters()]
        critic_before = [p.detach().clone() for p in learner.critic.parameters()]
        target_before = [p.detach().clone() for p in learner.target_critic.parameters()]
        alpha_before = float(learner.algorithm.log_alpha.detach())
        learner.algorithm.applied_action_scale = 0.25
        learner.learn()

        def changed(before, module):
            return any(
                not torch.equal(left, right.detach())
                for left, right in zip(before, module.parameters())
            )

        self.assertTrue(changed(actor_before, learner.actor))
        self.assertTrue(changed(critic_before, learner.critic))
        self.assertTrue(changed(target_before, learner.target_critic))
        self.assertNotEqual(alpha_before, float(learner.algorithm.log_alpha.detach()))
        self.assertEqual(
            set(learner.total_losses),
            {"critic", "actor", "q1", "q2", "alpha", "alpha_loss", "entropy", "log_prob"},
        )
        self.assertTrue(all(
            math.isfinite(value) for value in learner.total_losses.values()
        ))


if __name__ == "__main__":
    unittest.main()
