import unittest

import torch

from oht_routing.algorithms.rl.contextual_td7.targets import (
    bellman_target,
    scale_policy_action,
    target_applied_action,
)


class ContextualTargetTests(unittest.TestCase):
    def test_policy_scale_and_raw_policy_space_noise(self):
        policy = torch.tensor([[-1.0], [0.5], [1.0]])
        scaled = scale_policy_action(policy, 0.1)
        torch.testing.assert_close(
            scaled, torch.tensor([[-0.1], [0.05], [0.1]])
        )
        result = target_applied_action(
            policy, action_scale=0.1, noise_std=0.0, noise_clip=0.02,
            noise=torch.tensor([[-1.0], [0.01], [1.0]]),
        )
        torch.testing.assert_close(
            result, torch.tensor([[-0.1], [0.051], [0.1]])
        )

    def test_done_removes_bootstrap_and_uses_twin_min(self):
        reward = torch.tensor([[2.0], [3.0]])
        done = torch.tensor([[0.0], [1.0]])
        q1 = torch.tensor([[10.0], [20.0]])
        q2 = torch.tensor([[5.0], [30.0]])
        result = bellman_target(reward, done, q1, q2, gamma=0.9)
        torch.testing.assert_close(result, torch.tensor([[6.5], [3.0]]))

    def test_nonfinite_target_fails(self):
        with self.assertRaises(FloatingPointError):
            bellman_target(
                torch.tensor([[float("nan")]]), torch.zeros(1, 1),
                torch.zeros(1, 1), torch.zeros(1, 1), gamma=0.99,
            )


if __name__ == "__main__":
    unittest.main()
