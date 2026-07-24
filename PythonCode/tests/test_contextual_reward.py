import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from contextual_reward import (
    ContextualRewardBuilder,
    ContextualRewardConfig,
    ContextualRewardError,
)
from contextual_action import EXP_RESIDUAL, REGION_B_RL
from test_contextual_observation import (
    BOUNDARY_IDS,
    CONTROLLED_COUNT,
    FakePClient,
    make_rails,
    make_topology,
)


def reward_client(reverse=False):
    client = FakePClient(make_rails(reverse=reverse))
    client.OHT_DIC = {}
    client.CompletedCommandCount = 0
    client.WaitingCommandCount = 2
    client.QueuedCommandCount = 3
    return client


class ContextualRewardTests(unittest.TestCase):
    def setUp(self):
        self.topology = make_topology()
        self.config = ContextualRewardConfig(freeze_after_env_steps=100)

    def test_shape_alignment_components_and_batch_normalizer_contract(self):
        builder = ContextualRewardBuilder(self.topology, self.config)
        client = reward_client()
        action = np.linspace(-0.2, 0.2, CONTROLLED_COUNT)
        batch = builder.build(
            client, applied_action=action, previous_applied_action=np.zeros_like(action),
            env_step=7, episode_id=2,
        )
        self.assertEqual(batch.total.shape, (CONTROLLED_COUNT,))
        self.assertTrue(np.array_equal(
            batch.controlled_rail_ids, self.topology.controlled_rail_ids
        ))
        self.assertFalse(any(x in batch.controlled_rail_ids for x in BOUNDARY_IDS))
        expected = (
            batch.global_component + batch.local_component
            - batch.rail_tat_penalty - batch.smooth_penalty
        )
        np.testing.assert_allclose(batch.total, expected)
        self.assertEqual(builder.local_normalizer.update_calls, 1)
        self.assertEqual(builder.local_normalizer.count, CONTROLLED_COUNT)
        self.assertEqual(builder.global_normalizer.update_calls, 1)
        self.assertEqual(builder.global_normalizer.count, 1)
        self.assertGreater(float(np.std(batch.local_raw)), 0)
        self.assertFalse(batch.total.flags.writeable)

    def test_order_independent_and_first_applied_action_has_zero_smooth(self):
        first = ContextualRewardBuilder(self.topology, self.config).build(
            reward_client(), applied_action=np.ones(CONTROLLED_COUNT),
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        second = ContextualRewardBuilder(self.topology, self.config).build(
            reward_client(reverse=True), applied_action=np.ones(CONTROLLED_COUNT),
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        np.testing.assert_array_equal(first.total, second.total)
        np.testing.assert_array_equal(first.smooth_penalty, 0)

    def test_nonfinite_fails_fast_and_does_not_mutate_inputs(self):
        action = np.zeros(CONTROLLED_COUNT)
        before = action.copy()
        action[4] = np.nan
        with self.assertRaises(ContextualRewardError):
            ContextualRewardBuilder(self.topology).build(
                reward_client(), applied_action=action,
                previous_applied_action=None, env_step=0, episode_id=0,
            )
        action[4] = 0
        np.testing.assert_array_equal(action, before)

    def test_save_load_normalizer_state(self):
        source = ContextualRewardBuilder(self.topology, self.config)
        source.build(
            reward_client(), applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = source.save_normalizers(Path(directory) / "reward.npz")
            restored = ContextualRewardBuilder(self.topology, self.config)
            restored.load_normalizers(path)
            np.testing.assert_array_equal(
                source.local_normalizer.mean, restored.local_normalizer.mean
            )
            self.assertEqual(source.local_normalizer.count,
                             restored.local_normalizer.count)

    def test_no_completed_job_does_not_receive_maximum_tat_reward(self):
        builder = ContextualRewardBuilder(self.topology, self.config)
        client = reward_client()
        action = np.zeros(CONTROLLED_COUNT)
        builder.build(
            client, applied_action=action, previous_applied_action=None,
            env_step=0, episode_id=0,
        )
        second = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=1, episode_id=0,
        )
        self.assertEqual(second.tat_signal_available, 0.0)
        self.assertEqual(second.marginal_tat_ema, 0.0)
        self.assertAlmostEqual(
            second.global_raw,
            -self.config.backlog_weight * (
                client.WaitingCommandCount + client.QueuedCommandCount
            ),
        )

    def test_region_smooth_penalty_uses_b_rl_delta(self):
        config = ContextualRewardConfig(
            freeze_after_env_steps=100,
            action_mode=REGION_B_RL,
            smooth_b_rl_weight=0.05,
        )
        batch = ContextualRewardBuilder(self.topology, config).build(
            reward_client(),
            applied_action=np.ones(CONTROLLED_COUNT),
            previous_applied_action=-np.ones(CONTROLLED_COUNT),
            env_step=1,
            episode_id=0,
        )
        np.testing.assert_allclose(batch.smooth_control_delta, 1.0)
        np.testing.assert_allclose(batch.smooth_penalty, 0.05)

    def test_exp_residual_smooth_penalty_preserves_residual_delta(self):
        config = ContextualRewardConfig(
            freeze_after_env_steps=100,
            action_mode=EXP_RESIDUAL,
            smooth_exp_residual_weight=0.5,
        )
        batch = ContextualRewardBuilder(self.topology, config).build(
            reward_client(),
            applied_action=np.full(CONTROLLED_COUNT, 0.05),
            previous_applied_action=np.full(CONTROLLED_COUNT, -0.05),
            env_step=1,
            episode_id=0,
        )
        np.testing.assert_allclose(batch.smooth_control_delta, 0.1)
        np.testing.assert_allclose(batch.smooth_penalty, 0.05)


if __name__ == "__main__":
    unittest.main()
