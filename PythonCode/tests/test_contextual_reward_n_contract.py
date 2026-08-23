import unittest

import numpy as np

from oht_routing.mdp.action import REGION_B_RL
from oht_routing.mdp.reward.builder import ContextualRewardBuilder, ContextualRewardConfig
from oht_routing.mdp.reward.config import (
    RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
    reward_contract,
)
from oht_routing.mdp.termination import (
    TAT_TERMINATION_REWARD_PROFILE,
    tat_termination_profile,
)
from test_contextual_observation import CONTROLLED_COUNT, make_topology
from test_contextual_reward import prime_recent_tat, reward_client


class RewardOContractTests(unittest.TestCase):
    """Characterization tests for the v4 Reward O executable baseline."""

    def setUp(self):
        self.config = ContextualRewardConfig.for_version(
            "O", action_mode=REGION_B_RL
        )
        self.topology = make_topology()

    def test_locked_profile_matches_reward_o_run(self):
        config = self.config
        expected = {
            "reward_version": "O",
            "global_alpha": 0.5,
            "local_alpha": 0.5,
            "rail_tat_weight": 30.0,
            "rail_reward_mode": RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
            "rail_free_flow_neutral_ratio": 2.0,
            "smooth_b_rl_weight": 0.25,
            "smooth_exp_residual_weight": 0.5,
            "tat_reference": 165.0,
            "tat_weight": 4.3,
            "tat_window_seconds": 300.0,
            "op_weight": 0.0,
            "use_op": False,
            "backlog_weight": 0.0007,
            "backlog_growth_enabled": True,
            "backlog_growth_horizon": 300,
            "backlog_growth_scale": 30.0,
            "backlog_growth_weight": 0.17,
            "idle_reserve_target": 200.0,
            "idle_reserve_scale": 50.0,
            "idle_reserve_weight": 0.09,
            "local_oht_weight": 0.0,
            "local_predicted_oht_weight": 0.05,
            "local_stop_weight": 0.12,
            "local_idle_weight": 0.0,
            "local_capacity_weight": 0.0,
            "local_reward_scale": 2.0,
            "rail_tat_clip": 1.0,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(config, name), value)

    def test_one_sided_unbounded_tat_curve(self):
        action = np.zeros(CONTROLLED_COUNT, dtype=np.float32)
        for total_tat, excess in (
            (159.0, 0.0),
            (160.0, 0.0),
            (170.0, 10.0),
            (200.0, 40.0),
        ):
            with self.subTest(total_tat=total_tat):
                builder = ContextualRewardBuilder(
                    self.topology, self.config
                )
                client = reward_client()
                client.TotalOhtOperationRate = 0.8
                prime_recent_tat(builder, client, total_tat)
                batch = builder.build(
                    client,
                    applied_action=action,
                    previous_applied_action=None,
                    env_step=0,
                    episode_id=0,
                )
                tat_raw = -4.3 * excess / 165.0
                expected_global_raw = tat_raw - 0.0007 * 5.0 - 0.09
                self.assertAlmostEqual(batch.tat_raw, tat_raw)
                self.assertAlmostEqual(batch.global_raw, expected_global_raw)
                self.assertAlmostEqual(
                    batch.global_component, 0.5 * expected_global_raw
                )

    def test_reward_and_termination_identity(self):
        contract = reward_contract("O")
        self.assertEqual(contract.version, "O")
        self.assertEqual(contract.tat_window_seconds, 300.0)
        self.assertEqual(
            tat_termination_profile(
                TAT_TERMINATION_REWARD_PROFILE, contract
            ),
            {
                "tat_termination_start_episode": 1,
                "tat_termination_enabled": True,
                "tat_termination_grace_steps": 10_000,
                "early_stop_tat_threshold": 200.0,
                "tat_above_threshold_patience": 300,
                "tat_termination_inclusive": True,
            },
        )
        self.assertEqual(contract.terminal_tat_penalty, -20.0)


if __name__ == "__main__":
    unittest.main()
