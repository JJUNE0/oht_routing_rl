import unittest

import numpy as np

from oht_routing.mdp.action import REGION_B_RL
from oht_routing.mdp.reward.builder import ContextualRewardBuilder, ContextualRewardConfig
from oht_routing.mdp.reward.config import (
    RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
    REWARD_O_CONTRACT,
    REWARD_O_PROFILE,
    REWARD_P_CONTRACT,
    REWARD_P_PROFILE,
    REWARD_VERSION,
    REWARD_VERSIONS,
    TAT_SIGNAL_CUMULATIVE_TOTAL,
    TAT_SIGNAL_RECENT_COMPLETED_300S,
    reward_contract,
)
from oht_routing.mdp.termination import (
    TAT_TERMINATION_REWARD_PROFILE,
    tat_termination_profile,
)
from test_contextual_observation import CONTROLLED_COUNT, make_topology
from test_contextual_reward import prime_recent_tat, reward_client


class RewardPContractTests(unittest.TestCase):
    """Characterization tests for Reward P and historical Reward O."""

    def setUp(self):
        self.config = ContextualRewardConfig.for_version(
            "P", action_mode=REGION_B_RL
        )
        self.topology = make_topology()

    def test_locked_profile_matches_reward_p_run(self):
        config = self.config
        expected = {
            "reward_version": "P",
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
        self.assertEqual(REWARD_VERSION, "P")
        self.assertEqual(REWARD_VERSIONS, ("P",))
        self.assertIsNot(REWARD_P_PROFILE, REWARD_O_PROFILE)
        self.assertEqual(REWARD_P_PROFILE, REWARD_O_PROFILE)

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
                client.TotalTat = total_tat
                # The recent 300 s value is still observable diagnostically,
                # but must not influence Reward P's TAT term.
                prime_recent_tat(builder, client, total_tat + 500.0)
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
        contract = reward_contract("P")
        self.assertIs(contract, REWARD_P_CONTRACT)
        self.assertEqual(contract.version, "P")
        self.assertEqual(contract.tat_signal_mode, TAT_SIGNAL_CUMULATIVE_TOTAL)
        self.assertEqual(
            contract.tat_signal_description,
            "one_sided_cumulative_total_tat_penalty",
        )
        self.assertEqual(contract.tat_window_seconds, 0.0)
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

    def test_reward_o_is_preserved_but_not_executable(self):
        self.assertEqual(REWARD_O_CONTRACT.version, "O")
        self.assertEqual(
            REWARD_O_CONTRACT.tat_signal_mode,
            TAT_SIGNAL_RECENT_COMPLETED_300S,
        )
        self.assertEqual(REWARD_O_CONTRACT.tat_window_seconds, 300.0)
        with self.assertRaisesRegex(ValueError, "only reward_version='P'"):
            reward_contract("O")
        with self.assertRaisesRegex(ValueError, "only reward_version='P'"):
            ContextualRewardConfig.for_version("O")


if __name__ == "__main__":
    unittest.main()
