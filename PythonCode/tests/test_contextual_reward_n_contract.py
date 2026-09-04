import unittest

import numpy as np

from oht_routing.mdp.action import REGION_B_RL
from oht_routing.mdp.reward.builder import ContextualRewardBuilder, ContextualRewardConfig
from oht_routing.mdp.reward.config import (
    RAIL_REWARD_FREE_FLOW_NEUTRAL_1_7,
    REWARD_O_CONTRACT,
    REWARD_O_PROFILE,
    REWARD_P_CONTRACT,
    REWARD_P_PROFILE,
    REWARD_Q_CONTRACT,
    REWARD_Q_PROFILE,
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


class RewardQContractTests(unittest.TestCase):
    """Characterization tests for Reward Q and historical Rewards O and P."""

    def setUp(self):
        self.config = ContextualRewardConfig.for_version(
            "Q", action_mode=REGION_B_RL
        )
        self.topology = make_topology()

    def test_locked_profile_matches_reward_q_run(self):
        config = self.config
        expected = {
            "reward_version": "Q",
            "global_alpha": 0.5,
            "local_alpha": 0.5,
            "rail_tat_weight": 660.0,
            "rail_reward_mode": RAIL_REWARD_FREE_FLOW_NEUTRAL_1_7,
            "rail_free_flow_neutral_ratio": 1.70,
            "smooth_b_rl_weight": 0.25,
            "smooth_exp_residual_weight": 0.5,
            "tat_reference": 165.0,
            "tat_weight": 4.0,
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
            "local_predicted_oht_weight": 0.01,
            "local_stop_weight": 0.30,
            "local_density_weight": 5.5,
            "local_idle_weight": 0.0,
            "local_capacity_weight": 0.0,
            "local_reward_scale": 2.0,
            "rail_tat_clip": 22.0,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(config, name), value)
        self.assertEqual(REWARD_VERSION, "Q")
        self.assertEqual(REWARD_VERSIONS, ("Q",))
        self.assertIsNot(REWARD_P_PROFILE, REWARD_O_PROFILE)
        self.assertEqual(REWARD_P_PROFILE, REWARD_O_PROFILE)
        # Reward Q keeps every global coefficient from P and differs only
        # in how per-rail credit is distributed.
        for shared in (
            "tat_weight", "backlog_weight", "backlog_growth_weight",
            "idle_reserve_weight", "global_alpha", "local_alpha",
            "local_reward_scale", "smooth_b_rl_weight", "op_weight",
        ):
            self.assertEqual(
                REWARD_Q_PROFILE[shared], REWARD_P_PROFILE[shared]
            )
        self.assertNotEqual(
            REWARD_Q_PROFILE["local_stop_weight"],
            REWARD_P_PROFILE["local_stop_weight"],
        )
        self.assertNotIn("local_density_weight", REWARD_P_PROFILE)

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
                tat_raw = -4.0 * excess / 165.0
                expected_global_raw = tat_raw - 0.0007 * 5.0 - 0.09
                self.assertAlmostEqual(batch.tat_raw, tat_raw)
                self.assertAlmostEqual(batch.global_raw, expected_global_raw)
                self.assertAlmostEqual(
                    batch.global_component, 0.5 * expected_global_raw
                )

    def test_reward_and_termination_identity(self):
        contract = reward_contract("Q")
        self.assertIs(contract, REWARD_Q_CONTRACT)
        self.assertEqual(contract.version, "Q")
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

    def test_reward_o_and_p_are_preserved_but_not_executable(self):
        self.assertEqual(REWARD_O_CONTRACT.version, "O")
        self.assertEqual(REWARD_P_CONTRACT.version, "P")
        self.assertEqual(
            REWARD_O_CONTRACT.tat_signal_mode,
            TAT_SIGNAL_RECENT_COMPLETED_300S,
        )
        self.assertEqual(REWARD_O_CONTRACT.tat_window_seconds, 300.0)
        for retired in ("O", "P"):
            with self.assertRaisesRegex(
                ValueError, "only reward_version='Q'"
            ):
                reward_contract(retired)
            with self.assertRaisesRegex(
                ValueError, "only reward_version='Q'"
            ):
                ContextualRewardConfig.for_version(retired)


if __name__ == "__main__":
    unittest.main()
