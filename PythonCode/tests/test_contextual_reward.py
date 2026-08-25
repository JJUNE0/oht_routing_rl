import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from oht_routing.mdp.action import EXP_RESIDUAL, REGION_B_RL
from oht_routing.mdp.reward.builder import (
    ContextualRewardBuilder,
    ContextualRewardConfig,
    ContextualRewardError,
)
from test_contextual_observation import (
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


def rail_tat_client(
    *,
    state=0,
    job_id=0,
    dispatched_command=0,
    passes=(),
    not_passes=(),
    completions=None,
):
    return SimpleNamespace(
        OHT_DIC={
            7: SimpleNamespace(
                ID=7,
                State=state,
                JobID=job_id,
                DispatchedCommand=dispatched_command,
                PassTimes=list(passes),
                NotPassTimes=list(not_passes),
                CmdCompleteTat=dict(completions or {}),
            )
        }
    )


def rail_pass(rail_id, state, elapsed):
    return SimpleNamespace(ID=rail_id, State=state, PassTime=elapsed)


def completion(cmd_id, cmd_tat, oht_tat=0.0):
    return SimpleNamespace(CmdID=cmd_id, CmdTat=cmd_tat, OHTTat=oht_tat)


def prime_recent_tat(builder, client, tat_s, *, command_id=101):
    oht = SimpleNamespace(
        ID=7,
        State=0,
        JobID=0,
        DispatchedCommand=0,
        StopTime=0.0,
        PassTimes=[],
        NotPassTimes=[],
        CmdCompleteTat={},
    )
    client.OHT_DIC = {7: oht}
    client.SimTime = 0.0
    builder.update_recent_completed_tat(client)
    client.SimTime = 300.0
    oht.State = 5
    oht.JobID = command_id
    oht.DispatchedCommand = command_id
    oht.CmdCompleteTat = {
        command_id: completion(command_id, tat_s)
    }
    builder.update_recent_completed_tat(client)
    client.SimTime = 301.0
    oht.State = 0
    return builder.update_recent_completed_tat(client)


class ContextualRewardTests(unittest.TestCase):
    def setUp(self):
        self.topology = make_topology()
        self.action = np.zeros(CONTROLLED_COUNT)

    def build(self, client=None, config=None, *, previous=None, step=0):
        return ContextualRewardBuilder(
            self.topology, config or ContextualRewardConfig()
        ).build(
            client or reward_client(),
            applied_action=self.action,
            previous_applied_action=previous,
            env_step=step,
            episode_id=0,
        )

    def test_total_tat_uses_one_sided_unbounded_reward_p_curve(self):
        config = ContextualRewardConfig(
            use_op=False,
            use_backlog=False,
            backlog_growth_enabled=False,
            idle_reserve_weight=0.0,
        )
        for total_tat, expected in (
            (100.0, 0.0),
            (159.0, 0.0),
            (160.0, 0.0),
            (170.0, -4.3 * 10.0 / 165.0),
            (360.0, -4.3 * 200.0 / 165.0),
        ):
            with self.subTest(total_tat=total_tat):
                client = reward_client()
                client.TotalTat = total_tat
                builder = ContextualRewardBuilder(self.topology, config)
                # Deliberately disagree with TotalTat: the recent signal is
                # diagnostic-only in Reward P.
                snapshot = prime_recent_tat(builder, client, 999.0)
                self.assertTrue(snapshot.available)
                batch = builder.build(
                    client,
                    applied_action=self.action,
                    previous_applied_action=None,
                    env_step=0,
                    episode_id=0,
                )
                self.assertAlmostEqual(batch.tat_raw, expected)
                self.assertAlmostEqual(batch.global_raw, expected)
                self.assertAlmostEqual(
                    builder.diagnostics(batch)["reward/global/tat_excess"],
                    max(0.0, total_tat - 160.0),
                )
                self.assertEqual(batch.total_tat_level, total_tat)
                self.assertEqual(batch.cumulative_total_tat_level, total_tat)
                self.assertEqual(batch.recent_completed_tat_mean, 999.0)
                self.assertEqual(
                    builder.diagnostics(batch)[
                        "reward/global/recent_completed_tat_300s_mean"
                    ],
                    999.0,
                )

    def test_completed_count_does_not_scale_total_tat_reward(self):
        config = ContextualRewardConfig(
            use_op=False,
            use_backlog=False,
            backlog_growth_enabled=False,
            idle_reserve_weight=0.0,
        )
        values = []
        for completed in (0, 1, 10_000):
            client = reward_client()
            client.CompletedCommandCount = completed
            client.TotalTat = 200.0
            builder = ContextualRewardBuilder(self.topology, config)
            values.append(builder.build(
                client,
                applied_action=self.action,
                previous_applied_action=None,
                env_step=0,
                episode_id=0,
            ).tat_raw)
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[1], values[2])

    def test_zero_total_tat_is_masked_while_recent_tat_is_diagnostic(self):
        client = reward_client()
        client.TotalTat = 0.0
        builder = ContextualRewardBuilder(self.topology)
        prime_recent_tat(builder, client, 360.0)
        batch = builder.build(
            client,
            applied_action=self.action,
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )
        self.assertEqual(batch.tat_signal_available, 0.0)
        self.assertEqual(batch.total_tat_level, 0.0)
        self.assertEqual(batch.tat_raw, 0.0)
        self.assertEqual(batch.cumulative_total_tat_level, 0.0)
        self.assertEqual(batch.recent_completed_tat_mean, 360.0)

    def test_backlog_growth_pressure_is_one_sided_and_resets(self):
        config = ContextualRewardConfig(
            use_tat=False,
            use_op=False,
            use_backlog=False,
            backlog_growth_horizon=1,
            idle_reserve_weight=0.0,
        )
        builder = ContextualRewardBuilder(self.topology, config)
        client = reward_client()
        first = builder.build(
            client, applied_action=self.action,
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        self.assertEqual(first.backlog_growth_raw, 0.0)
        client.QueuedCommandCount += 15
        half = builder.build(
            client, applied_action=self.action,
            previous_applied_action=self.action, env_step=1, episode_id=0,
        )
        self.assertEqual(half.backlog_growth_signal, 0.5)
        self.assertAlmostEqual(half.backlog_growth_raw, -0.085)
        client.QueuedCommandCount += 30
        clipped = builder.build(
            client, applied_action=self.action,
            previous_applied_action=self.action, env_step=2, episode_id=0,
        )
        self.assertEqual(clipped.backlog_growth_signal, 1.0)
        self.assertAlmostEqual(clipped.backlog_growth_raw, -0.17)
        builder.reset_episode()
        reset = builder.build(
            client, applied_action=self.action,
            previous_applied_action=self.action, env_step=0, episode_id=1,
        )
        self.assertEqual(reset.backlog_growth_signal, 0.0)

    def test_idle_reserve_pressure_is_one_sided_and_bounded(self):
        config = ContextualRewardConfig(backlog_growth_enabled=False)
        for idle, expected_signal in (
            (250, 0.0), (200, 0.0), (175, 0.5), (150, 1.0), (100, 1.0)
        ):
            with self.subTest(idle=idle):
                client = reward_client()
                client.OHT_DIC = {
                    index: SimpleNamespace(State=0, StopTime=0.0)
                    for index in range(idle)
                }
                batch = self.build(client, config)
                self.assertEqual(batch.idle_reserve_signal, expected_signal)
                self.assertAlmostEqual(
                    batch.idle_reserve_raw, -0.09 * expected_signal
                )

    def test_stop_time_is_summed_per_rail_without_clipping(self):
        client = reward_client()
        client.RAILLINE_DIC[0].OhtList = [101, 102]
        client.OHT_DIC = {
            101: SimpleNamespace(State=0, StopTime=2.0),
            102: SimpleNamespace(State=0, StopTime=100.0),
        }
        config = ContextualRewardConfig(
            use_tat=False,
            use_backlog=False,
            backlog_growth_enabled=False,
            idle_reserve_weight=0.0,
            local_predicted_oht_weight=0.0,
            rail_tat_weight=0.0,
            smooth_b_rl_weight=0.0,
        )
        batch = self.build(client, config)
        row = int(self.topology.physical_index_to_controlled_row[0])
        expected_raw = -0.12 * (2.0 + 100.0)
        self.assertAlmostEqual(batch.local_stop_raw[row], expected_raw)
        self.assertAlmostEqual(
            batch.local_component[row],
            0.5 * expected_raw / 2.0,
        )
        self.assertAlmostEqual(batch.total[row], batch.local_component[row])

    def test_local_reward_is_always_fixed_scaled_without_normalizer(self):
        builder = ContextualRewardBuilder(
            self.topology, ContextualRewardConfig()
        )
        batch = builder.build(
            reward_client(), applied_action=self.action,
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        np.testing.assert_allclose(
            batch.local_normalized,
            batch.local_raw / builder.config.local_reward_scale,
        )
        self.assertFalse(hasattr(builder, "local_normalizer"))
        self.assertFalse(hasattr(builder, "global_normalizer"))

    def test_predicted_oht_weight_scales_linearly_and_idle_term_is_zero(self):
        low = self.build(
            config=ContextualRewardConfig(local_predicted_oht_weight=0.05)
        )
        high = self.build(
            config=ContextualRewardConfig(local_predicted_oht_weight=0.10)
        )
        np.testing.assert_allclose(
            high.local_predicted_raw, 2.0 * low.local_predicted_raw
        )
        np.testing.assert_array_equal(high.local_idle_raw, 0.0)

    def test_reward_contribution_budget_reconstructs_final_scale(self):
        builder = ContextualRewardBuilder(
            self.topology, ContextualRewardConfig()
        )
        batch = builder.build(
            reward_client(), applied_action=self.action,
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        diagnostics = builder.diagnostics(batch)
        expected_tat = abs(builder.config.global_alpha * batch.tat_raw)
        self.assertAlmostEqual(
            diagnostics["reward/contribution/tat_abs"], expected_tat
        )
        self.assertAlmostEqual(
            diagnostics["reward/budget/tat_abs"], expected_tat
        )
        self.assertLess(
            diagnostics["reward/global/raw_decomposition_error"], 1e-12
        )
        self.assertLess(
            diagnostics["reward/contribution/sum_error"], 1e-12
        )
        self.assertLess(diagnostics["reward/budget/share_sum_error"], 1e-12)
        self.assertLess(
            diagnostics["reward/term_scale/reconstruction_error"], 1e-12
        )
        self.assertLess(
            diagnostics["reward/term_scale/share_sum_error"], 1e-12
        )
        self.assertEqual(
            diagnostics["reward/term_scale/op_abs_mean"], 0.0
        )
        self.assertEqual(
            diagnostics["reward/term_scale/current_oht_abs_mean"], 0.0
        )
        self.assertEqual(
            diagnostics["reward/term_scale/capacity_abs_mean"], 0.0
        )
        terminal = replace(
            batch, total=batch.total - 20.0, terminal_penalty=-20.0
        )
        terminal_diagnostics = builder.diagnostics(terminal)
        self.assertEqual(
            diagnostics["reward/budget/tat_share"],
            terminal_diagnostics["reward/budget/tat_share"],
        )

    def test_rail_weight_and_clip_are_applied_exactly_once(self):
        config = ContextualRewardConfig(rail_tat_weight=30.0, rail_tat_clip=1.0)
        builder = ContextualRewardBuilder(self.topology, config)
        raw = np.zeros(CONTROLLED_COUNT)
        raw[:4] = (-0.01, -0.10, 0.01, 0.10)
        with patch.object(builder, "_rail_reward_raw", return_value=raw):
            batch = builder.build(
                reward_client(), applied_action=self.action,
                previous_applied_action=None, env_step=0, episode_id=0,
            )
        np.testing.assert_allclose(
            batch.rail_reward_weighted_preclip[:4], (-0.3, -3.0, 0.3, 3.0)
        )
        np.testing.assert_allclose(
            batch.rail_reward_postclip[:4], (-0.3, -1.0, 0.3, 1.0)
        )

    def test_first_action_has_no_smooth_penalty(self):
        applied = np.linspace(-1.0, 1.0, CONTROLLED_COUNT)
        batch = ContextualRewardBuilder(
            self.topology, ContextualRewardConfig()
        ).build(
            reward_client(), applied_action=applied,
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        np.testing.assert_array_equal(batch.smooth_penalty, 0.0)

    def test_region_and_residual_smoothing_use_their_control_spaces(self):
        region = ContextualRewardBuilder(
            self.topology,
            ContextualRewardConfig(action_mode=REGION_B_RL),
        ).build(
            reward_client(),
            applied_action=np.ones(CONTROLLED_COUNT),
            previous_applied_action=-np.ones(CONTROLLED_COUNT),
            env_step=1, episode_id=0,
        )
        np.testing.assert_allclose(region.smooth_control_delta, 1.0)
        np.testing.assert_allclose(region.smooth_penalty, 0.25)
        residual = ContextualRewardBuilder(
            self.topology,
            ContextualRewardConfig(action_mode=EXP_RESIDUAL),
        ).build(
            reward_client(),
            applied_action=np.full(CONTROLLED_COUNT, 0.05),
            previous_applied_action=np.full(CONTROLLED_COUNT, -0.05),
            env_step=1, episode_id=0,
        )
        np.testing.assert_allclose(residual.smooth_control_delta, 0.1)
        np.testing.assert_allclose(residual.smooth_penalty, 0.05)

    def test_order_independent_and_inputs_are_not_mutated(self):
        action = np.linspace(-1.0, 1.0, CONTROLLED_COUNT)
        original = action.copy()
        forward = ContextualRewardBuilder(
            self.topology, ContextualRewardConfig()
        ).build(
            reward_client(False), applied_action=action,
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        reverse = ContextualRewardBuilder(
            self.topology, ContextualRewardConfig()
        ).build(
            reward_client(True), applied_action=action,
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        np.testing.assert_allclose(forward.total, reverse.total)
        np.testing.assert_array_equal(action, original)

    def test_nonfinite_inputs_fail_fast(self):
        for total_tat in (np.nan, np.inf, -np.inf):
            with self.subTest(total_tat=total_tat):
                client = reward_client()
                client.TotalTat = total_tat
                with self.assertRaisesRegex(
                    ContextualRewardError, "NaN or Inf"
                ):
                    self.build(client)
        client = reward_client()
        client.TotalTat = -1.0
        with self.assertRaisesRegex(
            ContextualRewardError, "TotalTat must be non-negative"
        ):
            self.build(client)
        bad_action = self.action.copy()
        bad_action[0] = np.inf
        with self.assertRaises(ContextualRewardError):
            ContextualRewardBuilder(
                self.topology, ContextualRewardConfig()
            ).build(
                reward_client(), applied_action=bad_action,
                previous_applied_action=None, env_step=0, episode_id=0,
            )


if __name__ == "__main__":
    unittest.main()
