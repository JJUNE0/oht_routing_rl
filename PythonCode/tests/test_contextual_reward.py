import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import contextual_reward as reward_runtime
import contextual_reward_version_cfg as reward_version_cfg
from contextual_reward import (
    ContextualRewardBuilder,
    ContextualRewardConfig as _ContextualRewardConfig,
    ContextualRewardError,
    OHTCycleTracker,
)
from contextual_reward_version_cfg import (
    RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
    RAIL_REWARD_FIXED_TAT_REFERENCE,
    RAIL_TAT_FIXED_REFERENCE,
    RAIL_TAT_FREE_FLOW_RATIO,
    REWARD_CONTRACT_VERSION,
    REWARD_NORMALIZATION_VERSION,
    REWARD_TAT_VERSION,
    REWARD_VERSION,
    REWARD_VERSIONS,
    TAT_SIGNAL_COMPLETION_EVENT,
    TAT_SIGNAL_MARGINAL_TAT_EMA,
    TAT_SIGNAL_TOTAL_TAT_LEVEL,
    reward_contract,
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


def ContextualRewardConfig(*args, **kwargs):
    """Legacy-mode fixture unless a test explicitly selects Phase 1."""
    kwargs.setdefault("rail_reward_mode", RAIL_REWARD_FIXED_TAT_REFERENCE)
    kwargs.setdefault("tat_reference", 174.4236)
    kwargs.setdefault("tat_weight", 9.2)
    kwargs.setdefault("op_weight", 5.0)
    kwargs.setdefault("backlog_weight", 0.002)
    kwargs.setdefault("local_predicted_oht_weight", 0.2)
    kwargs.setdefault("local_reward_scale", 3.0)
    kwargs.setdefault("local_fixed_scale_enabled", True)
    kwargs.setdefault("global_normalization_enabled", False)
    kwargs.setdefault("local_normalization_enabled", False)
    kwargs.setdefault("local_idle_weight", 0.0)
    kwargs.setdefault("rail_tat_weight", 1.0)
    kwargs.setdefault("rail_tat_clip", 1.0)
    kwargs.setdefault("backlog_growth_enabled", False)
    kwargs.setdefault("idle_reserve_weight", 0.0)
    kwargs.setdefault("tat_one_sided", True)
    kwargs.setdefault("tat_signal_mode", TAT_SIGNAL_TOTAL_TAT_LEVEL)
    kwargs.setdefault("use_op", True)
    return _ContextualRewardConfig(*args, **kwargs)


class ContextualRewardTests(unittest.TestCase):
    def setUp(self):
        self.topology = make_topology()
        self.config = ContextualRewardConfig(freeze_after_env_steps=100)

    def _run_reward_u_completion(
        self,
        cmd_tats,
        *,
        total_tat=165.0,
        oht_tats=None,
        command_ids=None,
        builder=None,
        episode_id=0,
    ):
        cmd_tats = list(cmd_tats)
        oht_tats = list(
            oht_tats or [1_000.0 + index for index in range(len(cmd_tats))]
        )
        command_ids = list(command_ids or range(1, len(cmd_tats) + 1))
        builder = builder or ContextualRewardBuilder(
            self.topology, _ContextualRewardConfig()
        )
        client = reward_client()
        client.TotalTat = float(total_tat)
        client.JOB_DIC = {
            command_id: SimpleNamespace(ID=command_id)
            for command_id in command_ids
        }
        client.OHT_DIC = {
            100 + index: SimpleNamespace(
                ID=100 + index,
                State=0,
                JobID=command_id,
                DispatchedCommand=command_id,
                PassTimes=[],
                NotPassTimes=[],
                CmdCompleteTat={},
                StopTime=0.0,
            )
            for index, command_id in enumerate(command_ids)
        }
        action = np.zeros(CONTROLLED_COUNT)
        builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=0, episode_id=episode_id,
        )
        for oht in client.OHT_DIC.values():
            oht.State = 2
            oht.CmdCompleteTat = {
                oht.JobID: completion(oht.JobID, 1.0, 1.0)
            }
        builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=1, episode_id=episode_id,
        )
        for oht, cmd_tat, oht_tat in zip(
            client.OHT_DIC.values(), cmd_tats, oht_tats
        ):
            oht.State = 5
            oht.CmdCompleteTat = {
                oht.JobID: completion(oht.JobID, cmd_tat, oht_tat)
            }
        builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=2, episode_id=episode_id,
        )
        client.JOB_DIC = {}
        for oht in client.OHT_DIC.values():
            oht.State = 0
            oht.JobID = 0
            oht.DispatchedCommand = 0
            oht.CmdCompleteTat = {}
        final = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=3, episode_id=episode_id,
        )
        return builder, client, final

    def test_historical_e_to_u_profiles_are_complete_and_unique(self):
        self.assertIs(
            reward_runtime.REWARD_CONTRACTS,
            reward_version_cfg.REWARD_CONTRACTS,
        )
        self.assertIs(
            reward_runtime.REWARD_PROFILE_CONFIGS,
            reward_version_cfg.REWARD_PROFILE_CONFIGS,
        )
        self.assertEqual(
            REWARD_VERSIONS,
            (
                "E", "F_RAMP", "F_NO_RAMP", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R",
                "S_REBALANCE", "S_EHYBRID", "T", "U",
            ),
        )
        profiles = {
            key: _ContextualRewardConfig.for_version(key)
            for key in REWARD_VERSIONS
        }
        for key, profile in profiles.items():
            with self.subTest(key=key):
                self.assertEqual(profile.reward_version, key)
                self.assertEqual(profile.contract.version, key)

        for key in ("E", "F_RAMP", "F_NO_RAMP", "G"):
            self.assertEqual(
                profiles[key].tat_signal_mode,
                TAT_SIGNAL_MARGINAL_TAT_EMA,
            )
        self.assertEqual(
            profiles["U"].tat_signal_mode,
            TAT_SIGNAL_COMPLETION_EVENT,
        )
        self.assertTrue(profiles["F_RAMP"].tat_confidence_ramp)
        self.assertFalse(profiles["F_NO_RAMP"].tat_confidence_ramp)
        self.assertEqual(reward_contract("F").version, "F_RAMP")
        self.assertEqual(reward_contract("S").version, "S_EHYBRID")
        self.assertNotEqual(
            reward_contract("S_REBALANCE").contract_version,
            reward_contract("S_EHYBRID").contract_version,
        )

        for key in ("E", "F_RAMP", "F_NO_RAMP", "G", "H"):
            self.assertTrue(profiles[key].global_normalization_enabled)
            self.assertTrue(profiles[key].local_normalization_enabled)
            self.assertEqual(profiles[key].freeze_after_env_steps, 10_000)
        for key in (
            "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
            "S_REBALANCE",
        ):
            self.assertFalse(profiles[key].global_normalization_enabled)
            self.assertFalse(profiles[key].local_normalization_enabled)
            self.assertTrue(profiles[key].local_fixed_scale_enabled)
        self.assertTrue(profiles["S_EHYBRID"].global_normalization_enabled)
        self.assertTrue(profiles["S_EHYBRID"].local_normalization_enabled)
        for key in ("T", "U"):
            self.assertFalse(profiles[key].global_normalization_enabled)
            self.assertTrue(profiles[key].local_normalization_enabled)

    def test_historical_marginal_ramp_and_one_sided_tat_profiles(self):
        action = np.zeros(CONTROLLED_COUNT)

        e_builder = ContextualRewardBuilder(
            self.topology, _ContextualRewardConfig.for_version("E")
        )
        client = reward_client()
        client.TotalTat = 150.0
        client.TotalOhtOperationRate = 0.75
        client.CompletedCommandCount = 1
        first_e = e_builder.build(
            client,
            applied_action=action,
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )
        self.assertEqual(first_e.global_raw, 0.0)

        expected_tat = 9.2 * (174.4236 - 150.0) / 174.4236
        expected_backlog = -0.01 * 5.0
        for key, confidence in (
            ("F_RAMP", 1.0 / 501.0),
            ("F_NO_RAMP", 1.0),
        ):
            builder = ContextualRewardBuilder(
                self.topology, _ContextualRewardConfig.for_version(key)
            )
            sample = reward_client()
            sample.TotalTat = 150.0
            sample.TotalOhtOperationRate = 0.75
            sample.CompletedCommandCount = 1
            batch = builder.build(
                sample,
                applied_action=action,
                previous_applied_action=None,
                env_step=0,
                episode_id=0,
            )
            self.assertAlmostEqual(batch.tat_confidence, confidence)
            self.assertAlmostEqual(
                batch.global_raw,
                expected_tat * confidence + expected_backlog,
            )

        for key, expected in (
            ("M", -11.0 * 10.0 / 165.0),
            ("N", -11.0 * 40.0 / 165.0),
        ):
            builder = ContextualRewardBuilder(
                self.topology, _ContextualRewardConfig.for_version(key)
            )
            sample = reward_client()
            sample.TotalTat = 200.0
            sample.TotalOhtOperationRate = 0.8
            batch = builder.build(
                sample,
                applied_action=action,
                previous_applied_action=None,
                env_step=0,
                episode_id=0,
            )
            self.assertAlmostEqual(batch.tat_raw_preclip, expected)

    def test_reward_u_defaults_match_completion_global_raw_contract(self):
        config = _ContextualRewardConfig()
        self.assertEqual(REWARD_VERSION, "U")
        self.assertEqual(
            REWARD_CONTRACT_VERSION,
            "contextual_controlled_reward_v23_completion_tat_global_raw_local_running_norm",
        )
        self.assertEqual(
            REWARD_TAT_VERSION,
            "actual_new_completion_cmd_tat_ref165_v1",
        )
        self.assertEqual(
            REWARD_NORMALIZATION_VERSION,
            "reward_u_global_raw_local_running_normalizer_v1",
        )
        self.assertEqual(config.rail_reward_mode, RAIL_REWARD_FIXED_TAT_REFERENCE)
        self.assertEqual(config.rail_free_flow_neutral_ratio, 2.0)
        for invalid in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _ContextualRewardConfig(
                    rail_free_flow_neutral_ratio=invalid
                )
        self.assertEqual(config.tat_reference, 165.0)
        self.assertEqual(config.tat_weight, 2.3)
        self.assertFalse(config.tat_one_sided)
        self.assertEqual(config.op_weight, 0.0)
        self.assertFalse(config.use_op)
        self.assertEqual(config.backlog_weight, 0.0025)
        self.assertFalse(config.backlog_growth_enabled)
        self.assertEqual(config.backlog_growth_horizon, 300)
        self.assertEqual(config.backlog_growth_scale, 30.0)
        self.assertEqual(config.backlog_growth_weight, 0.0)
        self.assertEqual(config.idle_reserve_target, 200.0)
        self.assertEqual(config.idle_reserve_scale, 50.0)
        self.assertEqual(config.idle_reserve_weight, 0.0)
        self.assertEqual(config.local_oht_weight, 0.3)
        self.assertEqual(config.local_predicted_oht_weight, 0.2)
        self.assertEqual(config.local_stop_weight, 0.3)
        self.assertEqual(config.local_idle_weight, 0.1)
        self.assertEqual(config.local_capacity_weight, 0.1)
        self.assertFalse(config.global_normalization_enabled)
        self.assertTrue(config.local_normalization_enabled)
        self.assertFalse(config.local_fixed_scale_enabled)
        self.assertEqual(config.freeze_after_env_steps, 30_000)
        self.assertEqual(config.smooth_b_rl_weight, 0.05)
        self.assertEqual(config.rail_tat_weight, 1.0)
        self.assertIsNone(config.rail_tat_clip)
        self.assertIsNone(config.tat_raw_clip)
        for field in ("backlog_weight", "local_predicted_oht_weight"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                _ContextualRewardConfig(**{field: -0.01})

    def test_named_t_u_profiles_cover_the_full_reward_and_normalizer_contract(self):
        reward_t = _ContextualRewardConfig.for_version("T")
        reward_u = _ContextualRewardConfig.for_version("U")
        self.assertEqual(reward_t.reward_version, "T")
        self.assertEqual(reward_u.reward_version, "U")
        self.assertEqual(
            _ContextualRewardConfig(reward_version="T").tat_signal_mode,
            TAT_SIGNAL_TOTAL_TAT_LEVEL,
        )
        self.assertEqual(reward_t.tat_signal_mode, TAT_SIGNAL_TOTAL_TAT_LEVEL)
        self.assertEqual(reward_u.tat_signal_mode, TAT_SIGNAL_COMPLETION_EVENT)
        self.assertEqual(
            reward_t.contract.contract_version,
            "contextual_controlled_reward_v22_global_raw_local_running_norm",
        )
        self.assertEqual(
            reward_u.contract.contract_version,
            REWARD_CONTRACT_VERSION,
        )
        t_fields = asdict(reward_t)
        u_fields = asdict(reward_u)
        for name in ("reward_version", "tat_signal_mode"):
            t_fields.pop(name)
            u_fields.pop(name)
        self.assertEqual(t_fields, u_fields)
        self.assertFalse(reward_t.global_normalization_enabled)
        self.assertTrue(reward_t.local_normalization_enabled)
        self.assertEqual(reward_t.freeze_after_env_steps, 30_000)

    def test_named_t_and_u_profiles_select_total_level_vs_completion_event(self):
        action = np.zeros(CONTROLLED_COUNT)
        client = reward_client()
        client.TotalTat = 150.0
        client.JOB_DIC = {}

        reward_t_builder = ContextualRewardBuilder(
            self.topology, _ContextualRewardConfig.for_version("T")
        )
        reward_u_builder = ContextualRewardBuilder(
            self.topology, _ContextualRewardConfig.for_version("U")
        )
        reward_t = reward_t_builder.build(
            client,
            applied_action=action,
            previous_applied_action=action,
            env_step=0,
            episode_id=0,
        )
        reward_u = reward_u_builder.build(
            client,
            applied_action=action,
            previous_applied_action=action,
            env_step=0,
            episode_id=0,
        )
        expected_tat = 2.3 * (165.0 - 150.0) / 165.0
        expected_backlog = -0.0025 * 5.0
        self.assertAlmostEqual(reward_t.global_raw, expected_tat + expected_backlog)
        self.assertAlmostEqual(reward_u.global_raw, expected_backlog)
        self.assertEqual(reward_t.completion_count, 0)
        self.assertEqual(reward_u.completion_count, 0)
        for builder in (reward_t_builder, reward_u_builder):
            self.assertEqual(builder.global_normalizer.count, 0)
            self.assertEqual(builder.local_normalizer.count, CONTROLLED_COUNT)

    def test_reward_u_single_and_multiple_completion_mean(self):
        single_builder, _, single = self._run_reward_u_completion([150.0])
        self.assertAlmostEqual(single.completion_tat_raw, (165.0 - 150.0) / 165.0)
        self.assertAlmostEqual(
            single.completion_tat_weighted_raw,
            2.3 * (165.0 - 150.0) / 165.0,
        )
        self.assertAlmostEqual(
            single.global_raw,
            single.completion_tat_weighted_raw - 0.0025 * 5.0,
        )
        self.assertAlmostEqual(single.global_component, 0.5 * single.global_raw)
        self.assertEqual(single.completion_count, 1)
        self.assertEqual(single.completion_valid_count, 1)
        single_diagnostics = single_builder.diagnostics(single)
        self.assertEqual(single_diagnostics["reward/completion/tat_mean"], 150.0)
        self.assertEqual(single_diagnostics["reward/completion/tat_std"], 0.0)
        self.assertEqual(single_diagnostics["reward/completion/tat_min"], 150.0)
        self.assertEqual(single_diagnostics["reward/completion/tat_max"], 150.0)
        self.assertAlmostEqual(
            single_diagnostics["reward/contribution/completion_tat_abs"],
            abs(0.5 * single.completion_tat_weighted_raw),
        )

        _, _, multiple = self._run_reward_u_completion([150.0, 165.0, 180.0])
        expected = np.mean([
            (165.0 - 150.0) / 165.0,
            (165.0 - 165.0) / 165.0,
            (165.0 - 180.0) / 165.0,
        ])
        self.assertAlmostEqual(multiple.completion_tat_raw, expected)
        self.assertEqual(multiple.completion_count, 3)
        self.assertEqual(multiple.completion_valid_count, 3)

    def test_reward_u_no_completion_and_duplicate_suppression(self):
        builder, client, completed = self._run_reward_u_completion([150.0])
        self.assertNotEqual(completed.completion_tat_weighted_raw, 0.0)
        action = np.zeros(CONTROLLED_COUNT)
        duplicate_tick = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=4, episode_id=0,
        )
        self.assertEqual(duplicate_tick.completion_count, 0)
        self.assertEqual(duplicate_tick.completion_tat_raw, 0.0)
        self.assertEqual(duplicate_tick.completion_tat_weighted_raw, 0.0)

    def test_reward_u_completion_is_independent_of_total_tat(self):
        observed = []
        for total_tat in (100.0, 165.0, 300.0):
            _, _, batch = self._run_reward_u_completion(
                [150.0], total_tat=total_tat
            )
            observed.append(batch.completion_tat_weighted_raw)
        self.assertEqual(observed, [observed[0]] * len(observed))

    def test_reward_u_uses_cmd_tat_not_oht_tat(self):
        _, _, batch = self._run_reward_u_completion(
            [150.0], oht_tats=[9_999.0]
        )
        self.assertAlmostEqual(batch.completion_tat_raw, (165.0 - 150.0) / 165.0)
        self.assertEqual(batch.completion_tat_mean, 150.0)

    def test_reward_u_episode_reset_clears_dedup_not_local_normalizer(self):
        builder, _, first = self._run_reward_u_completion(
            [150.0], command_ids=[123]
        )
        self.assertEqual(first.completion_valid_count, 1)
        normalizer_count = builder.local_normalizer.count
        builder.reset_episode()
        self.assertEqual(builder._seen_completed_command_ids, set())
        self.assertIsNone(builder._trace_command_id)
        self.assertFalse(builder._trace_command_finished)
        self.assertEqual(builder.local_normalizer.count, normalizer_count)
        _, _, second = self._run_reward_u_completion(
            [150.0], command_ids=[123], builder=builder, episode_id=1
        )
        self.assertEqual(second.completion_valid_count, 1)
        second_trace = builder.diagnostics(second)
        self.assertEqual(second_trace["trace/command/id"], 123.0)
        self.assertEqual(second_trace["trace/command/completed"], 1.0)

    def test_reward_u_allows_reused_id_only_after_new_job_lifetime(self):
        builder, _, first = self._run_reward_u_completion(
            [150.0], command_ids=[123]
        )
        self.assertEqual(first.completion_valid_count, 1)
        _, _, reused = self._run_reward_u_completion(
            [180.0], command_ids=[123], builder=builder
        )
        self.assertEqual(reused.completion_valid_count, 1)
        diagnostics = builder.diagnostics(reused)
        self.assertEqual(
            diagnostics["reward/completion/command_id_reuse_count"], 1.0
        )

    def test_reward_u_traces_one_command_across_ohts_until_completion(self):
        builder = ContextualRewardBuilder(
            self.topology, _ContextualRewardConfig()
        )
        client = reward_client()
        client.JOB_DIC = {10: SimpleNamespace(ID=10)}
        client.OHT_DIC = {
            7: SimpleNamespace(
                ID=7, State=2, JobID=10, DispatchedCommand=10,
                PassTimes=[], NotPassTimes=[], StopTime=0.0,
                CmdCompleteTat={10: completion(10, 10.0, 3.0)},
            )
        }
        action = np.zeros(CONTROLLED_COUNT)

        first = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=0, episode_id=0,
        )
        first_trace = builder.diagnostics(first)
        self.assertEqual(first_trace["trace/command/id"], 10.0)
        self.assertEqual(first_trace["trace/command/lifetime"], 1.0)
        self.assertEqual(first_trace["trace/command/cmd_tat"], 10.0)
        self.assertEqual(first_trace["trace/command/oht_tat"], 3.0)
        self.assertEqual(first_trace["trace/command/oht_id"], 7.0)
        self.assertEqual(first_trace["trace/command/available"], 1.0)
        self.assertEqual(first_trace["trace/command/completed"], 0.0)

        client.OHT_DIC[7].State = 4
        client.OHT_DIC[7].CmdCompleteTat = {
            10: completion(10, 20.0, 8.0)
        }
        second = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=1, episode_id=0,
        )
        second_trace = builder.diagnostics(second)
        self.assertEqual(second_trace["trace/command/id"], 10.0)
        self.assertEqual(second_trace["trace/command/cmd_tat"], 20.0)
        self.assertEqual(second_trace["trace/command/oht_tat"], 8.0)

        client.OHT_DIC[7].State = 0
        client.OHT_DIC[7].JobID = 0
        client.OHT_DIC[7].DispatchedCommand = 0
        client.OHT_DIC[7].CmdCompleteTat = {}
        client.OHT_DIC[8] = SimpleNamespace(
            ID=8, State=4, JobID=10, DispatchedCommand=10,
            PassTimes=[], NotPassTimes=[], StopTime=0.0,
            CmdCompleteTat={10: completion(10, 30.0, 12.0)},
        )
        reassigned = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=2, episode_id=0,
        )
        reassigned_trace = builder.diagnostics(reassigned)
        self.assertEqual(reassigned_trace["trace/command/id"], 10.0)
        self.assertEqual(reassigned_trace["trace/command/oht_id"], 8.0)
        self.assertEqual(reassigned_trace["trace/command/cmd_tat"], 30.0)
        self.assertEqual(reassigned_trace["trace/command/oht_tat"], 12.0)

        client.OHT_DIC[8].State = 5
        client.OHT_DIC[8].CmdCompleteTat = {
            10: completion(10, 40.0, 20.0)
        }
        builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=3, episode_id=0,
        )
        client.JOB_DIC = {}
        client.OHT_DIC[8].State = 0
        client.OHT_DIC[8].JobID = 0
        client.OHT_DIC[8].DispatchedCommand = 0
        client.OHT_DIC[8].CmdCompleteTat = {}
        completed = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=4, episode_id=0,
        )
        completed_trace = builder.diagnostics(completed)
        self.assertEqual(completed_trace["trace/command/id"], 10.0)
        self.assertEqual(completed_trace["trace/command/available"], 0.0)
        self.assertEqual(completed_trace["trace/command/cmd_tat"], 40.0)
        self.assertEqual(completed_trace["trace/command/oht_tat"], 20.0)
        self.assertEqual(completed_trace["trace/command/completed"], 1.0)

        after_completion = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=5, episode_id=0,
        )
        self.assertFalse(any(
            key.startswith("trace/command/")
            for key in builder.diagnostics(after_completion)
        ))

    def test_reward_u_malformed_completion_is_excluded_and_finite(self):
        for malformed in (np.nan, "bad", -1.0, 0.0):
            with self.subTest(malformed=malformed):
                _, _, batch = self._run_reward_u_completion([malformed])
                self.assertEqual(batch.completion_count, 1)
                self.assertEqual(batch.completion_valid_count, 0)
                self.assertEqual(batch.completion_invalid_count, 1)
                self.assertEqual(batch.completion_tat_raw, 0.0)
                self.assertTrue(np.isfinite(batch.total).all())

    def test_reward_u_local_raw_preserves_all_reward_e_terms(self):
        client = reward_client()
        rail_id = int(self.topology.controlled_rail_ids[0])
        rail = client.RAILLINE_DIC[rail_id]
        rail.OhtList = [101, 102]
        rail.PredictedOHTCount = 3
        rail.IdleOHTCount = 4
        rail.PortCount = 1
        client.OHT_DIC = {
            101: SimpleNamespace(StopTime=4.0, State=0),
            102: SimpleNamespace(StopTime=6.0, State=0),
        }
        builder = ContextualRewardBuilder(
            self.topology, _ContextualRewardConfig()
        )
        local_raw = builder._local_raw(client)
        expected = -(
            0.3 * 2
            + 0.2 * 3
            + 0.3 * 5.0
            + 0.1 * 4
            + 0.1 * (2 / (1 + 1))
        )
        self.assertAlmostEqual(local_raw[0], expected)

    def test_reward_u_only_local_normalizes_before_update_and_freezes(self):
        config = _ContextualRewardConfig(freeze_after_env_steps=2)
        builder = ContextualRewardBuilder(self.topology, config)
        action = np.zeros(CONTROLLED_COUNT)
        first_client = reward_client()
        first = builder.build(
            first_client, applied_action=action, previous_applied_action=action,
            env_step=0, episode_id=0,
        )
        self.assertAlmostEqual(first.global_normalized, first.global_raw, places=6)
        self.assertEqual(builder.global_normalizer.count, 0)
        self.assertEqual(builder.local_normalizer.count, CONTROLLED_COUNT)

        counts_before_reset = (
            builder.global_normalizer.count, builder.local_normalizer.count
        )
        builder.reset_episode()
        self.assertEqual(
            counts_before_reset,
            (builder.global_normalizer.count, builder.local_normalizer.count),
        )

        second_client = reward_client()
        second_client.TotalTat = 180.0
        second_global_raw = -0.0025 * 5
        second_local_raw = builder._local_raw(second_client)
        expected_local = builder.local_normalizer.normalize(
            second_local_raw[:, None], name="expected_local"
        )[:, 0]
        second = builder.build(
            second_client, applied_action=action, previous_applied_action=action,
            env_step=1, episode_id=1,
        )
        self.assertAlmostEqual(second.global_normalized, second_global_raw)
        np.testing.assert_allclose(second.local_normalized, expected_local)
        self.assertFalse(builder.global_normalizer.frozen)
        self.assertTrue(builder.local_normalizer.frozen)
        frozen = (
            builder.global_normalizer.mean.copy(),
            builder.local_normalizer.mean.copy(),
            builder.global_normalizer.count,
            builder.local_normalizer.count,
        )
        second_client.TotalTat = 120.0
        builder.build(
            second_client, applied_action=action, previous_applied_action=action,
            env_step=2, episode_id=1,
        )
        np.testing.assert_array_equal(builder.global_normalizer.mean, frozen[0])
        np.testing.assert_array_equal(builder.local_normalizer.mean, frozen[1])
        self.assertEqual(builder.global_normalizer.count, frozen[2])
        self.assertEqual(builder.local_normalizer.count, frozen[3])

    def test_total_tat_change_does_not_change_reward_u_terms(self):
        action = np.linspace(-0.5, 0.5, CONTROLLED_COUNT)
        previous = np.zeros(CONTROLLED_COUNT)
        batches = []
        for total_tat in (150.0, 180.0):
            client = reward_client()
            client.TotalTat = total_tat
            batches.append(ContextualRewardBuilder(
                self.topology, _ContextualRewardConfig()
            ).build(
                client,
                applied_action=action,
                previous_applied_action=previous,
                env_step=0,
                episode_id=0,
            ))
        low, high = batches
        for field in (
            "op_raw", "backlog_raw", "backlog_growth_raw",
            "idle_reserve_raw",
        ):
            self.assertEqual(getattr(low, field), getattr(high, field))
        for field in (
            "local_raw", "local_component", "rail_reward_raw",
            "rail_reward_postclip", "smooth_penalty",
        ):
            np.testing.assert_array_equal(
                getattr(low, field), getattr(high, field)
            )
        self.assertAlmostEqual(
            high.global_raw - low.global_raw,
            high.tat_raw_ramped - low.tat_raw_ramped,
        )
        self.assertEqual(low.tat_raw_ramped, 0.0)
        self.assertEqual(high.tat_raw_ramped, 0.0)

    def test_phase2_predicted_oht_weight_scales_linearly_and_idle_stays_zero(self):
        client = reward_client()
        low = ContextualRewardBuilder(
            self.topology,
            ContextualRewardConfig(local_predicted_oht_weight=0.05),
        ).build(
            client, applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        high = ContextualRewardBuilder(
            self.topology,
            ContextualRewardConfig(local_predicted_oht_weight=0.10),
        ).build(
            client, applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        np.testing.assert_allclose(
            high.local_predicted_raw, 2.0 * low.local_predicted_raw
        )
        np.testing.assert_array_equal(high.local_idle_raw, 0.0)

    def test_backlog_growth_pressure_contract_and_reset(self):
        config = _ContextualRewardConfig(
            backlog_growth_enabled=True,
            backlog_growth_horizon=1,
            backlog_growth_scale=30.0,
            backlog_growth_weight=0.16,
            idle_reserve_weight=0.0,
            use_tat=False,
            use_op=False,
            use_backlog=False,
            global_normalization_enabled=False,
            local_normalization_enabled=False,
        )
        builder = ContextualRewardBuilder(self.topology, config)
        client = reward_client()
        action = np.zeros(CONTROLLED_COUNT)

        first = builder.build(
            client, applied_action=action, previous_applied_action=None,
            env_step=0, episode_id=0,
        )
        self.assertEqual(first.backlog_growth_raw, 0.0)
        client.QueuedCommandCount -= 1
        decreasing = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=1, episode_id=0,
        )
        self.assertEqual(decreasing.backlog_growth_raw, 0.0)
        unchanged = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=2, episode_id=0,
        )
        self.assertEqual(unchanged.backlog_growth_raw, 0.0)
        client.QueuedCommandCount += 15
        half = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=3, episode_id=0,
        )
        self.assertEqual(half.backlog_growth_signal, 0.5)
        self.assertAlmostEqual(half.backlog_growth_raw, -0.08)
        self.assertAlmostEqual(half.global_raw, -0.08)
        self.assertAlmostEqual(half.global_component, -0.04)
        client.QueuedCommandCount += 30
        clipped = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=4, episode_id=0,
        )
        self.assertEqual(clipped.backlog_growth_signal, 1.0)
        self.assertAlmostEqual(clipped.backlog_growth_raw, -0.16)
        self.assertAlmostEqual(clipped.global_raw, -0.16)
        self.assertAlmostEqual(clipped.global_component, -0.08)

        builder.reset_episode()
        after_reset = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=0, episode_id=1,
        )
        self.assertEqual(after_reset.backlog_growth_signal, 0.0)
        for invalid in (np.nan, np.inf):
            bad_client = reward_client()
            bad_client.QueuedCommandCount = invalid
            with self.subTest(invalid=invalid), self.assertRaises(
                ContextualRewardError
            ):
                ContextualRewardBuilder(
                    self.topology, config
                ).build(
                    bad_client,
                    applied_action=action,
                    previous_applied_action=None,
                    env_step=0,
                    episode_id=0,
                )

    def test_idle_reserve_pressure_is_disabled_in_reward_q(self):
        config = _ContextualRewardConfig(backlog_growth_enabled=False)
        action = np.zeros(CONTROLLED_COUNT)
        for idle, expected_signal in (
            (250, 0.0), (200, 0.0), (175, 0.5), (150, 1.0), (100, 1.0)
        ):
            with self.subTest(idle=idle):
                client = reward_client()
                client.OHT_DIC = {
                    100_000 + index: SimpleNamespace(State=0)
                    for index in range(idle)
                }
                batch = ContextualRewardBuilder(
                    self.topology, config
                ).build(
                    client,
                    applied_action=action,
                    previous_applied_action=None,
                    env_step=0,
                    episode_id=0,
                )
                self.assertEqual(batch.idle_reserve_signal, expected_signal)
                self.assertAlmostEqual(
                    batch.idle_reserve_raw,
                    0.0,
                )

    def test_contribution_budget_uses_final_reward_scale_and_excludes_terminal(self):
        builder = ContextualRewardBuilder(
            self.topology, _ContextualRewardConfig()
        )
        action = np.zeros(CONTROLLED_COUNT)
        batch = builder.build(
            reward_client(), applied_action=action,
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        rail = np.zeros(CONTROLLED_COUNT)
        rail[:2] = (0.2, 1.0)
        rail.setflags(write=False)
        base_total = (
            batch.global_component + batch.local_component + rail
            - batch.smooth_penalty
        )
        batch = replace(batch, rail_reward_postclip=rail, total=base_total)
        diagnostics = builder.diagnostics(batch)
        expected = {
            "tat": abs(builder.config.global_alpha * batch.tat_raw_ramped),
            "backlog": abs(
                builder.config.global_alpha * batch.backlog_raw
            ),
            "local": float(np.mean(np.abs(batch.local_component))),
            "rail": 1.2 / CONTROLLED_COUNT,
            "smooth": float(np.mean(np.abs(batch.smooth_penalty))),
        }
        self.assertAlmostEqual(
            diagnostics["reward/contribution/tat_abs"], expected["tat"]
        )
        self.assertAlmostEqual(
            diagnostics["reward/contribution/backlog_abs"],
            expected["backlog"],
        )
        self.assertAlmostEqual(
            diagnostics["reward/contribution/local_abs"], expected["local"]
        )
        self.assertAlmostEqual(
            diagnostics["reward/budget/tat_abs"], expected["tat"]
        )
        self.assertAlmostEqual(
            diagnostics["reward/budget/backlog_abs"], expected["backlog"]
        )
        self.assertAlmostEqual(
            diagnostics["reward/budget/rail_abs"], expected["rail"]
        )
        budget_total = sum(expected.values())
        for name, value in expected.items():
            self.assertAlmostEqual(
                diagnostics[f"reward/budget/{name}_share"],
                value / budget_total,
            )
        self.assertAlmostEqual(
            batch.global_component,
            builder.config.global_alpha
            * (batch.tat_raw_ramped + batch.backlog_raw),
        )
        np.testing.assert_allclose(
            batch.total,
            builder.config.global_alpha * batch.tat_raw_ramped
            + builder.config.global_alpha * batch.backlog_raw
            + batch.local_component
            + batch.rail_reward_postclip
            - batch.smooth_penalty,
        )
        self.assertLess(diagnostics["reward/budget/share_sum_error"], 1e-12)

        terminal = replace(
            batch,
            total=batch.total - 20.0,
            terminal_penalty=-20.0,
        )
        terminal_diagnostics = builder.diagnostics(terminal)
        for key in diagnostics:
            if key.startswith("reward/budget/"):
                self.assertEqual(diagnostics[key], terminal_diagnostics[key])
        self.assertLess(
            terminal_diagnostics["reward/contribution/sum_error"], 1e-6
        )

    def test_route_ratio_tail_does_not_copy_stale_step_values(self):
        builder = ContextualRewardBuilder(self.topology, _ContextualRewardConfig())
        builder._recent_route_ratios.extend((1.0, 2.5))
        builder._recent_route_negative.extend((0.0, 1.0))
        builder._route_ratio_sample_added = True
        available = builder.route_ratio_diagnostics()
        self.assertEqual(available["lead/route_ratio/available"], 1.0)
        self.assertAlmostEqual(available["lead/route_ratio/mean"], 1.75)

        builder._route_ratio_sample_added = False
        unavailable = builder.route_ratio_diagnostics()
        self.assertEqual(unavailable["lead/route_ratio/available"], 0.0)
        self.assertEqual(unavailable["lead/route_ratio/mean"], 0.0)
        self.assertEqual(unavailable["lead/route_ratio/p95"], 0.0)

    def test_reward_i_uses_raw_scaled_components_without_normalizer(self):
        config = ContextualRewardConfig(
            freeze_after_env_steps=1,
            rail_tat_weight=100.0,
            rail_tat_clip=0.5,
        )
        builder = ContextualRewardBuilder(self.topology, config)
        batch = builder.build(
            reward_client(),
            applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )

        self.assertEqual(builder.global_normalizer.count, 0)
        self.assertEqual(builder.local_normalizer.count, 0)
        self.assertFalse(builder.global_normalizer.frozen)
        self.assertFalse(builder.local_normalizer.frozen)
        self.assertEqual(batch.global_normalized, batch.global_raw)
        np.testing.assert_allclose(
            batch.local_normalized,
            batch.local_raw / config.local_reward_scale,
        )
        np.testing.assert_allclose(
            batch.total,
            0.5 * batch.global_raw
            + 0.5 * batch.local_raw / 3.0
            - batch.rail_tat_penalty
            - batch.smooth_penalty,
        )

    def test_reward_i_never_calls_reward_normalizer_methods(self):
        builder = ContextualRewardBuilder(self.topology, self.config)
        def forbidden(*args, **kwargs):
            raise AssertionError("reward normalizer must remain inactive")
        for normalizer in (
            builder.global_normalizer, builder.local_normalizer
        ):
            normalizer.normalize = forbidden
            normalizer.update = forbidden
            normalizer.freeze = forbidden
        builder.build(
            reward_client(),
            applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )

    def test_idle_observation_changes_but_idle_reward_stays_zero(self):
        first_client = reward_client()
        second_client = reward_client()
        for rail in second_client.RAILLINE_DIC.values():
            rail.IdleOHTCount += 7
        first = ContextualRewardBuilder(self.topology, self.config).build(
            first_client,
            applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )
        second = ContextualRewardBuilder(self.topology, self.config).build(
            second_client,
            applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )
        np.testing.assert_allclose(first.local_raw, second.local_raw)
        np.testing.assert_array_equal(first.local_idle_raw, 0.0)
        np.testing.assert_array_equal(second.local_idle_raw, 0.0)
        self.assertGreater(
            second.idle_oht_observation.mean(),
            first.idle_oht_observation.mean(),
        )

    def test_rail_tat_weight_and_clip_are_applied_exactly_once(self):
        builder = ContextualRewardBuilder(
            self.topology,
            ContextualRewardConfig(
                rail_tat_weight=100.0, rail_tat_clip=0.5
            ),
        )
        raw = np.asarray([0.001, 0.01, -0.01])
        weighted, postclip = builder._scale_rail_tat(raw)
        np.testing.assert_allclose(weighted, [0.1, 1.0, -1.0])
        np.testing.assert_allclose(postclip, [0.1, 0.5, -0.5])

    def test_runtime_keeps_n_tat_unbounded_and_clips_weighted_rail_tat(self):
        tat_config = ContextualRewardConfig(
            tat_reference=100.0,
            tat_weight=20.0,
            tat_raw_clip=1.0,
            use_op=False,
            use_backlog=False,
        )
        tat_client = reward_client()
        tat_client.TotalTat = 1_000.0
        tat_batch = ContextualRewardBuilder(
            self.topology, tat_config
        ).build(
            tat_client,
            applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )
        self.assertAlmostEqual(tat_batch.tat_error, -8.4)
        self.assertAlmostEqual(tat_batch.tat_raw_preclip, -168.0)
        self.assertEqual(tat_batch.tat_raw_postclip, -168.0)

        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        rail_builder = ContextualRewardBuilder(
            topology,
            ContextualRewardConfig(
                tat_reference=100.0,
                rail_tat_weight=100.0,
                rail_tat_clip=0.5,
            ),
        )
        client = rail_tat_client(state=0)
        rail_builder._rail_tat(client, env_step=0)
        oht = client.OHT_DIC[7]
        oht.State = 2
        oht.JobID = oht.DispatchedCommand = 1
        oht.PassTimes = [rail_pass(1, 2, 1.0)]
        oht.CmdCompleteTat = {1: completion(1, 10.0, 10.0)}
        rail_builder._rail_tat(client, env_step=1)
        oht.State = 5
        oht.PassTimes = []
        oht.CmdCompleteTat = {1: completion(1, 200.0, 200.0)}
        rail_builder._rail_tat(client, env_step=2)
        oht.State = 0
        oht.CmdCompleteTat = {}
        raw = rail_builder._rail_tat_raw(client, env_step=3)
        np.testing.assert_allclose(raw, [1.0])
        np.testing.assert_allclose(
            np.clip(100.0 * raw, -0.5, 0.5), [0.5]
        )

    def test_shape_alignment_components_and_inactive_normalizer_contract(self):
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
        self.assertEqual(builder.local_normalizer.update_calls, 0)
        self.assertEqual(builder.local_normalizer.count, 0)
        self.assertEqual(builder.global_normalizer.update_calls, 0)
        self.assertEqual(builder.global_normalizer.count, 0)
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
            ContextualRewardBuilder(
                self.topology, ContextualRewardConfig()
            ).build(
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

    def test_normalizer_npz_records_and_strictly_checks_reward_version(self):
        source = ContextualRewardBuilder(self.topology, self.config)
        source.build(
            reward_client(), applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None, env_step=0, episode_id=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = source.save_normalizers(directory / "reward.npz")
            with np.load(path, allow_pickle=False) as saved:
                payload = {key: saved[key] for key in saved.files}
            self.assertEqual(str(payload["reward_version"].item()), REWARD_VERSION)
            self.assertEqual(
                str(payload["reward_contract_version"].item()),
                REWARD_CONTRACT_VERSION,
            )

            for version, contract in (
                (None, REWARD_CONTRACT_VERSION),
                ("legacy_reward_contract", REWARD_CONTRACT_VERSION),
                (REWARD_VERSION, None),
                (REWARD_VERSION, "legacy_reward_contract"),
            ):
                changed = dict(payload)
                if version is None:
                    changed.pop("reward_version")
                else:
                    changed["reward_version"] = np.asarray(version)
                if contract is None:
                    changed.pop("reward_contract_version")
                else:
                    changed["reward_contract_version"] = np.asarray(contract)
                incompatible = directory / (
                    f"incompatible_{version}_{contract}.npz"
                )
                np.savez_compressed(incompatible, **changed)
                target = ContextualRewardBuilder(self.topology, self.config)
                original_count = target.local_normalizer.count
                with self.assertRaisesRegex(
                    ContextualRewardError, "normalizer version mismatch"
                ):
                    target.load_normalizers(incompatible)
                self.assertEqual(
                    target.local_normalizer.count, original_count
                )

    def test_reward_t_normalizer_state_is_rejected_by_reward_u(self):
        reward_t = ContextualRewardBuilder(
            self.topology, _ContextualRewardConfig.for_version("T")
        )
        reward_u = ContextualRewardBuilder(
            self.topology, _ContextualRewardConfig.for_version("U")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = reward_t.save_normalizers(Path(directory) / "reward_t.npz")
            with self.assertRaisesRegex(
                ContextualRewardError, "normalizer version mismatch"
            ):
                reward_u.load_normalizers(path)

    def test_reward_normalizer_rejects_changed_reward_coefficient(self):
        source = ContextualRewardBuilder(self.topology, self.config)
        source.build(
            reward_client(),
            applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )
        changed_config = replace(
            self.config,
            tat_weight=self.config.tat_weight + 1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = source.save_normalizers(Path(directory) / "reward.npz")
            changed = ContextualRewardBuilder(self.topology, changed_config)
            with self.assertRaisesRegex(
                ContextualRewardError, "profile fingerprint mismatch"
            ):
                changed.load_normalizers(path)

    def test_nonpositive_total_tat_does_not_receive_a_tat_reward(self):
        action = np.zeros(CONTROLLED_COUNT)
        for total_tat in (0.0, -1.0):
            with self.subTest(total_tat=total_tat):
                builder = ContextualRewardBuilder(self.topology, self.config)
                client = reward_client()
                client.TotalTat = total_tat
                batch = builder.build(
                    client, applied_action=action,
                    previous_applied_action=None,
                    env_step=0, episode_id=0,
                )
                self.assertEqual(batch.tat_signal_available, 0.0)
                self.assertEqual(batch.total_tat_level, total_tat)
                self.assertEqual(batch.tat_raw_unramped, 0.0)
                self.assertEqual(batch.tat_raw_ramped, 0.0)
                self.assertAlmostEqual(
                    batch.global_raw,
                    self.config.op_weight * (
                        self.config.op_reference
                        - client.TotalOhtOperationRate
                    )
                    - self.config.backlog_weight * (
                        client.WaitingCommandCount
                        + client.QueuedCommandCount
                    ),
                )

    def test_completed_count_never_scales_tat_reward(self):
        action = np.zeros(CONTROLLED_COUNT)
        ramp_config = ContextualRewardConfig(
            freeze_after_env_steps=100,
            tat_confidence_ramp=True,
        )
        values = []
        for completed in (1, 50, 200):
            builder = ContextualRewardBuilder(self.topology, ramp_config)
            client = reward_client()
            client.CompletedCommandCount = completed
            batch = builder.build(
                client, applied_action=action, previous_applied_action=None,
                env_step=0, episode_id=0,
            )
            self.assertEqual(batch.tat_confidence, 1.0)
            values.append(batch.tat_raw_postclip)
        self.assertEqual(values, [values[0]] * len(values))

    def test_tat_confidence_ramp_can_be_disabled(self):
        config = ContextualRewardConfig(
            freeze_after_env_steps=100,
            tat_confidence_ramp=False,
        )
        builder = ContextualRewardBuilder(self.topology, config)
        client = reward_client()
        client.CompletedCommandCount = 1
        batch = builder.build(
            client,
            applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )
        self.assertEqual(batch.tat_signal_available, 1.0)
        self.assertEqual(batch.tat_confidence, 1.0)
        self.assertEqual(batch.tat_raw_ramped, batch.tat_raw_unramped)

    def test_configured_global_terms_use_one_sided_tat(self):
        self.assertTrue(self.config.use_tat)
        self.assertTrue(self.config.use_op)
        self.assertTrue(self.config.use_backlog)
        self.assertFalse(self.config.tat_confidence_ramp)
        self.assertEqual(self.config.tat_reference, 174.4236)
        self.assertEqual(self.config.tat_weight, 9.2)
        self.assertEqual(self.config.op_reference, 0.80)
        self.assertEqual(self.config.op_weight, 5.0)
        self.assertEqual(self.config.backlog_weight, 0.002)

        builder = ContextualRewardBuilder(self.topology, self.config)
        client = reward_client()
        client.TotalTat = 170.0
        client.TotalOhtOperationRate = 0.79
        client.CompletedCommandCount = 1
        client.WaitingCommandCount = 40
        client.QueuedCommandCount = 60
        batch = builder.build(
            client,
            applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=None,
            env_step=0,
            episode_id=0,
        )

        expected_tat = -9.2 * (170.0 - 160.0) / 174.4236
        expected_op = 5.0 * (0.80 - 0.79)
        expected_backlog = -0.002 * 100.0
        self.assertAlmostEqual(batch.tat_raw_ramped, expected_tat)
        self.assertEqual(batch.total_tat_level, 170.0)
        self.assertAlmostEqual(batch.op_rate, 0.79)
        self.assertAlmostEqual(batch.op_reference, 0.80)
        self.assertAlmostEqual(batch.op_error, 0.01)
        self.assertEqual(batch.op_delta, 0.0)
        self.assertAlmostEqual(batch.op_raw, expected_op)
        self.assertAlmostEqual(batch.backlog_raw, expected_backlog)
        self.assertAlmostEqual(
            batch.global_raw,
            expected_tat + expected_op + expected_backlog,
        )

        client.TotalOhtOperationRate = 0.81
        client.CompletedCommandCount = 0
        second = builder.build(
            client,
            applied_action=np.zeros(CONTROLLED_COUNT),
            previous_applied_action=np.zeros(CONTROLLED_COUNT),
            env_step=1,
            episode_id=0,
        )
        self.assertAlmostEqual(second.op_delta, -0.02)
        self.assertAlmostEqual(second.op_raw, -0.05)

    def test_total_tat_reward_is_independent_of_completed_count(self):
        expected = -9.2 * (170.0 - 160.0) / 174.4236
        observed = []
        for completed in (1, 100, 100_000):
            builder = ContextualRewardBuilder(self.topology, self.config)
            client = reward_client()
            client.TotalTat = 170.0
            client.CompletedCommandCount = completed
            batch = builder.build(
                client,
                applied_action=np.zeros(CONTROLLED_COUNT),
                previous_applied_action=None,
                env_step=0,
                episode_id=0,
            )
            observed.append(batch.tat_raw_unramped)
            self.assertAlmostEqual(batch.tat_raw_unramped, expected)
        self.assertTrue(all(value == observed[0] for value in observed))

    def test_total_tat_quantization_change_is_not_count_amplified(self):
        rewards = []
        for total_tat in (165.0, 165.1):
            builder = ContextualRewardBuilder(self.topology, self.config)
            client = reward_client()
            client.TotalTat = total_tat
            client.CompletedCommandCount = 100_000
            rewards.append(builder.build(
                client,
                applied_action=np.zeros(CONTROLLED_COUNT),
                previous_applied_action=None,
                env_step=0,
                episode_id=0,
            ).tat_raw_unramped)
        self.assertAlmostEqual(
            rewards[0] - rewards[1],
            self.config.tat_weight * 0.1 / self.config.tat_reference,
        )

    def test_invalid_total_tat_signal_contract(self):
        action = np.zeros(CONTROLLED_COUNT)
        client = reward_client()
        client.TotalTat = 0.0
        zero = ContextualRewardBuilder(self.topology, self.config).build(
            client, applied_action=action, previous_applied_action=None,
            env_step=0, episode_id=0,
        )
        self.assertEqual(zero.tat_signal_available, 0.0)
        self.assertEqual(zero.tat_raw_unramped, 0.0)

        for invalid in (np.nan, np.inf):
            client.TotalTat = invalid
            with self.assertRaisesRegex(
                ContextualRewardError, "global reward input contains NaN or Inf"
            ):
                ContextualRewardBuilder(self.topology, self.config).build(
                    client, applied_action=action,
                    previous_applied_action=None, env_step=0, episode_id=0,
                )

    def test_reset_has_no_marginal_state_and_preserves_normalizers(self):
        builder = ContextualRewardBuilder(self.topology, self.config)
        client = reward_client()
        client.CompletedCommandCount = 50
        action = np.zeros(CONTROLLED_COUNT)
        builder.build(
            client, applied_action=action, previous_applied_action=None,
            env_step=0, episode_id=0,
        )
        counts = (
            builder.global_normalizer.count,
            builder.local_normalizer.count,
        )
        builder.reset_episode()
        self.assertEqual(builder._total_completed_jobs, 0)
        self.assertFalse(hasattr(builder, "_tat_ema"))
        self.assertFalse(hasattr(builder, "_prev_tat_sum"))
        self.assertFalse(hasattr(builder, "_prev_completed"))
        self.assertEqual(
            counts,
            (builder.global_normalizer.count, builder.local_normalizer.count),
        )
        client.CompletedCommandCount = 0
        reset_batch = builder.build(
            client, applied_action=action, previous_applied_action=action,
            env_step=1, episode_id=1,
        )
        self.assertEqual(reset_batch.total_tat_level, client.TotalTat)
        self.assertEqual(reset_batch.tat_confidence, 1.0)

    def test_full_reward_diagnostics_reproduce_contract(self):
        builder = ContextualRewardBuilder(self.topology, self.config)
        client = reward_client()
        client.CompletedCommandCount = 50
        action = np.linspace(-1, 1, CONTROLLED_COUNT)
        batch = builder.build(
            client, applied_action=action,
            previous_applied_action=np.zeros_like(action),
            env_step=0, episode_id=0,
        )
        diagnostics = builder.diagnostics(batch)
        expected_global_keys = {
            "reward/global/total_tat",
            "reward/global/tat_reference",
            "reward/global/tat_error",
            "reward/global/tat_weight",
            "reward/global/tat_signal_available",
            "reward/global/tat_raw_preclip",
            "reward/global/tat_raw_postclip",
            "reward/global/tat_confidence",
            "reward/global/tat_raw_ramped",
            "reward/global/op_rate",
            "reward/global/op_reference",
            "reward/global/op_error",
            "reward/global/op_raw",
            "reward/global/backlog",
            "reward/global/backlog_weight",
            "reward/global/backlog_raw",
            "reward/global/raw_sum",
            "reward/global/raw_decomposition_error",
        }
        self.assertTrue(expected_global_keys <= diagnostics.keys())
        required_reward_u_keys = {
            "reward/global/tat_component_raw",
            "reward/global/backlog_component_raw",
            "reward/global/raw",
            "reward/global/normalized",
            "reward/global/component",
            "reward/completion/count",
            "reward/completion/valid_count",
            "reward/completion/invalid_count",
            "reward/completion/tat_mean",
            "reward/completion/tat_std",
            "reward/completion/tat_min",
            "reward/completion/tat_max",
            "reward/completion/raw",
            "reward/completion/weighted_raw",
            "reward/local/raw_mean",
            "reward/local/raw_std",
            "reward/local/normalized_mean",
            "reward/local/normalized_std",
            "reward/local/component_mean",
            "reward/local/component_std",
            "reward/rail_tat_mean",
            "reward/smooth_penalty_mean",
            "reward/total_mean",
            "reward/total_std",
            "reward/global_normalizer_mean",
            "reward/global_normalizer_std",
            "reward/local_normalizer_mean",
            "reward/local_normalizer_std",
            "reward/budget/tat_raw_abs",
            "reward/budget/backlog_raw_abs",
            "reward/budget/global_abs",
            "reward/budget/local_abs",
            "reward/budget/rail_abs",
            "reward/budget/smooth_abs",
            "reward/contribution/tat_abs",
            "reward/contribution/completion_tat_abs",
            "reward/contribution/backlog_abs",
            "reward/contribution/local_abs",
            "reward/budget/tat_abs",
            "reward/budget/backlog_abs",
            "reward/budget/tat_share",
            "reward/budget/completion_tat_share",
            "reward/budget/backlog_share",
            "local/oht_abs_mean",
            "local/oht_std",
            "local/pred_abs_mean",
            "local/pred_std",
            "local/stop_abs_mean",
            "local/stop_std",
            "local/idle_abs_mean",
            "local/idle_std",
            "local/capacity_abs_mean",
            "local/capacity_std",
        }
        self.assertTrue(required_reward_u_keys <= diagnostics.keys())
        for name, values in (
            ("oht", batch.local_oht_raw),
            ("pred", batch.local_predicted_raw),
            ("stop", batch.local_stop_raw),
            ("idle", batch.local_idle_raw),
            ("capacity", batch.local_capacity_raw),
        ):
            self.assertAlmostEqual(
                diagnostics[f"local/{name}_abs_mean"],
                float(np.abs(values).mean()),
            )
            self.assertAlmostEqual(
                diagnostics[f"local/{name}_std"],
                float(values.std()),
            )
        self.assertAlmostEqual(
            batch.global_raw,
            batch.tat_raw_ramped + batch.op_raw + batch.backlog_raw,
        )
        np.testing.assert_allclose(
            batch.local_raw,
            batch.local_oht_raw + batch.local_predicted_raw
            + batch.local_stop_raw + batch.local_idle_raw
            + batch.local_capacity_raw,
        )
        np.testing.assert_allclose(
            batch.total,
            batch.global_component + batch.local_component
            - batch.rail_tat_penalty - batch.smooth_penalty,
        )
        self.assertLess(
            diagnostics["reward/global/raw_decomposition_error"], 1e-12
        )
        self.assertLess(
            diagnostics["reward/local/raw_decomposition_error_max"], 1e-12
        )
        self.assertLess(
            diagnostics["reward/contribution/sum_error"], 1e-6
        )
        self.assertLess(
            diagnostics["reward/scale/abs_share_sum_error"], 1e-12
        )
        self.assertTrue(np.isfinite(tuple(diagnostics.values())).all())

    def test_reward_normalizer_state_cannot_change_reward_i(self):
        config = ContextualRewardConfig(
            freeze_after_env_steps=100,
            tat_weight=100.0,
        )
        builder = ContextualRewardBuilder(self.topology, config)
        builder.global_normalizer.mean[:] = 0
        builder.global_normalizer.m2[:] = 1
        builder.global_normalizer.count = 1
        builder.local_normalizer.mean[:] = -1
        builder.local_normalizer.m2[:] = 4
        builder.local_normalizer.count = 2
        client = reward_client()
        client.CompletedCommandCount = 1_000
        action = np.zeros(CONTROLLED_COUNT)
        first = builder.build(
            client, applied_action=action, previous_applied_action=None,
            env_step=0, episode_id=0,
        )
        builder.global_normalizer.mean[:] = 1e9
        builder.local_normalizer.mean[:] = -1e9
        second = builder.build(
            client, applied_action=action, previous_applied_action=None,
            env_step=1, episode_id=0,
        )
        np.testing.assert_allclose(first.total, second.total)
        self.assertEqual(builder.global_normalizer.update_calls, 0)
        self.assertEqual(builder.local_normalizer.update_calls, 0)

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

    def test_rail_tat_oht_cycle_completion_job_change_and_logging(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1, 2], dtype=np.int64)
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        diagnostic_path = Path(directory.name) / "rail_tat.jsonl"
        global_step = {"value": 0}
        builder = ContextualRewardBuilder(
            topology,
            ContextualRewardConfig(tat_reference=100.0),
            completion_diagnostic_path=diagnostic_path,
            global_step_provider=lambda: global_step["value"],
            completion_diagnostic_max_global_step=1_000,
        )
        client = rail_tat_client(
            state=0,
            passes=[rail_pass(99, 0, 50.0)],
        )

        # Initial idle movement is not part of a transport cycle.
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=0),
            np.zeros(2),
        )
        tracker = builder._oht_cycle_trackers[7]
        self.assertFalse(tracker.cycle_active)
        self.assertEqual(tracker.route_count, 0)

        # 0 -> 2 starts a cycle; command appearance itself gives no reward.
        global_step["value"] = 1
        client.OHT_DIC[7].State = 2
        client.OHT_DIC[7].JobID = 10
        client.OHT_DIC[7].DispatchedCommand = 10
        client.OHT_DIC[7].PassTimes = [rail_pass(1, 2, 2.0)]
        client.OHT_DIC[7].CmdCompleteTat = {
            10: completion(10, 20.0, 10.0)
        }
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=1), np.zeros(2)
        )
        self.assertTrue(tracker.cycle_active)
        self.assertEqual(tracker.rail_time_by_id, {1: 2.0})

        # A JobID change does not end the OHT cycle or clear its route.
        global_step["value"] = 2
        client.OHT_DIC[7].State = 4
        client.OHT_DIC[7].JobID = 11
        client.OHT_DIC[7].DispatchedCommand = 11
        client.OHT_DIC[7].PassTimes = [
            rail_pass(2, 4, 3.0),
            rail_pass(99, 4, 5.0),
        ]
        client.OHT_DIC[7].CmdCompleteTat = {
            11: completion(11, 50.0, 40.0)
        }
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=2), np.zeros(2)
        )
        self.assertEqual(tracker.rail_time_by_id, {1: 2.0, 2: 3.0, 99: 5.0})

        # 4 -> 5 and repeated 5 are active, non-terminal snapshots.
        global_step["value"] = 3
        client.OHT_DIC[7].State = 5
        client.OHT_DIC[7].PassTimes = []
        client.OHT_DIC[7].CmdCompleteTat = {
            11: completion(11, 250.0, 200.0)
        }
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=3), np.zeros(2)
        )
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=4), np.zeros(2)
        )

        # 5 -> 0 applies final state-5 OHTTat exactly once. Rail 99 remains
        # in the denominator but is absent from controlled output.
        global_step["value"] = 5
        client.OHT_DIC[7].State = 0
        client.OHT_DIC[7].JobID = 0
        client.OHT_DIC[7].DispatchedCommand = 0
        client.OHT_DIC[7].CmdCompleteTat = {}
        completed = builder._rail_tat(client, env_step=5)
        # Segment A contributes 10 and final segment B contributes 200.
        np.testing.assert_allclose(completed, [0.22, 0.33])
        self.assertEqual(builder._last_rail_tat_cycle_count, 1)
        self.assertEqual(
            builder._last_rail_tat_controlled_assignment_count, 2
        )
        self.assertEqual(
            builder._last_rail_tat_uncontrolled_assignment_count, 1
        )
        self.assertEqual(builder._last_rail_tat_event_count, 2)
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=6), np.zeros(2)
        )
        self.assertEqual(builder._last_rail_tat_event_count, 0)
        self.assertFalse(tracker.cycle_active)

        records = [
            json.loads(line)
            for line in diagnostic_path.read_text(encoding="utf-8").splitlines()
        ]
        events = [record["event"] for record in records]
        self.assertIn("oht_cycle_start", events)
        self.assertIn("oht_job_changed", events)
        self.assertIn("oht_unloading", events)
        self.assertIn("oht_cycle_completed", events)
        final = next(
            record
            for record in records
            if record["event"] == "oht_cycle_completed"
        )
        self.assertEqual(final["last_oht_tat"], 200.0)
        self.assertEqual(final["reward_oht_tat_used"], 210.0)
        self.assertEqual(final["controlled_reward_count"], 2)
        self.assertEqual(final["uncontrolled_reward_count"], 1)
        self.assertTrue(final["reward_applied"])

        # Exclusive diagnostic cutoff: a new cycle at step 1000 is not logged.
        lines_before_cutoff = diagnostic_path.read_text(
            encoding="utf-8"
        ).splitlines()
        global_step["value"] = 1_000
        client.OHT_DIC[7].State = 2
        client.OHT_DIC[7].JobID = 12
        client.OHT_DIC[7].DispatchedCommand = 12
        client.OHT_DIC[7].PassTimes = [rail_pass(1, 2, 1.0)]
        client.OHT_DIC[7].CmdCompleteTat = {
            12: completion(12, 1.0, 1.0)
        }
        builder._rail_tat(client, env_step=7)
        self.assertEqual(
            diagnostic_path.read_text(encoding="utf-8").splitlines(),
            lines_before_cutoff,
        )

    def test_rail_tat_five_to_two_finalizes_before_new_cycle(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1, 2, 3], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology, ContextualRewardConfig(tat_reference=100.0)
        )
        client = rail_tat_client(state=0)
        builder._rail_tat(client, env_step=0)

        client.OHT_DIC[7].State = 2
        client.OHT_DIC[7].JobID = 20
        client.OHT_DIC[7].DispatchedCommand = 20
        client.OHT_DIC[7].PassTimes = [rail_pass(1, 2, 2.0)]
        client.OHT_DIC[7].CmdCompleteTat = {
            20: completion(20, 20.0, 10.0)
        }
        builder._rail_tat(client, env_step=1)

        client.OHT_DIC[7].State = 5
        client.OHT_DIC[7].PassTimes = [rail_pass(2, 4, 3.0)]
        client.OHT_DIC[7].CmdCompleteTat = {
            20: completion(20, 250.0, 200.0)
        }
        builder._rail_tat(client, env_step=2)

        # Old state-4 passes before the first state-2 pass are not mixed into
        # the restarted cycle, and current state-2 OHTTat cannot overwrite the
        # previous cycle's final state-5 value.
        client.OHT_DIC[7].State = 2
        client.OHT_DIC[7].JobID = 21
        client.OHT_DIC[7].DispatchedCommand = 21
        client.OHT_DIC[7].PassTimes = [
            rail_pass(99, 4, 9.0),
            rail_pass(3, 2, 1.0),
        ]
        client.OHT_DIC[7].CmdCompleteTat = {
            21: completion(21, 1.0, 1.0)
        }
        restarted = builder._rail_tat(client, env_step=3)
        # The state-4 completion in the boundary packet is still old-cycle
        # time, so uncontrolled rail 99 remains in the denominator.
        np.testing.assert_allclose(
            restarted, [2.0 / 14.0, 3.0 / 14.0, 0.0]
        )
        tracker = builder._oht_cycle_trackers[7]
        self.assertTrue(tracker.cycle_active)
        self.assertFalse(tracker.bootstrapped)
        self.assertEqual(tracker.rail_time_by_id, {3: 1.0})
        self.assertEqual(tracker.last_valid_oht_tat, 1.0)

        # The restarted cycle completes independently and preserves the
        # signed fast-cycle reward direction.
        client.OHT_DIC[7].State = 5
        client.OHT_DIC[7].PassTimes = []
        client.OHT_DIC[7].CmdCompleteTat = {
            21: completion(21, 80.0, 50.0)
        }
        builder._rail_tat(client, env_step=4)
        client.OHT_DIC[7].State = 0
        client.OHT_DIC[7].CmdCompleteTat = {}
        np.testing.assert_allclose(
            builder._rail_tat(client, env_step=5),
            [0.0, 0.0, -0.5],
        )

    def test_rail_tat_bootstrap_abort_ambiguity_and_reset(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1, 2], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology,
            ContextualRewardConfig(tat_reference=100.0),
        )
        client = rail_tat_client(
            state=4,
            job_id=10,
            dispatched_command=10,
            passes=[rail_pass(1, 4, 2.0)],
            completions={10: completion(10, 220.0, 200.0)},
        )

        # A mid-cycle first snapshot is tracked but excluded from training.
        bootstrapped = builder._rail_tat(client, env_step=0)
        np.testing.assert_array_equal(bootstrapped, np.zeros(2))
        tracker = builder._oht_cycle_trackers[7]
        self.assertTrue(tracker.bootstrapped)
        client.OHT_DIC[7].PassTimes = []
        client.OHT_DIC[7].State = 5
        builder._rail_tat(client, env_step=1)
        client.OHT_DIC[7].State = 0
        client.OHT_DIC[7].CmdCompleteTat = {}
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=2), np.zeros(2)
        )

        # A normal active cycle that leaves 2/3/4 directly is aborted.
        client.OHT_DIC[7].State = 2
        client.OHT_DIC[7].JobID = 30
        client.OHT_DIC[7].DispatchedCommand = 30
        client.OHT_DIC[7].PassTimes = [rail_pass(1, 2, 1.0)]
        client.OHT_DIC[7].CmdCompleteTat = {
            30: completion(30, 10.0, 10.0)
        }
        builder._rail_tat(client, env_step=3)
        client.OHT_DIC[7].State = 0
        client.OHT_DIC[7].PassTimes = []
        client.OHT_DIC[7].CmdCompleteTat = {}
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=4), np.zeros(2)
        )
        self.assertFalse(tracker.cycle_active)
        self.assertEqual(tracker.rail_time_by_id, {})

        # Multiple unmatched entries are never selected by dict order.
        client.OHT_DIC[7].State = 2
        client.OHT_DIC[7].JobID = 999
        client.OHT_DIC[7].DispatchedCommand = 0
        client.OHT_DIC[7].PassTimes = [rail_pass(2, 2, 1.0)]
        client.OHT_DIC[7].CmdCompleteTat = {
            41: completion(41, 30.0, 20.0),
            40: completion(40, 40.0, 30.0),
        }
        builder._rail_tat(client, env_step=5)
        self.assertIsNone(tracker.last_valid_oht_tat)
        selected, reason = builder._select_oht_tat_entry(
            client.OHT_DIC[7],
            {
                40: completion(40, 40.0, 30.0),
                41: completion(41, 30.0, 20.0),
            },
        )
        self.assertIsNone(selected)
        self.assertEqual(reason, "ambiguous_tat_entries")
        client.OHT_DIC[7].State = 5
        client.OHT_DIC[7].PassTimes = []
        builder._rail_tat(client, env_step=6)
        client.OHT_DIC[7].State = 0
        client.OHT_DIC[7].CmdCompleteTat = {}
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=7), np.zeros(2)
        )

        # A state-4 TAT is not substituted for a missing final state-5 TAT.
        client.OHT_DIC[7].State = 2
        client.OHT_DIC[7].JobID = 50
        client.OHT_DIC[7].DispatchedCommand = 50
        client.OHT_DIC[7].PassTimes = [rail_pass(1, 2, 1.0)]
        client.OHT_DIC[7].CmdCompleteTat = {
            50: completion(50, 10.0, 10.0)
        }
        builder._rail_tat(client, env_step=8)
        client.OHT_DIC[7].State = 4
        client.OHT_DIC[7].CmdCompleteTat = {
            50: completion(50, 90.0, 90.0)
        }
        builder._rail_tat(client, env_step=9)
        client.OHT_DIC[7].State = 5
        client.OHT_DIC[7].CmdCompleteTat = {}
        builder._rail_tat(client, env_step=10)
        client.OHT_DIC[7].State = 0
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=11), np.zeros(2)
        )

        # Episode reset removes the complete OHT-cycle temporal state.
        builder.reset_episode()
        self.assertEqual(builder._oht_cycle_trackers, {})
        self.assertEqual(builder._rail_pass_trackers, {})

    def test_rail_tat_five_to_three_completes_then_restarts_once(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1, 2, 3], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology, ContextualRewardConfig(tat_reference=100.0)
        )
        client = rail_tat_client(state=0)
        builder._rail_tat(client, env_step=0)

        oht = client.OHT_DIC[7]
        oht.State = 2
        oht.JobID = 70
        oht.DispatchedCommand = 70
        oht.PassTimes = [rail_pass(1, 2, 2.0)]
        oht.CmdCompleteTat = {70: completion(70, 10.0, 10.0)}
        builder._rail_tat(client, env_step=1)
        oht.State = 5
        oht.PassTimes = [rail_pass(2, 5, 3.0)]
        oht.CmdCompleteTat = {70: completion(70, 220.0, 200.0)}
        builder._rail_tat(client, env_step=2)

        # Job identity is irrelevant: 5 -> 3 itself is the boundary.
        oht.State = 3
        oht.JobID = 70
        oht.PassTimes = [rail_pass(3, 3, 1.0)]
        oht.CmdCompleteTat = {70: completion(70, 1.0, 1.0)}
        first = builder._rail_tat(client, env_step=3)
        np.testing.assert_allclose(first, [0.4, 0.6, 0.0])
        tracker = builder._oht_cycle_trackers[7]
        self.assertTrue(tracker.cycle_active)
        self.assertEqual(tracker.current_state, 3)
        self.assertEqual(tracker.rail_time_by_id, {3: 1.0})
        self.assertEqual(tracker.current_segment_last_valid_oht_tat, 1.0)

        oht.PassTimes = []
        second = builder._rail_tat(client, env_step=4)
        np.testing.assert_array_equal(second, np.zeros(3))
        self.assertEqual(tracker.rail_time_by_id, {3: 1.0})

    def test_rail_tat_job_segments_sum_and_missing_value_ordering(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1, 2, 3], dtype=np.int64)
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "segments.jsonl"
        builder = ContextualRewardBuilder(
            topology,
            ContextualRewardConfig(tat_reference=100.0),
            completion_diagnostic_path=path,
        )
        client = rail_tat_client(state=0)
        builder._rail_tat(client, env_step=0)
        oht = client.OHT_DIC[7]

        oht.State = 2
        oht.JobID = oht.DispatchedCommand = 1
        oht.PassTimes = [rail_pass(1, 2, 1.0)]
        oht.CmdCompleteTat = {1: completion(1, 12.0, 12.0)}
        builder._rail_tat(client, env_step=1)

        # Close A at 12 before selecting B. B has no current packet TAT.
        oht.JobID = oht.DispatchedCommand = 2
        oht.PassTimes = [rail_pass(2, 2, 1.0)]
        oht.CmdCompleteTat = {}
        builder._rail_tat(client, env_step=2)
        tracker = builder._oht_cycle_trackers[7]
        self.assertEqual(tracker.closed_segment_oht_tat_sum, 12.0)
        self.assertIsNone(tracker.current_segment_last_valid_oht_tat)

        oht.CmdCompleteTat = {2: completion(2, 8.0, 8.0)}
        oht.PassTimes = []
        builder._rail_tat(client, env_step=3)
        oht.JobID = oht.DispatchedCommand = 3
        oht.PassTimes = [rail_pass(3, 2, 1.0)]
        oht.CmdCompleteTat = {3: completion(3, 0.5, 0.5)}
        builder._rail_tat(client, env_step=4)
        self.assertEqual(tracker.closed_segment_oht_tat_sum, 20.0)
        self.assertEqual(tracker.tat_segment_count, 3)

        oht.State = 5
        oht.PassTimes = []
        oht.CmdCompleteTat = {3: completion(3, 35.0, 35.0)}
        builder._rail_tat(client, env_step=5)
        oht.State = 0
        oht.CmdCompleteTat = {}
        penalty = builder._rail_tat(client, env_step=6)
        np.testing.assert_allclose(penalty, [-0.15, -0.15, -0.15])

        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        final = next(
            item for item in records
            if item["event"] == "oht_cycle_completed"
        )
        self.assertEqual(final["reward_oht_tat_used"], 55.0)
        self.assertAlmostEqual(final["signed_excess"], -0.45)
        self.assertAlmostEqual(final["reward_sum"], 0.45)

    def test_rail_boundary_partial_delta_and_completed_residual(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([100], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology, ContextualRewardConfig(tat_reference=100.0)
        )
        client = rail_tat_client(
            state=0,
            not_passes=[rail_pass(100, 0, 18.0)],
        )
        builder._rail_tat(client, env_step=0)
        oht = client.OHT_DIC[7]

        oht.State = 2
        oht.JobID = oht.DispatchedCommand = 9
        oht.NotPassTimes = [rail_pass(100, 2, 18.5)]
        oht.CmdCompleteTat = {9: completion(9, 1.0, 1.0)}
        builder._rail_tat(client, env_step=1)
        tracker = builder._oht_cycle_trackers[7]
        self.assertAlmostEqual(tracker.rail_time_by_id[100], 0.5)

        oht.NotPassTimes = [rail_pass(100, 2, 19.0)]
        builder._rail_tat(client, env_step=2)
        # Duplicate cumulative snapshot adds exactly zero.
        builder._rail_tat(client, env_step=3)
        oht.NotPassTimes = []
        oht.PassTimes = [rail_pass(100, 2, 20.0)]
        builder._rail_tat(client, env_step=4)
        self.assertAlmostEqual(tracker.rail_time_by_id[100], 2.0)
        self.assertAlmostEqual(tracker.route_time, 2.0)
        self.assertEqual(tracker.pass_delta_count, 3)
        self.assertAlmostEqual(tracker.boundary_time_subtracted, 18.0)

    def test_unexplained_oht_tat_reset_invalidates_cycle(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology, ContextualRewardConfig(tat_reference=100.0)
        )
        client = rail_tat_client(state=0)
        builder._rail_tat(client, env_step=0)
        oht = client.OHT_DIC[7]
        oht.State = 2
        oht.JobID = oht.DispatchedCommand = 5
        oht.PassTimes = [rail_pass(1, 2, 1.0)]
        oht.CmdCompleteTat = {5: completion(5, 20.0, 20.0)}
        builder._rail_tat(client, env_step=1)
        oht.PassTimes = []
        oht.CmdCompleteTat = {5: completion(5, 2.0, 2.0)}
        builder._rail_tat(client, env_step=2)
        self.assertFalse(
            builder._oht_cycle_trackers[7].cycle_tat_valid
        )
        oht.State = 5
        oht.CmdCompleteTat = {5: completion(5, 200.0, 200.0)}
        builder._rail_tat(client, env_step=3)
        oht.State = 0
        oht.CmdCompleteTat = {}
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=4), np.zeros(1)
        )

    def test_missing_closed_segment_tat_is_not_treated_as_zero(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology, ContextualRewardConfig(tat_reference=100.0)
        )
        client = rail_tat_client(state=0)
        builder._rail_tat(client, env_step=0)
        oht = client.OHT_DIC[7]
        oht.State = 2
        oht.JobID = oht.DispatchedCommand = 1
        oht.PassTimes = [rail_pass(1, 2, 1.0)]
        oht.CmdCompleteTat = {}
        builder._rail_tat(client, env_step=1)

        oht.JobID = oht.DispatchedCommand = 2
        oht.PassTimes = []
        oht.CmdCompleteTat = {2: completion(2, 5.0, 5.0)}
        builder._rail_tat(client, env_step=2)
        tracker = builder._oht_cycle_trackers[7]
        self.assertFalse(tracker.cycle_tat_valid)
        self.assertEqual(tracker.closed_segment_oht_tat_sum, 0.0)

        oht.State = 5
        oht.CmdCompleteTat = {2: completion(2, 200.0, 200.0)}
        builder._rail_tat(client, env_step=3)
        oht.State = 0
        oht.CmdCompleteTat = {}
        np.testing.assert_array_equal(
            builder._rail_tat(client, env_step=4), np.zeros(1)
        )

    def test_nonmonotonic_partial_and_ambiguous_boundary_pass_are_skipped(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology, ContextualRewardConfig(tat_reference=100.0)
        )
        client = rail_tat_client(state=0)
        builder._rail_tat(client, env_step=0)
        oht = client.OHT_DIC[7]
        oht.State = 2
        oht.JobID = oht.DispatchedCommand = 1
        # State 4 cannot be proven to belong after an idle -> state-2
        # boundary, so it is not arbitrarily assigned.
        oht.PassTimes = [rail_pass(1, 4, 9.0)]
        oht.NotPassTimes = [rail_pass(1, 2, 10.0)]
        oht.CmdCompleteTat = {1: completion(1, 1.0, 1.0)}
        builder._rail_tat(client, env_step=1)
        tracker = builder._oht_cycle_trackers[7]
        self.assertEqual(tracker.rail_time_by_id, {})
        self.assertEqual(tracker.ambiguous_pass_count, 1)

        # A backwards cumulative snapshot is ignored and does not replace
        # the prior baseline. Growth to 11 therefore contributes only 1.
        oht.PassTimes = []
        oht.NotPassTimes = [rail_pass(1, 2, 9.0)]
        builder._rail_tat(client, env_step=2)
        oht.NotPassTimes = [rail_pass(1, 2, 11.0)]
        builder._rail_tat(client, env_step=3)
        self.assertEqual(tracker.invalid_pass_delta_count, 1)
        self.assertAlmostEqual(tracker.rail_time_by_id[1], 1.0)

    def test_event_count_keeps_same_rail_from_two_oht_cycles_separate(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology, ContextualRewardConfig(tat_reference=100.0)
        )
        client = SimpleNamespace(OHT_DIC={})
        for oht_id in (7, 8):
            client.OHT_DIC[oht_id] = SimpleNamespace(
                ID=oht_id,
                State=0,
                JobID=0,
                DispatchedCommand=0,
                PassTimes=[],
                NotPassTimes=[],
                CmdCompleteTat={},
            )
        builder._rail_tat(client, env_step=0)

        for oht_id, oht in client.OHT_DIC.items():
            oht.State = 2
            oht.JobID = oht.DispatchedCommand = oht_id
            oht.PassTimes = [rail_pass(1, 2, 1.0)]
            oht.CmdCompleteTat = {
                oht_id: completion(oht_id, 10.0, 10.0)
            }
        builder._rail_tat(client, env_step=1)
        for oht_id, oht in client.OHT_DIC.items():
            oht.State = 5
            oht.PassTimes = []
            oht.CmdCompleteTat = {
                oht_id: completion(oht_id, 200.0, 200.0)
            }
        builder._rail_tat(client, env_step=2)
        for oht in client.OHT_DIC.values():
            oht.State = 0
            oht.CmdCompleteTat = {}
        penalty = builder._rail_tat(client, env_step=3)

        np.testing.assert_allclose(penalty, [2.0])
        self.assertEqual(builder._last_rail_tat_cycle_count, 2)
        self.assertEqual(
            builder._last_rail_tat_controlled_assignment_count, 2
        )
        self.assertEqual(
            builder._last_rail_tat_uncontrolled_assignment_count, 0
        )
        self.assertEqual(builder._last_rail_tat_event_count, 2)

    @staticmethod
    def free_flow_tracker(route_time, free_flow_time):
        tracker = OHTCycleTracker(oht_id=7)
        tracker.cycle_active = True
        tracker.route_time = float(route_time)
        tracker.route_count = 2
        tracker.rail_time_by_id = {
            1: float(route_time) * 0.6,
            99: float(route_time) * 0.4,
        }
        tracker.route_free_flow_time = float(free_flow_time)
        tracker.completed_rail_occurrence_count = 2
        return tracker

    def test_free_flow_reward_is_neutral_at_two_and_path_length_invariant(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology,
            ContextualRewardConfig(
                rail_reward_mode=RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
                rail_free_flow_neutral_ratio=2.0,
                rail_tat_weight=1.0,
                rail_tat_clip=1.0,
            ),
        )
        cases = (
            (200.0, 100.0, 0.0),
            (150.0, 100.0, 0.5),
            (250.0, 100.0, -0.5),
            (100.0, 50.0, 0.0),
        )
        for route_time, free_flow, expected in cases:
            with self.subTest(route_time=route_time, free_flow=free_flow):
                credit = {}
                outcome = builder._finalize_oht_cycle(
                    self.free_flow_tracker(route_time, free_flow), credit
                )
                self.assertTrue(outcome.reward_applied)
                self.assertAlmostEqual(outcome.cycle_rail_reward_raw, expected)
                self.assertAlmostEqual(sum(credit.values()), expected)
                self.assertAlmostEqual(
                    credit[1] + credit[99], outcome.cycle_rail_reward_raw
                )
        self.assertGreater(2.0 - 1.5, 0.0)
        self.assertLess(2.0 - 2.5, 0.0)

    def test_free_flow_invalid_cycle_skips_and_fixed_mode_is_compatible(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        free_flow_builder = ContextualRewardBuilder(
            topology,
            ContextualRewardConfig(
                rail_reward_mode=RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
            ),
        )
        invalid = self.free_flow_tracker(170.0, 0.0)
        invalid.completed_rail_occurrence_count = 0
        outcome = free_flow_builder._finalize_oht_cycle(invalid, {})
        self.assertFalse(outcome.reward_applied)
        self.assertEqual(outcome.skip_reason, "invalid_free_flow_cycle")

        fixed_builder = ContextualRewardBuilder(
            topology,
            ContextualRewardConfig(
                rail_reward_mode=RAIL_REWARD_FIXED_TAT_REFERENCE,
                tat_reference=100.0,
            ),
        )
        fixed = self.free_flow_tracker(10.0, 5.0)
        fixed.current_segment_last_state5_oht_tat = 200.0
        credit = {}
        outcome = fixed_builder._finalize_oht_cycle(fixed, credit)
        self.assertTrue(outcome.reward_applied)
        self.assertAlmostEqual(outcome.cycle_rail_reward_raw, -1.0)
        self.assertAlmostEqual(sum(credit.values()), -1.0)


if __name__ == "__main__":
    unittest.main()
