import sys
import unittest
from unittest.mock import patch

import torch

from ClientAlgorithm_contextual import ContextualRuntimeConfig
from contextual_dispatch import (
    DISPATCH_COST,
    DISPATCH_FIRST_MATCH,
)
from cocel_rl.algorithms.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualTD7Learner,
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    contextual_algorithm_variant,
)
from main_contextual import parse_args
from contextual_termination import (
    TAT_TERMINATION_EPISODE2_TAT175,
)
from contextual_reward_version_cfg import (
    RAIL_REWARD_FIXED_TAT_REFERENCE,
    TAT_SIGNAL_COMPLETION_EVENT,
    TAT_SIGNAL_MARGINAL_TAT_EMA,
    TAT_SIGNAL_TOTAL_TAT_LEVEL,
)
from test_contextual_learner import SMALL_NETWORK
from test_contextual_sale import sale_replay


EXPECTED = {
    (True, True): "contextual_td7_sale_lap_v3_prevact",
    (False, True): "contextual_td7_no_sale_lap_v3_prevact",
    (True, False): "contextual_td7_sale_uniform_v3_prevact",
    (False, False): "contextual_twin_delayed_uniform_v3_prevact",
}


class ContextualVariantTests(unittest.TestCase):
    def test_cli_selects_complete_locked_reward_t_or_u_profile(self):
        for argument, expected, signal in (
            ([], "E", TAT_SIGNAL_MARGINAL_TAT_EMA),
            (["--reward-version", "T"], "T", TAT_SIGNAL_TOTAL_TAT_LEVEL),
            (["--reward-version", "u"], "U", TAT_SIGNAL_COMPLETION_EVENT),
        ):
            with self.subTest(argument=argument), patch.object(
                sys, "argv", ["main_contextual.py", *argument]
            ):
                parsed = parse_args()
            config = ContextualRuntimeConfig(
                reward_version=parsed.reward_version
            )
            self.assertEqual(config.reward_version, expected)
            self.assertEqual(config.make_reward_config().tat_signal_mode, signal)
        with self.assertRaisesRegex(ValueError, "locked full reward profile"):
            ContextualRuntimeConfig(
                reward_version="T", backlog_weight=0.5
            )

    def test_cli_defaults_to_fixed_scale_reward_e_experiment(self):
        with patch.object(sys, "argv", ["main_contextual.py"]):
            parsed = parse_args()

        self.assertEqual(parsed.reward_version, "E")
        self.assertEqual(parsed.curriculum_scale_start, 1.0)
        self.assertEqual(parsed.curriculum_scale_end, 1.0)
        self.assertEqual(parsed.replay_capacity_env_steps, 100_000)
        self.assertEqual(parsed.batch_size, 1_024)
        self.assertFalse(parsed.resume_inference_until_replay_full)
        self.assertFalse(parsed.resume_deterministic_first_episode)
        self.assertEqual(parsed.warmup_steps, 10_000)
        self.assertTrue(parsed.terminate_on_warmup_complete)
        self.assertIsNone(parsed.load_state_normalizer)
        self.assertIsNone(parsed.save_state_normalizer)
        self.assertIsNone(parsed.load_reward_normalizer)
        self.assertIsNone(parsed.save_reward_normalizer)
        self.assertEqual(parsed.exploration_noise_std, 0.10)
        self.assertEqual(parsed.exploration_noise_final_std, 0.02)
        self.assertEqual(parsed.seed, 0)
        self.assertEqual(parsed.tat_termination_policy, "episode2_tat180")
        self.assertEqual(parsed.tat_termination_start_episode, 2)
        self.assertEqual(parsed.early_stop_tat_threshold, 180.0)
        self.assertEqual(parsed.tat_termination_grace_steps, 0)
        self.assertEqual(parsed.tat_above_threshold_patience, 1)

        with patch.object(
            sys,
            "argv",
            ["main_contextual.py", "--no-terminate-on-warmup-complete"],
        ):
            no_warmup_boundary = parse_args()
        self.assertFalse(no_warmup_boundary.terminate_on_warmup_complete)

        with patch.object(
            sys,
            "argv",
            [
                "main_contextual.py",
                "--resume-checkpoint", "checkpoint.pt",
                "--resume-inference-until-replay-full",
                "--batch-size", "2048",
            ],
        ):
            full_refill = parse_args()
        self.assertTrue(full_refill.resume_inference_until_replay_full)
        self.assertEqual(full_refill.batch_size, 2_048)

        with patch.object(
            sys,
            "argv",
            [
                "main_contextual.py",
                "--resume-checkpoint", "checkpoint.pt",
                "--resume-deterministic-first-episode",
            ],
        ):
            deterministic_first = parse_args()
        self.assertTrue(
            deterministic_first.resume_deterministic_first_episode
        )

        with patch.object(
            sys,
            "argv",
            ["main_contextual.py", "--save-reward-normalizer"],
        ):
            auto_reward_save = parse_args()
        self.assertEqual(auto_reward_save.save_reward_normalizer, "auto")
        with patch.object(
            sys,
            "argv",
            [
                "main_contextual.py",
                "--save-reward-normalizer", "reward_stats.npz",
                "--load-reward-normalizer", "reward_source.npz",
            ],
        ):
            explicit_reward_paths = parse_args()
        self.assertEqual(
            explicit_reward_paths.save_reward_normalizer,
            "reward_stats.npz",
        )
        self.assertEqual(
            explicit_reward_paths.load_reward_normalizer,
            "reward_source.npz",
        )

        with patch.object(
            sys,
            "argv",
            [
                "main_contextual.py",
                "--tat-termination-policy", "episode2_tat175",
            ],
        ):
            tat175 = parse_args()
        self.assertEqual(
            tat175.tat_termination_policy,
            TAT_TERMINATION_EPISODE2_TAT175,
        )
        self.assertEqual(tat175.tat_termination_start_episode, 2)
        self.assertEqual(tat175.early_stop_tat_threshold, 175.0)
        tat175_config = ContextualRuntimeConfig(
            reward_version="E",
            tat_termination_policy=tat175.tat_termination_policy,
        )
        self.assertEqual(tat175_config.early_stop_tat_threshold, 175.0)

        with patch.object(
            sys,
            "argv",
            [
                "main_contextual.py",
                "--load-state-normalizer", "normalizers/state_e.npz",
                "--save-state-normalizer", "normalizers/state_copy.npz",
            ],
        ):
            reuse = parse_args()
        reuse_config = ContextualRuntimeConfig(
            reward_version="E",
            tat_termination_policy=reuse.tat_termination_policy,
            load_state_normalizer_path=str(reuse.load_state_normalizer),
            save_state_normalizer_path=str(reuse.save_state_normalizer),
        )
        self.assertEqual(reuse_config.warmup_steps, 10_000)
        self.assertEqual(reuse_config.effective_warmup_steps, 0)
        self.assertTrue(reuse_config.state_normalizer_warmup_bypass)
        self.assertEqual(reuse_config.tat_termination_start_episode, 2)
        self.assertEqual(reuse_config.early_stop_tat_threshold, 180.0)
        self.assertEqual(
            reuse_config.effective_tat_termination_start_episode, 1
        )

        with patch.object(
            sys,
            "argv",
            [
                "main_contextual.py",
                "--reward-version", "E",
                "--tat-termination-policy", "reward_profile",
            ],
        ):
            historical = parse_args()
        self.assertEqual(historical.tat_termination_start_episode, 1)
        self.assertEqual(historical.early_stop_tat_threshold, 500.0)
        self.assertEqual(historical.tat_termination_grace_steps, 101)

    def test_cli_selects_historical_aliases_and_locked_termination(self):
        cases = (
            ("E", "E", TAT_SIGNAL_MARGINAL_TAT_EMA),
            ("F", "F_RAMP", TAT_SIGNAL_MARGINAL_TAT_EMA),
            ("F-NO-RAMP", "F_NO_RAMP", TAT_SIGNAL_MARGINAL_TAT_EMA),
            ("S", "S_EHYBRID", TAT_SIGNAL_TOTAL_TAT_LEVEL),
            ("S-REBALANCE", "S_REBALANCE", TAT_SIGNAL_TOTAL_TAT_LEVEL),
            ("U", "U", TAT_SIGNAL_COMPLETION_EVENT),
        )
        for argument, expected, signal in cases:
            with self.subTest(argument=argument), patch.object(
                sys,
                "argv",
                ["main_contextual.py", "--reward-version", argument],
            ):
                parsed = parse_args()
            config = ContextualRuntimeConfig(reward_version=parsed.reward_version)
            self.assertEqual(config.reward_version, expected)
            self.assertEqual(config.make_reward_config().tat_signal_mode, signal)

        reward_e = ContextualRuntimeConfig(reward_version="E")
        self.assertEqual(reward_e.reward_normalizer_freeze_steps, 10_000)
        self.assertTrue(reward_e.global_normalization_enabled)
        self.assertTrue(reward_e.local_normalization_enabled)
        self.assertEqual(reward_e.early_stop_tat_threshold, 500.0)
        self.assertEqual(reward_e.tat_above_threshold_patience, 1)

        reward_o = ContextualRuntimeConfig(reward_version="O")
        self.assertFalse(reward_o.tat_termination_enabled)
        reward_r = ContextualRuntimeConfig(reward_version="R")
        self.assertTrue(reward_r.tat_termination_enabled)
        self.assertEqual(reward_r.early_stop_tat_threshold, 200.0)
        self.assertEqual(reward_r.terminal_tat_penalty, -20.0)

    def test_cli_dispatch_mode_defaults_to_first_match_and_accepts_cost(self):
        cases = (
            ([], DISPATCH_FIRST_MATCH),
            (["--dispatch-mode", "first-match"], DISPATCH_FIRST_MATCH),
            (["--dispatch-mode", "cost"], DISPATCH_COST),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments), patch.object(
                sys, "argv", ["main_contextual.py", *arguments]
            ):
                self.assertEqual(parse_args().dispatch_mode, expected)
        with patch.object(
            sys,
            "argv",
            ["main_contextual.py", "--dispatch-mode", "unknown"],
        ), self.assertRaises(SystemExit):
            parse_args()
        with self.assertRaises(ValueError):
            ContextualRuntimeConfig(dispatch_mode="unknown")

    def test_cli_boolean_optional_flags_and_string_false_rejected(self):
        cases = (
            ([], True, True),
            (["--no-sale"], False, True),
            (["--no-lap"], True, False),
            (["--no-sale", "--no-lap"], False, False),
        )
        for arguments, sale, lap in cases:
            with self.subTest(arguments=arguments), patch.object(
                sys, "argv", ["main_contextual.py", *arguments]
            ):
                parsed = parse_args()
                self.assertIs(parsed.sale, sale)
                self.assertIs(parsed.lap, lap)
        with patch.object(
            sys, "argv", ["main_contextual.py", "--sale", "false"]
        ), self.assertRaises(SystemExit):
            parse_args()

    def test_cli_tat_confidence_can_be_enabled_or_disabled(self):
        cases = (
            ([], False),
            (["--use-tat-confidence"], True),
            (["--use-tat-cofidence"], True),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments), patch.object(
                sys, "argv", ["main_contextual.py", *arguments]
            ):
                self.assertIs(
                    parse_args().use_tat_cofidence,
                    expected,
                )

    def test_cli_experiment_termination_and_diagnostic_opt_in(self):
        with patch.object(sys, "argv", ["main_contextual.py"]):
            parsed = parse_args()
            self.assertIsNone(parsed.reward_diagnostic_dir)
            self.assertEqual(
                parsed.rail_reward_mode, RAIL_REWARD_FIXED_TAT_REFERENCE
            )
            self.assertEqual(parsed.rail_free_flow_neutral_ratio, 2.0)
            self.assertEqual(
                parsed.reward_diagnostic_windows,
                "0:1000,10000:11000,20000:21000",
            )
            self.assertEqual(parsed.early_stop_tat_threshold, 180.0)
            self.assertEqual(parsed.tat_termination_grace_steps, 0)
            self.assertEqual(parsed.tat_above_threshold_patience, 1)
            self.assertEqual(parsed.tat_termination_start_episode, 2)
        with patch.object(
            sys,
            "argv",
            [
                "main_contextual.py",
                "--reward-diagnostic-dir", "diag",
                "--reward-diagnostic-windows", "5:10",
            ],
        ):
            parsed = parse_args()
            self.assertEqual(parsed.reward_diagnostic_dir, "diag")
            self.assertEqual(parsed.reward_diagnostic_windows, "5:10")

        config = ContextualRuntimeConfig()
        self.assertIsNone(config.reward_diagnostic_dir)
        self.assertEqual(config.smooth_b_rl_weight, 0.05)
        self.assertEqual(config.reward_normalizer_freeze_steps, 30_000)
        self.assertEqual(config.tat_reference, 165.0)
        self.assertEqual(config.tat_weight, 2.3)
        self.assertFalse(config.tat_one_sided)
        self.assertEqual(config.op_weight, 0.0)
        self.assertFalse(config.use_op)
        self.assertEqual(config.backlog_weight, 0.0025)
        self.assertFalse(config.global_normalization_enabled)
        self.assertTrue(config.local_normalization_enabled)
        self.assertFalse(config.backlog_growth_enabled)
        self.assertEqual(config.backlog_growth_weight, 0.0)
        self.assertEqual(config.idle_reserve_weight, 0.0)
        self.assertEqual(config.local_oht_weight, 0.3)
        self.assertEqual(config.local_predicted_oht_weight, 0.2)
        self.assertEqual(config.local_stop_weight, 0.3)
        self.assertEqual(config.local_idle_weight, 0.1)
        self.assertEqual(config.local_capacity_weight, 0.1)
        self.assertFalse(config.local_fixed_scale_enabled)
        self.assertEqual(config.local_reward_scale, 1.0)
        self.assertEqual(config.reward_rail_tat_weight, 1.0)
        self.assertIsNone(config.reward_rail_tat_clip)
        self.assertIsNone(config.tat_raw_clip)
        self.assertEqual(config.early_stop_tat_threshold, 200.0)
        self.assertEqual(config.tat_termination_grace_steps, 10_000)
        self.assertEqual(config.tat_above_threshold_patience, 300)
        self.assertEqual(config.terminal_tat_penalty, -20.0)
        self.assertEqual(
            config.rail_reward_mode, RAIL_REWARD_FIXED_TAT_REFERENCE
        )
        self.assertEqual(config.rail_free_flow_neutral_ratio, 2.0)

    def test_cli_replay_sampling_modes_are_mutually_exclusive(self):
        cases = (
            ([], REPLAY_SAMPLING_RAIL),
            (["--replay-buffer-rail"], REPLAY_SAMPLING_RAIL),
            (["--replay-buffer-snapshot"], REPLAY_SAMPLING_SNAPSHOT),
            (
                ["--replay-buffer-random-rail"],
                REPLAY_SAMPLING_RANDOM_RAIL,
            ),
            (["--random-rail-mode"], REPLAY_SAMPLING_RANDOM_RAIL),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments), patch.object(
                sys, "argv", ["main_contextual.py", *arguments]
            ):
                self.assertEqual(parse_args().replay_sampling_mode, expected)
        with patch.object(
            sys,
            "argv",
            [
                "main_contextual.py",
                "--replay-buffer-rail",
                "--replay-buffer-snapshot",
            ],
        ), self.assertRaises(SystemExit):
            parse_args()

    def test_snapshot_runtime_config_requires_uniform_replay(self):
        config = ContextualRuntimeConfig(
            replay_sampling_mode=REPLAY_SAMPLING_SNAPSHOT,
            lap_enabled=False,
        )
        self.assertEqual(
            config.replay_sampling_mode, REPLAY_SAMPLING_SNAPSHOT
        )
        with self.assertRaises(ValueError):
            ContextualRuntimeConfig(
                replay_sampling_mode=REPLAY_SAMPLING_SNAPSHOT,
                lap_enabled=True,
            )
        random_rail = ContextualRuntimeConfig(
            replay_sampling_mode=REPLAY_SAMPLING_RANDOM_RAIL,
            lap_enabled=False,
        )
        self.assertEqual(
            random_rail.replay_sampling_mode,
            REPLAY_SAMPLING_RANDOM_RAIL,
        )
        with self.assertRaises(ValueError):
            ContextualRuntimeConfig(
                replay_sampling_mode=REPLAY_SAMPLING_RANDOM_RAIL,
                lap_enabled=True,
            )

    def test_four_variant_names_loss_modes_and_finite_updates(self):
        for flags, name in EXPECTED.items():
            sale, lap = flags
            with self.subTest(sale=sale, lap=lap):
                replay = sale_replay(lap=lap)
                config = ContextualLearnerConfig(
                    batch_size=8, action_scale=0.05,
                    sale_enabled=sale, lap_enabled=lap,
                    sale_embedding_dim=16, sale_feature_dim=16,
                    minimum_replay_env_steps=1,
                    minimum_action_enabled_env_steps=1,
                    require_normalizer_frozen=False,
                    target_update_interval=4,
                )
                self.assertEqual(config.algorithm_variant, name)
                self.assertEqual(
                    contextual_algorithm_variant(sale, lap), name
                )
                self.assertEqual(
                    config.resolved_critic_loss_mode,
                    "huber" if lap else "mse",
                )
                learner = ContextualTD7Learner(
                    replay, network_config=SMALL_NETWORK,
                    config=config, device="cpu", seed=31,
                )
                for _ in range(4):
                    result = learner.update()
                    self.assertTrue(all(
                        torch.isfinite(torch.tensor(value))
                        for value in result.diagnostics.values()
                    ))
                self.assertEqual(learner.actor_update_count, 2)
                self.assertEqual(learner.target_update_count, 1)
                self.assertTrue(
                    learner.fixed_target_q_min
                    <= learner.fixed_target_q_max
                )
                self.assertEqual(learner.sale_online is not None, sale)
                self.assertEqual(replay._priority is not None, lap)

    def test_config_is_frozen_against_mid_run_flag_mutation(self):
        config = ContextualLearnerConfig()
        with self.assertRaises(Exception):
            config.sale_enabled = False
        with self.assertRaises(Exception):
            config.lap_enabled = False


if __name__ == "__main__":
    unittest.main()
