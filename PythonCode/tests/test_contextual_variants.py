import sys
import unittest
from unittest.mock import patch

import torch

from oht_dispatching.config import DISPATCH_COST, DISPATCH_FIRST_MATCH
from oht_routing.runtime.client import ContextualRuntimeConfig
from oht_routing.runtime.config import (
    RESUME_LAUNCH_CONTROL_FIELDS,
    runtime_config_from_args,
)
from oht_routing.runtime.config_validation import make_reward_config
from oht_routing.mdp.reward.config import (
    RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
    REWARD_VERSION,
)
from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualTD7Learner,
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    contextual_algorithm_variant,
)
from main import parse_args
from test_contextual_learner import SMALL_NETWORK
from test_contextual_sale import sale_replay


EXPECTED = {
    (True, True): "contextual_td7_sale_lap_v4_recenttat",
    (False, True): "contextual_td7_no_sale_lap_v4_recenttat",
    (True, False): "contextual_td7_sale_uniform_v4_recenttat",
    (False, False): "contextual_twin_delayed_uniform_v4_recenttat",
}


class ContextualVariantTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(
            sys, "argv", ["main.py", *arguments]
        ):
            return parse_args()

    def test_cli_and_runtime_are_locked_to_reward_o(self):
        parsed = self.parse()
        self.assertNotIn("reward_version", vars(parsed))
        self.assertEqual(
            runtime_config_from_args(parsed).reward_version,
            REWARD_VERSION,
        )
        self.assertEqual(
            runtime_config_from_args(
                self.parse("--reward-version", "o")
            ).reward_version,
            "O",
        )
        with self.assertRaises(SystemExit):
            self.parse("--reward-version", "E")
        with self.assertRaisesRegex(ValueError, "only reward_version='O'"):
            ContextualRuntimeConfig(reward_version="U")

        config = ContextualRuntimeConfig()
        self.assertEqual(config.reward_version, "O")
        self.assertEqual(config.early_stop_tat_threshold, 200.0)
        self.assertEqual(config.tat_termination_grace_steps, 10_000)
        self.assertEqual(config.tat_above_threshold_patience, 300)
        self.assertEqual(config.terminal_tat_penalty, -20.0)
        reward = make_reward_config(config)
        self.assertEqual(reward.tat_weight, 4.3)
        self.assertEqual(reward.op_weight, 0.0)
        self.assertFalse(reward.use_op)
        self.assertEqual(reward.tat_window_seconds, 300.0)
        self.assertEqual(reward.backlog_weight, 0.0007)
        self.assertEqual(reward.backlog_growth_weight, 0.17)
        self.assertEqual(reward.idle_reserve_weight, 0.09)
        self.assertEqual(reward.local_predicted_oht_weight, 0.05)
        self.assertEqual(reward.local_reward_scale, 2.0)
        self.assertEqual(reward.rail_tat_weight, 30.0)
        self.assertEqual(reward.rail_tat_clip, 1.0)
        self.assertEqual(
            reward.rail_reward_mode, RAIL_REWARD_FREE_FLOW_NEUTRAL_2
        )

    def test_cli_omits_runtime_defaults_and_config_resolves_them(self):
        parsed = self.parse()
        self.assertEqual(vars(parsed), {})
        config = runtime_config_from_args(parsed)
        self.assertEqual(config.curriculum_scale_start, 0.05)
        self.assertEqual(config.curriculum_scale_end, 1.0)
        self.assertEqual(config.curriculum_end_step, 20_000)
        self.assertEqual(config.replay_capacity_env_steps, 100_000)
        self.assertEqual(config.replay_sampling_mode, REPLAY_SAMPLING_RAIL)
        self.assertEqual(config.batch_size, 1_024)
        self.assertEqual(config.warmup_steps, 10_000)
        self.assertTrue(config.terminate_on_warmup_complete)
        self.assertIsNone(config.load_state_normalizer_path)
        self.assertIsNone(config.save_state_normalizer_path)
        self.assertTrue(config.sale_enabled)
        self.assertTrue(config.lap_enabled)
        self.assertFalse(config.wandb_enabled)
        self.assertEqual(config.console_log_interval, 100)
        self.assertEqual(config.sim_end_time, 45_000)

    def test_explicit_cli_values_override_only_selected_config_fields(self):
        config = runtime_config_from_args(self.parse(
            "--curriculum-scale-start",
            "0.2",
            "--curriculum-scale-end",
            "0.8",
            "--curriculum-end-step",
            "30000",
            "--batch-size",
            "2048",
            "--no-terminate-on-warmup-complete",
            "--device",
            "cpu",
            "--console-log-interval",
            "200",
            "--sim-end-time",
            "55000",
        ))
        self.assertEqual(config.curriculum_scale_start, 0.2)
        self.assertEqual(config.curriculum_scale_end, 0.8)
        self.assertEqual(config.curriculum_end_step, 30_000)
        self.assertEqual(config.batch_size, 2_048)
        self.assertFalse(config.terminate_on_warmup_complete)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.console_log_interval, 200)
        self.assertEqual(config.sim_end_time, 55_000)
        self.assertEqual(config.warmup_steps, 10_000)
        self.assertEqual(config.replay_capacity_env_steps, 100_000)

    def test_runtime_and_resume_launch_control_contract_is_declared(self):
        self.assertIn("replay_capacity_env_steps", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("lap_enabled", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("console_log_interval", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("sim_end_time", RESUME_LAUNCH_CONTROL_FIELDS)

    def test_mode_aware_wandb_default_and_explicit_cli_overrides(self):
        self.assertFalse(ContextualRuntimeConfig().wandb_enabled)
        self.assertTrue(
            ContextualRuntimeConfig(mode="actor_inference").wandb_enabled
        )
        self.assertTrue(ContextualRuntimeConfig(
            mode="training", action_enabled=True
        ).wandb_enabled)
        self.assertTrue(runtime_config_from_args(
            self.parse("--mode", "actor_inference")
        ).wandb_enabled)
        self.assertTrue(runtime_config_from_args(self.parse(
            "--mode", "training", "--action-enabled"
        )).wandb_enabled)
        self.assertFalse(runtime_config_from_args(self.parse(
            "--mode", "actor_inference", "--no-wandb"
        )).wandb_enabled)
        self.assertTrue(
            runtime_config_from_args(self.parse("--wandb")).wandb_enabled
        )

    def test_save_data_aliases_and_runtime_validation(self):
        self.assertTrue(
            self.parse(
                "--mode", "actor_inference", "--save_data"
            ).save_data_enabled
        )
        self.assertTrue(
            self.parse(
                "--mode", "actor_inference", "--save-data"
            ).save_data_enabled
        )
        with self.assertRaisesRegex(ValueError, "only supported in actor_inference"):
            ContextualRuntimeConfig(
                mode="training",
                action_enabled=True,
                save_data_enabled=True,
            )

    def test_path_cli_destinations_and_runtime_string_conversion(self):
        reuse = self.parse(
            "--load-state-normalizer",
            "normalizers/state_n.npz",
            "--save-state-normalizer",
            "normalizers/state_copy.npz",
        )
        self.assertIn("load_state_normalizer_path", vars(reuse))
        self.assertIn("save_state_normalizer_path", vars(reuse))
        self.assertNotIn("load_state_normalizer", vars(reuse))
        self.assertNotIn("save_state_normalizer", vars(reuse))
        config = runtime_config_from_args(reuse)
        self.assertEqual(
            config.load_state_normalizer_path,
            "normalizers\\state_n.npz",
        )
        self.assertEqual(
            config.save_state_normalizer_path,
            "normalizers\\state_copy.npz",
        )
        self.assertEqual(config.effective_warmup_steps, 0)
        self.assertTrue(config.state_normalizer_warmup_bypass)

    @patch("oht_routing.runtime.config.read_contextual_runtime_config")
    def test_resume_keeps_current_lap_and_capacity_controls(self, read_config):
        read_config.return_value = (
            {
                "lap_enabled": True,
                "replay_capacity_env_steps": 10_000,
                "warmup_steps": 321,
                "console_log_interval": 7,
                "sim_end_time": 9,
            },
            True,
        )
        config = runtime_config_from_args(self.parse(
            "--resume-checkpoint",
            "checkpoint.pt",
            "--resume-inference-until-replay-full",
            "--resume-deterministic-first-episode",
            "--no-lap",
            "--replay-capacity-env-steps",
            "100000",
            "--console-log-interval",
            "200",
            "--sim-end-time",
            "55000",
        ))
        self.assertEqual(config.resume_checkpoint_path, "checkpoint.pt")
        self.assertTrue(config.resume_inference_until_replay_full)
        self.assertTrue(config.resume_deterministic_first_episode)
        self.assertFalse(config.lap_enabled)
        self.assertEqual(config.replay_capacity_env_steps, 100_000)
        self.assertEqual(config.warmup_steps, 321)
        self.assertEqual(config.console_log_interval, 200)
        self.assertEqual(config.sim_end_time, 55_000)
        self.assertTrue(config.state_normalizer_warmup_bypass)
        read_config.assert_called_once_with("checkpoint.pt")

    def test_cli_diagnostics_are_opt_in(self):
        parsed = self.parse()
        self.assertNotIn("reward_diagnostic_dir", vars(parsed))
        self.assertNotIn("reward_diagnostic_windows", vars(parsed))
        config = runtime_config_from_args(parsed)
        self.assertIsNone(config.reward_diagnostic_dir)
        self.assertEqual(
            config.reward_diagnostic_windows,
            "0:1000,10000:11000,20000:21000",
        )
        opted_in = self.parse(
            "--reward-diagnostic-dir",
            "diag",
            "--reward-diagnostic-windows",
            "5:10",
        )
        config = runtime_config_from_args(opted_in)
        self.assertEqual(config.reward_diagnostic_dir, "diag")
        self.assertEqual(config.reward_diagnostic_windows, "5:10")

    def test_cli_dispatch_mode_defaults_to_first_match_and_accepts_cost(self):
        cases = (
            ((), DISPATCH_FIRST_MATCH),
            (("--dispatch-mode", "first-match"), DISPATCH_FIRST_MATCH),
            (("--dispatch-mode", "cost"), DISPATCH_COST),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                config = runtime_config_from_args(self.parse(*arguments))
                self.assertEqual(config.dispatch_mode, expected)
        with self.assertRaises(SystemExit):
            self.parse("--dispatch-mode", "unknown")
        with self.assertRaises(ValueError):
            ContextualRuntimeConfig(dispatch_mode="unknown")

    def test_cli_boolean_optional_flags_and_string_false_rejected(self):
        cases = (
            ((), True, True),
            (("--no-sale",), False, True),
            (("--no-lap",), True, False),
            (("--no-sale", "--no-lap"), False, False),
        )
        for arguments, sale, lap in cases:
            with self.subTest(arguments=arguments):
                parsed = self.parse(*arguments)
                config = runtime_config_from_args(parsed)
                self.assertIs(config.sale_enabled, sale)
                self.assertIs(config.lap_enabled, lap)
        with self.assertRaises(SystemExit):
            self.parse("--sale", "false")

    def test_cli_replay_sampling_modes_are_mutually_exclusive(self):
        cases = (
            ((), REPLAY_SAMPLING_RAIL),
            (("--replay-buffer-rail",), REPLAY_SAMPLING_RAIL),
            (
                ("--replay-buffer-snapshot", "--no-lap"),
                REPLAY_SAMPLING_SNAPSHOT,
            ),
            (
                ("--replay-buffer-random-rail", "--no-lap"),
                REPLAY_SAMPLING_RANDOM_RAIL,
            ),
            (
                ("--random-rail-mode", "--no-lap"),
                REPLAY_SAMPLING_RANDOM_RAIL,
            ),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                parsed = self.parse(*arguments)
                self.assertEqual(
                    runtime_config_from_args(parsed).replay_sampling_mode,
                    expected,
                )
        with self.assertRaises(SystemExit):
            self.parse(
                "--replay-buffer-rail", "--replay-buffer-snapshot"
            )

    def test_snapshot_runtime_config_requires_uniform_replay(self):
        for mode in (
            REPLAY_SAMPLING_SNAPSHOT,
            REPLAY_SAMPLING_RANDOM_RAIL,
        ):
            with self.subTest(mode=mode):
                config = ContextualRuntimeConfig(
                    replay_sampling_mode=mode, lap_enabled=False
                )
                self.assertEqual(config.replay_sampling_mode, mode)
                with self.assertRaises(ValueError):
                    ContextualRuntimeConfig(
                        replay_sampling_mode=mode, lap_enabled=True
                    )

    def test_four_variant_names_loss_modes_and_finite_updates(self):
        for flags, name in EXPECTED.items():
            sale, lap = flags
            with self.subTest(sale=sale, lap=lap):
                replay = sale_replay(lap=lap)
                config = ContextualLearnerConfig(
                    batch_size=8,
                    action_scale=0.05,
                    sale_enabled=sale,
                    lap_enabled=lap,
                    sale_embedding_dim=16,
                    sale_feature_dim=16,
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
                    replay,
                    network_config=SMALL_NETWORK,
                    config=config,
                    device="cpu",
                    seed=31,
                )
                for _ in range(4):
                    result = learner.update()
                    self.assertTrue(all(
                        torch.isfinite(torch.tensor(value))
                        for value in result.diagnostics.values()
                    ))
                self.assertEqual(learner.actor_update_count, 2)
                self.assertEqual(learner.target_update_count, 1)
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
