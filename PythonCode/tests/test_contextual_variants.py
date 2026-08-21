import sys
import unittest
from unittest.mock import patch

import torch

from oht_dispatching.config import DISPATCH_COST, DISPATCH_FIRST_MATCH
from oht_routing.runtime.client import ContextualRuntimeConfig
from oht_routing.runtime.config import (
    RESUME_LAUNCH_CONTROL_FIELDS,
    restore_checkpoint_runtime_config,
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
    (True, True): "contextual_td7_sale_lap_v3_prevact",
    (False, True): "contextual_td7_no_sale_lap_v3_prevact",
    (True, False): "contextual_td7_sale_uniform_v3_prevact",
    (False, False): "contextual_twin_delayed_uniform_v3_prevact",
}


class ContextualVariantTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(
            sys, "argv", ["main.py", *arguments]
        ):
            return parse_args()

    def test_cli_and_runtime_are_locked_to_reward_n(self):
        parsed = self.parse()
        self.assertEqual(parsed.reward_version, REWARD_VERSION)
        self.assertEqual(self.parse("--reward-version", "n").reward_version, "N")
        with self.assertRaises(SystemExit):
            self.parse("--reward-version", "E")
        with self.assertRaisesRegex(ValueError, "only reward_version='N'"):
            ContextualRuntimeConfig(reward_version="U")

        config = ContextualRuntimeConfig()
        self.assertEqual(config.reward_version, "N")
        self.assertEqual(config.early_stop_tat_threshold, 200.0)
        self.assertEqual(config.tat_termination_grace_steps, 10_000)
        self.assertEqual(config.tat_above_threshold_patience, 300)
        self.assertEqual(config.terminal_tat_penalty, -20.0)
        reward = make_reward_config(config)
        self.assertEqual(reward.tat_weight, 11.0)
        self.assertEqual(reward.op_weight, 4.0)
        self.assertEqual(reward.backlog_weight, 0.0004)
        self.assertEqual(reward.backlog_growth_weight, 0.16)
        self.assertEqual(reward.idle_reserve_weight, 0.20)
        self.assertEqual(reward.local_predicted_oht_weight, 0.075)
        self.assertEqual(reward.local_reward_scale, 2.0)
        self.assertEqual(reward.rail_tat_weight, 30.0)
        self.assertEqual(reward.rail_tat_clip, 1.0)
        self.assertEqual(
            reward.rail_reward_mode, RAIL_REWARD_FREE_FLOW_NEUTRAL_2
        )

    def test_cli_keeps_runtime_and_resume_controls(self):
        parsed = self.parse()
        self.assertEqual(parsed.curriculum_scale_start, 1.0)
        self.assertEqual(parsed.curriculum_scale_end, 1.0)
        self.assertEqual(parsed.replay_capacity_env_steps, 100_000)
        self.assertIn("replay_capacity_env_steps", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("lap_enabled", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertEqual(parsed.batch_size, 1_024)
        self.assertEqual(parsed.warmup_steps, 10_000)
        self.assertTrue(parsed.terminate_on_warmup_complete)
        self.assertIsNone(parsed.load_state_normalizer)
        self.assertIsNone(parsed.save_state_normalizer)

        self.assertFalse(parsed.wandb)
        self.assertTrue(self.parse("--mode", "training").wandb)
        self.assertTrue(self.parse("--mode", "actor_inference").wandb)
        self.assertFalse(
            self.parse("--mode", "actor_inference", "--no-wandb").wandb
        )
        self.assertTrue(self.parse("--wandb").wandb)
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

        no_boundary = self.parse("--no-terminate-on-warmup-complete")
        self.assertFalse(no_boundary.terminate_on_warmup_complete)
        full_refill = self.parse(
            "--resume-checkpoint",
            "checkpoint.pt",
            "--resume-inference-until-replay-full",
            "--batch-size",
            "2048",
        )
        self.assertTrue(full_refill.resume_inference_until_replay_full)
        self.assertEqual(full_refill.batch_size, 2_048)
        deterministic = self.parse(
            "--resume-checkpoint",
            "checkpoint.pt",
            "--resume-deterministic-first-episode",
        )
        self.assertTrue(deterministic.resume_deterministic_first_episode)

        reuse = self.parse(
            "--load-state-normalizer",
            "normalizers/state_n.npz",
            "--save-state-normalizer",
            "normalizers/state_copy.npz",
        )
        config = ContextualRuntimeConfig(
            load_state_normalizer_path=str(reuse.load_state_normalizer),
            save_state_normalizer_path=str(reuse.save_state_normalizer),
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
            },
            True,
        )
        restored = restore_checkpoint_runtime_config(
            {
                "lap_enabled": False,
                "replay_capacity_env_steps": 100_000,
                "warmup_steps": 10_000,
            },
            "checkpoint.pt",
        )
        self.assertFalse(restored["lap_enabled"])
        self.assertEqual(restored["replay_capacity_env_steps"], 100_000)
        self.assertEqual(restored["warmup_steps"], 321)

    def test_cli_diagnostics_are_opt_in(self):
        parsed = self.parse()
        self.assertIsNone(parsed.reward_diagnostic_dir)
        self.assertEqual(
            parsed.reward_diagnostic_windows,
            "0:1000,10000:11000,20000:21000",
        )
        opted_in = self.parse(
            "--reward-diagnostic-dir",
            "diag",
            "--reward-diagnostic-windows",
            "5:10",
        )
        self.assertEqual(opted_in.reward_diagnostic_dir, "diag")
        self.assertEqual(opted_in.reward_diagnostic_windows, "5:10")

    def test_cli_dispatch_mode_defaults_to_first_match_and_accepts_cost(self):
        cases = (
            ((), DISPATCH_FIRST_MATCH),
            (("--dispatch-mode", "first-match"), DISPATCH_FIRST_MATCH),
            (("--dispatch-mode", "cost"), DISPATCH_COST),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.parse(*arguments).dispatch_mode, expected)
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
                self.assertIs(parsed.sale, sale)
                self.assertIs(parsed.lap, lap)
        with self.assertRaises(SystemExit):
            self.parse("--sale", "false")

    def test_cli_replay_sampling_modes_are_mutually_exclusive(self):
        cases = (
            ((), REPLAY_SAMPLING_RAIL),
            (("--replay-buffer-rail",), REPLAY_SAMPLING_RAIL),
            (("--replay-buffer-snapshot",), REPLAY_SAMPLING_SNAPSHOT),
            (("--replay-buffer-random-rail",), REPLAY_SAMPLING_RANDOM_RAIL),
            (("--random-rail-mode",), REPLAY_SAMPLING_RANDOM_RAIL),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    self.parse(*arguments).replay_sampling_mode, expected
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
