import sys
import unittest
from unittest.mock import patch

import torch

from oht_dispatching.config import DISPATCH_COST, DISPATCH_FIRST_MATCH
from oht_routing.runtime.client import ClientAlgorithm, ContextualRuntimeConfig
from oht_routing.runtime.config import (
    RESUME_LAUNCH_CONTROL_FIELDS,
    restore_checkpoint_runtime_config,
    runtime_config_from_args,
)
from oht_routing.runtime.config_validation import make_reward_config
from oht_routing.mdp.action import REGION_B_RL
from oht_routing.mdp.reward.config import (
    RAIL_REWARD_FREE_FLOW_NEUTRAL_1_7,
    REWARD_VERSION,
)
from oht_routing.version import CONTEXTUAL_VERSION
from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualTD7Learner,
    REPLAY_EVICTION_FIFO,
    REPLAY_EVICTION_RANDOM,
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    contextual_algorithm_variant,
)
from oht_routing.utils.wandb_logging import runtime_exp_meta
from main import parse_args
from test_contextual_learner import SMALL_NETWORK
from test_contextual_sale import sale_replay


EXPECTED = {
    (True, True): "contextual_td7_sale_lap_v6_compact_actor_critic_tat",
    (False, True): "contextual_td7_no_sale_lap_v6_compact_actor_critic_tat",
    (True, False): "contextual_td7_sale_uniform_v6_compact_actor_critic_tat",
    (False, False): (
        "contextual_twin_delayed_uniform_v6_compact_actor_critic_tat"
    ),
}


class ContextualVariantTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(
            sys, "argv", ["main.py", *arguments]
        ):
            return parse_args()

    def test_cli_and_runtime_are_locked_to_reward_q(self):
        parsed = self.parse()
        self.assertNotIn("reward_version", vars(parsed))
        self.assertEqual(
            runtime_config_from_args(parsed).reward_version,
            REWARD_VERSION,
        )
        self.assertEqual(
            runtime_config_from_args(
                self.parse("--reward-version", "q")
            ).reward_version,
            "Q",
        )
        for retired in ("O", "P"):
            with self.assertRaises(SystemExit):
                self.parse("--reward-version", retired)
        with self.assertRaisesRegex(ValueError, r"only reward_version in \('Q', 'N'\)"):
            ContextualRuntimeConfig(reward_version="O")

        config = ContextualRuntimeConfig()
        self.assertEqual(config.reward_version, "Q")
        self.assertEqual(config.action_mode, REGION_B_RL)
        self.assertEqual(config.rl_cost_lambda, 0.5)
        self.assertEqual(config.early_stop_tat_threshold, 200.0)
        self.assertEqual(config.tat_termination_grace_steps, 10_000)
        self.assertEqual(config.tat_above_threshold_patience, 300)
        self.assertEqual(config.terminal_tat_penalty, -20.0)
        reward = make_reward_config(config)
        self.assertEqual(reward.tat_weight, 4.0)
        self.assertEqual(reward.op_weight, 0.0)
        self.assertFalse(reward.use_op)
        self.assertEqual(reward.tat_window_seconds, 300.0)
        self.assertEqual(reward.contract.tat_window_seconds, 0.0)
        self.assertEqual(
            reward.contract.tat_signal_description,
            "one_sided_cumulative_total_tat_penalty",
        )
        self.assertEqual(reward.backlog_weight, 0.0007)
        self.assertEqual(reward.backlog_growth_weight, 0.17)
        self.assertEqual(reward.idle_reserve_weight, 0.09)
        self.assertEqual(reward.local_predicted_oht_weight, 0.01)
        self.assertEqual(reward.local_stop_weight, 0.30)
        self.assertEqual(reward.local_density_weight, 5.5)
        self.assertEqual(reward.local_reward_scale, 2.0)
        self.assertEqual(reward.rail_tat_weight, 660.0)
        self.assertEqual(reward.rail_tat_clip, 22.0)
        self.assertEqual(
            reward.rail_reward_mode, RAIL_REWARD_FREE_FLOW_NEUTRAL_1_7
        )

    def test_cli_omits_runtime_defaults_and_config_resolves_them(self):
        parsed = self.parse()
        self.assertEqual(vars(parsed), {})
        config = runtime_config_from_args(parsed)
        self.assertEqual(config.curriculum_scale_start, 0.05)
        self.assertEqual(config.curriculum_scale_end, 1.0)
        self.assertEqual(config.curriculum_end_step, 20_000)
        self.assertEqual(config.replay_capacity_env_steps, 100_000)
        self.assertEqual(config.replay_eviction_mode, REPLAY_EVICTION_FIFO)
        self.assertEqual(config.replay_sampling_mode, REPLAY_SAMPLING_RAIL)
        self.assertEqual(config.batch_size, 1_024)
        self.assertEqual(config.warmup_steps, 10_000)
        self.assertTrue(config.terminate_on_warmup_complete)
        self.assertIsNone(config.load_state_normalizer_path)
        self.assertIsNone(config.save_state_normalizer_path)
        self.assertTrue(config.sale_enabled)
        self.assertTrue(config.lap_enabled)
        self.assertFalse(config.use_attention)
        self.assertFalse(config.wandb_enabled)
        self.assertEqual(config.console_log_interval, 100)
        self.assertEqual(config.num_sim, 1)
        self.assertIsNone(config.sim_ports)
        self.assertIsNone(config.stage)
        self.assertEqual(config.sim_end_time, 45_000)
        self.assertIsNone(config.load_stage1_policy_path)
        self.assertEqual(config.periodic_checkpoint_interval, 5_000)
        self.assertEqual(config.resume_warmstart_steps, 0)
        self.assertEqual(config.exploration_noise_std, 0.10)
        self.assertEqual(config.exploration_noise_final_std, 0.02)
        self.assertEqual(config.action_mode, REGION_B_RL)
        self.assertEqual(config.rl_cost_lambda, 0.5)

        override = runtime_config_from_args(
            self.parse("--rl-cost-lambda", "0.25")
        )
        self.assertEqual(override.rl_cost_lambda, 0.25)
        for invalid in (-0.1, 1.1, float("nan")):
            with self.subTest(rl_cost_lambda=invalid):
                with self.assertRaisesRegex(ValueError, "rl_cost_lambda"):
                    ContextualRuntimeConfig(rl_cost_lambda=invalid)

    def test_explicit_cli_values_survive_checkpoint_resume(self):
        """A typed option must beat the value stored in the checkpoint."""
        saved = {
            "reward_version": REWARD_VERSION,
            "exploration_noise_std": 0.05,
            "exploration_noise_final_std": 0.05,
            "warmup_steps": 7_777,
        }
        with patch(
            "oht_routing.runtime.config.read_contextual_runtime_config",
            return_value=(saved, True),
        ):
            # Untouched options still come back from the checkpoint.
            restored = restore_checkpoint_runtime_config(
                {"reward_version": REWARD_VERSION,
                 "exploration_noise_std": 0.1,
                 "warmup_steps": 256},
                "stage2.pt",
            )
            self.assertEqual(restored["exploration_noise_std"], 0.05)
            self.assertEqual(restored["warmup_steps"], 7_777)

            # Explicitly typed options win.
            restored = restore_checkpoint_runtime_config(
                {"reward_version": REWARD_VERSION,
                 "exploration_noise_std": 0.0,
                 "exploration_noise_final_std": 0.0,
                 "warmup_steps": 256},
                "stage2.pt",
                explicit_fields=(
                    "exploration_noise_std", "exploration_noise_final_std"
                ),
            )
            self.assertEqual(restored["exploration_noise_std"], 0.0)
            self.assertEqual(restored["exploration_noise_final_std"], 0.0)
            # and everything else is still restored
            self.assertEqual(restored["warmup_steps"], 7_777)

    def test_resume_deterministic_episodes_rejects_bad_combinations(self):
        with self.assertRaisesRegex(ValueError, "requires checkpoint resume"):
            ContextualRuntimeConfig(
                mode="training", action_enabled=True,
                resume_deterministic_episodes=2,
            )
        with self.assertRaisesRegex(ValueError, "use only one"):
            ContextualRuntimeConfig(
                mode="training", action_enabled=True,
                resume_checkpoint_path="stage2.pt",
                resume_deterministic_episodes=2,
                resume_deterministic_first_episode=True,
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            ContextualRuntimeConfig(
                mode="training", action_enabled=True,
                resume_checkpoint_path="stage2.pt",
                resume_deterministic_episodes=-1,
            )

    def test_stage_two_actor_inference_keeps_the_frozen_prefix(self):
        """Evaluating a Stage 2 policy must reproduce its training contract."""
        config = ContextualRuntimeConfig(
            mode="actor_inference",
            action_enabled=True,
            stage=2,
            load_stage1_policy_path="stage1.pt",
            resume_checkpoint_path="stage2.pt",
        )
        self.assertEqual(config.mode, "actor_inference")
        self.assertEqual(config.stage, 2)
        self.assertEqual(config.sim_end_time, 45_000)
        self.assertFalse(config.stage1_policy_warm_start)

        # Without the Stage 2 policy there is nothing to evaluate.
        with self.assertRaisesRegex(
            ValueError, "requires --resume-checkpoint"
        ):
            ContextualRuntimeConfig(
                mode="actor_inference",
                action_enabled=True,
                stage=2,
                load_stage1_policy_path="stage1.pt",
            )
        # baseline_only still has no Stage 2 contract.
        with self.assertRaisesRegex(
            ValueError, "training or actor_inference"
        ):
            ContextualRuntimeConfig(
                mode="baseline_only",
                action_enabled=True,
                stage=2,
                load_stage1_policy_path="stage1.pt",
                resume_checkpoint_path="stage2.pt",
            )

    def test_stage_one_preserves_identity_and_resolves_end_time(self):
        parsed = self.parse("--stage", "1")
        self.assertEqual(vars(parsed), {"stage": 1})
        config = runtime_config_from_args(parsed)
        self.assertEqual(config.stage, 1)
        self.assertEqual(config.sim_end_time, 2_000)

    def test_stage_two_requires_and_accepts_frozen_stage1_policy(self):
        for option in ("--load-stage1-policy", "--load_stage1_policy"):
            with self.subTest(option=option):
                parsed = self.parse(
                    "--mode", "training", "--action-enabled",
                    "--stage", "2", option, "stage1.pt",
                )
                config = runtime_config_from_args(parsed)
                self.assertEqual(config.stage, 2)
                self.assertEqual(config.sim_end_time, 45_000)
                self.assertEqual(config.load_stage1_policy_path, "stage1.pt")
                self.assertEqual(config.effective_warmup_steps, 0)
                self.assertTrue(config.state_normalizer_warmup_bypass)
                self.assertEqual(config.exploration_noise_std, 0.05)
                self.assertEqual(config.exploration_noise_final_std, 0.05)

        with self.assertRaisesRegex(
            ValueError, "stage 2 requires --load-stage1-policy"
        ):
            runtime_config_from_args(self.parse(
                "--mode", "training", "--action-enabled", "--stage", "2"
            ))
        with self.assertRaisesRegex(ValueError, "requires stage 2"):
            runtime_config_from_args(self.parse(
                "--mode", "training", "--action-enabled",
                "--load_stage1_policy", "stage1.pt",
            ))

        explicit = runtime_config_from_args(self.parse(
            "--mode", "training", "--action-enabled",
            "--stage", "2", "--load-stage1-policy", "stage1.pt",
            "--exploration-noise-std", "0.08",
            "--exploration-noise-final-std", "0.03",
        ))
        self.assertEqual(explicit.exploration_noise_std, 0.08)
        self.assertEqual(explicit.exploration_noise_final_std, 0.03)

    def test_stage_rejects_unknown_values_and_explicit_end_time(self):
        for arguments in (
            ("--stage", "0"),
            ("--stage", "3"),
            ("--stage", "first"),
            ("--stage", "1", "--sim-end-time", "3000"),
            ("--sim-end-time", "3000", "--stage", "1"),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    self.parse(*arguments)

    def test_removed_episode_burnin_flag_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--episode-burnin-steps", "100")

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
            "--periodic-checkpoint-interval",
            "2000",
            "--use-attention",
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
        self.assertEqual(config.periodic_checkpoint_interval, 2_000)
        self.assertTrue(config.use_attention)
        self.assertEqual(config.sim_end_time, 55_000)
        self.assertEqual(config.warmup_steps, 10_000)
        self.assertEqual(config.replay_capacity_env_steps, 100_000)

    def test_runtime_and_resume_launch_control_contract_is_declared(self):
        self.assertIn("replay_capacity_env_steps", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("replay_eviction_mode", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("lap_enabled", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("reward_version", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("console_log_interval", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("num_sim", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("sim_ports", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("sim_end_time", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("stage", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("load_stage1_policy_path", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn(
            "periodic_checkpoint_interval", RESUME_LAUNCH_CONTROL_FIELDS
        )
        self.assertIn("resume_warmstart_steps", RESUME_LAUNCH_CONTROL_FIELDS)
        self.assertIn("use_attention", RESUME_LAUNCH_CONTROL_FIELDS)
        # The ensemble size and the aggregation rule define the saved critics,
        # so a resumed run must take them from the checkpoint rather than from
        # whatever the current defaults happen to be.
        for field in (
            "num_critics",
            "critic_target_mode",
            "uboc_beta",
            "actor_q_aggregation",
        ):
            self.assertNotIn(field, RESUME_LAUNCH_CONTROL_FIELDS)

    def test_cli_accepts_single_and_multiple_simulator_ports(self):
        single = self.parse("--port", "9100")
        self.assertEqual(
            vars(single), {"sim_ports": [9_100]}
        )
        single_config = runtime_config_from_args(single)
        self.assertEqual(single_config.num_sim, 1)
        self.assertEqual(single_config.sim_ports, (9_100,))

        for option in ("--port", "--ports"):
            with self.subTest(option=option):
                parsed = self.parse(
                    "--mode", "training",
                    "--action-enabled",
                    "--stage", "2",
                    "--load-stage1-policy", "stage1.pt",
                    "--num-sim", "4", option,
                    "9100", "9101", "9102", "9103",
                )
                self.assertEqual(parsed.num_sim, 4)
                self.assertEqual(
                    parsed.sim_ports, [9_100, 9_101, 9_102, 9_103]
                )
                config = runtime_config_from_args(parsed)
                self.assertEqual(config.num_sim, 4)
                self.assertEqual(
                    config.sim_ports, (9_100, 9_101, 9_102, 9_103)
                )

    def test_multi_simulator_runtime_is_stage2_policy_bootstrap_only(self):
        endpoints = {
            "num_sim": 2,
            "sim_ports": (9_100, 9_101),
        }
        with self.assertRaisesRegex(
            ValueError, "requires Stage 2 training"
        ):
            ContextualRuntimeConfig(**endpoints)
        with self.assertRaisesRegex(
            ValueError, "distributed full-state checkpoint resume"
        ):
            ContextualRuntimeConfig(
                **endpoints,
                mode="training",
                action_enabled=True,
                stage=2,
                load_stage1_policy_path="stage1.pt",
                resume_checkpoint_path="stage2.pt",
            )

    def test_runtime_rejects_invalid_multi_simulator_ports(self):
        cases = (
            ({"num_sim": 0}, "num_sim must be a positive integer"),
            ({"num_sim": True}, "num_sim must be a positive integer"),
            (
                {"num_sim": 2},
                "num_sim > 1 requires explicit sim_ports",
            ),
            (
                {"num_sim": 2, "sim_ports": (9_100,)},
                "sim_ports count must equal num_sim",
            ),
            (
                {"num_sim": 2, "sim_ports": (9_100, 9_100)},
                "sim_ports must contain unique ports",
            ),
            (
                {"num_sim": 2, "sim_ports": (0, 9_101)},
                "sim_ports values must be integers",
            ),
            (
                {"num_sim": 2, "sim_ports": (9_100, 65_536)},
                "sim_ports values must be integers",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    ContextualRuntimeConfig(**kwargs)

    def test_periodic_checkpoint_interval_must_be_positive(self):
        with self.assertRaisesRegex(
            ValueError, "training counts/intervals must be positive"
        ):
            runtime_config_from_args(self.parse(
                "--periodic-checkpoint-interval", "0"
            ))

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
                "reward_version": "Q",
                "lap_enabled": True,
                "replay_capacity_env_steps": 10_000,
                "replay_eviction_mode": REPLAY_EVICTION_FIFO,
                "warmup_steps": 321,
                "console_log_interval": 7,
                "sim_end_time": 9,
                "periodic_checkpoint_interval": 5_000,
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
            "--replay-eviction-mode",
            "random",
            "--console-log-interval",
            "200",
            "--periodic-checkpoint-interval",
            "2000",
            "--sim-end-time",
            "55000",
        ))
        self.assertEqual(config.resume_checkpoint_path, "checkpoint.pt")
        self.assertTrue(config.resume_inference_until_replay_full)
        self.assertTrue(config.resume_deterministic_first_episode)
        self.assertFalse(config.lap_enabled)
        self.assertEqual(config.replay_capacity_env_steps, 100_000)
        self.assertEqual(config.replay_eviction_mode, REPLAY_EVICTION_RANDOM)
        self.assertEqual(config.warmup_steps, 321)
        self.assertEqual(config.console_log_interval, 200)
        self.assertEqual(config.periodic_checkpoint_interval, 2_000)
        self.assertEqual(config.sim_end_time, 55_000)
        self.assertTrue(config.state_normalizer_warmup_bypass)
        read_config.assert_called_once_with(
            "checkpoint.pt", expected_reward_version="Q"
        )

    @patch("oht_routing.runtime.config.read_contextual_runtime_config")
    def test_resume_keeps_current_simulator_launch_controls(self, read_config):
        read_config.return_value = (
            {
                "reward_version": "Q",
                "num_sim": 8,
                "sim_ports": tuple(range(9_200, 9_208)),
            },
            True,
        )
        config = runtime_config_from_args(self.parse(
            "--resume-checkpoint", "checkpoint.pt",
            "--num-sim", "1",
            "--ports", "9100",
        ))

        self.assertEqual(config.num_sim, 1)
        self.assertEqual(config.sim_ports, (9_100,))
        read_config.assert_called_once_with(
            "checkpoint.pt", expected_reward_version="Q"
        )

    @patch("oht_routing.runtime.config.read_contextual_runtime_config")
    def test_resume_warmstart_steps_stay_launch_controlled(self, read_config):
        read_config.return_value = (
            {
                "reward_version": "Q",
                "resume_warmstart_steps": 0,
            },
            True,
        )
        config = runtime_config_from_args(self.parse(
            "--mode",
            "training",
            "--action-enabled",
            "--resume-checkpoint",
            "checkpoint.pt",
            "--resume-warmstart-steps",
            "100",
        ))
        self.assertEqual(config.resume_warmstart_steps, 100)
        read_config.assert_called_once_with(
            "checkpoint.pt", expected_reward_version="Q"
        )

    @patch("oht_routing.runtime.config.read_contextual_runtime_config")
    def test_retired_episode_burnin_config_is_ignored_on_resume(
        self, read_config
    ):
        read_config.return_value = (
            {
                "reward_version": "Q",
                "episode_burnin_steps": 100,
            },
            True,
        )
        config = runtime_config_from_args(self.parse(
            "--resume-checkpoint", "checkpoint.pt"
        ))
        self.assertFalse(hasattr(config, "episode_burnin_steps"))

    @patch("oht_routing.runtime.config.read_contextual_runtime_config")
    def test_attention_mode_stays_launch_controlled_on_resume(self, read_config):
        read_config.return_value = (
            {"reward_version": "Q", "use_attention": True},
            True,
        )
        default_flat = runtime_config_from_args(self.parse(
            "--resume-checkpoint", "checkpoint.pt"
        ))
        self.assertFalse(default_flat.use_attention)

        opted_in = runtime_config_from_args(self.parse(
            "--resume-checkpoint", "checkpoint.pt", "--use-attention"
        ))
        self.assertTrue(opted_in.use_attention)

    @patch("oht_routing.runtime.config.read_contextual_runtime_config")
    def test_stage_one_survives_checkpoint_runtime_restore(self, read_config):
        read_config.return_value = (
            {
                "reward_version": "Q",
                "sim_end_time": 9_000,
                "warmup_steps": 321,
            },
            True,
        )
        config = runtime_config_from_args(self.parse(
            "--resume-checkpoint",
            "checkpoint.pt",
            "--stage",
            "1",
        ))
        self.assertEqual(config.sim_end_time, 2_000)
        self.assertEqual(config.warmup_steps, 321)
        self.assertTrue(config.state_normalizer_warmup_bypass)
        read_config.assert_called_once_with(
            "checkpoint.pt", expected_reward_version="Q"
        )

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
        self.assertTrue(
            runtime_config_from_args(self.parse("--use-attention")).use_attention
        )
        with self.assertRaises(SystemExit):
            self.parse("--use-attention", "false")

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

    def test_cli_replay_eviction_modes_and_stack_contract(self):
        cases = (
            ((), REPLAY_EVICTION_FIFO),
            (("--replay-eviction-mode", "fifo"), REPLAY_EVICTION_FIFO),
            (("--replay-eviction-mode", "random"), REPLAY_EVICTION_RANDOM),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                config = runtime_config_from_args(self.parse(*arguments))
                self.assertEqual(config.replay_eviction_mode, expected)

        with self.assertRaises(SystemExit):
            self.parse("--replay-eviction-mode", "unknown")
        with self.assertRaisesRegex(ValueError, "replay_eviction_mode"):
            ContextualRuntimeConfig(replay_eviction_mode="unknown")
        with self.assertRaisesRegex(ValueError, "requires num_stacks=1"):
            ContextualRuntimeConfig(
                replay_eviction_mode=REPLAY_EVICTION_RANDOM,
                num_stacks=2,
            )

        random_meta = runtime_exp_meta(ContextualRuntimeConfig(
            replay_eviction_mode=REPLAY_EVICTION_RANDOM
        ))
        self.assertEqual(
            random_meta["replay_eviction_mode"], REPLAY_EVICTION_RANDOM
        )
        self.assertIn("evictrandom", random_meta["note"])
        self.assertIn("evict_random", random_meta["replay"])
        self.assertIn("replay_eviction=random", random_meta["description"])

    def test_replay_eviction_mode_changes_runtime_and_checkpoint_identity(self):
        fifo_runtime = object.__new__(ClientAlgorithm)
        fifo_runtime.config = ContextualRuntimeConfig(
            replay_eviction_mode=REPLAY_EVICTION_FIFO
        )
        random_runtime = object.__new__(ClientAlgorithm)
        random_runtime.config = ContextualRuntimeConfig(
            replay_eviction_mode=REPLAY_EVICTION_RANDOM
        )

        self.assertNotEqual(
            fifo_runtime.runtime_variant, random_runtime.runtime_variant
        )
        self.assertNotEqual(
            fifo_runtime.checkpoint_variant, random_runtime.checkpoint_variant
        )
        self.assertIn("evict_fifo", fifo_runtime.runtime_variant)
        self.assertIn("evict_random", random_runtime.runtime_variant)
        self.assertIn("efifo", fifo_runtime.checkpoint_variant)
        self.assertIn("erandom", random_runtime.checkpoint_variant)

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
                self.assertEqual(
                    config.algorithm_variant_for(5), f"{name}_uboc5"
                )
                self.assertEqual(
                    contextual_algorithm_variant(sale, lap, "uboc", 5),
                    f"{name}_uboc5",
                )
                self.assertEqual(
                    contextual_algorithm_variant(sale, lap, "cdq", 2),
                    f"{name}_cdq2",
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
