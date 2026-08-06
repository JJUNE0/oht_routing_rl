import sys
import unittest
from unittest.mock import patch

import torch

from cocel_rl.algorithms.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualTD7Learner,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    contextual_algorithm_variant,
)
from main_contextual import parse_args
from ClientAlgorithm_contextual import ContextualRuntimeConfig
from test_contextual_learner import SMALL_NETWORK
from test_contextual_sale import sale_replay


EXPECTED = {
    (True, True): "contextual_td7_sale_lap_v2",
    (False, True): "contextual_td7_no_sale_lap_v2",
    (True, False): "contextual_td7_sale_uniform_v2",
    (False, False): "contextual_twin_delayed_uniform_v2",
}


class ContextualVariantTests(unittest.TestCase):
    def test_cli_boolean_optional_flags_and_string_false_rejected(self):
        cases = (
            ([], True, False),
            (["--no-sale"], False, False),
            (["--no-lap"], True, False),
            (["--no-sale", "--no-lap"], False, False),
            (["--algorithm", "td7", "--replay-buffer-rail"], True, True),
            (["--replay-buffer-rail", "--no-lap"], True, False),
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

    def test_cli_defaults_to_one_complete_factory_snapshot(self):
        with patch.object(sys, "argv", ["main_contextual.py"]):
            parsed = parse_args()
        self.assertEqual(parsed.replay_sampling_mode, REPLAY_SAMPLING_SNAPSHOT)
        self.assertEqual(parsed.batch_size, 1)
        self.assertFalse(parsed.lap)

        with patch.object(
            sys,
            "argv",
            ["main_contextual.py", "--replay-buffer-rail"],
        ):
            parsed = parse_args()
        self.assertEqual(parsed.replay_sampling_mode, REPLAY_SAMPLING_RAIL)
        self.assertEqual(parsed.batch_size, 1_024)

        runtime_config = ContextualRuntimeConfig()
        self.assertEqual(
            runtime_config.replay_sampling_mode,
            REPLAY_SAMPLING_SNAPSHOT,
        )
        self.assertEqual(runtime_config.batch_size, 1)
        self.assertFalse(runtime_config.lap_enabled)

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
