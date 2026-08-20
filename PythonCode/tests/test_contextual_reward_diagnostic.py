import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oht_routing.mdp.reward.builder import ContextualRewardBuilder, ContextualRewardConfig
from oht_routing.utils.reward_diagnostic import (
    RewardDiagnosticWriter,
    diagnostic_window_name,
    parse_diagnostic_windows,
)
from oht_routing.version import CONTEXTUAL_VERSION
from test_contextual_reward import completion, rail_pass, rail_tat_client


class ContextualRewardDiagnosticTests(unittest.TestCase):
    def test_window_boundaries_are_start_inclusive_end_exclusive(self):
        windows = parse_diagnostic_windows(None)
        expected = {
            0: "00000_01000",
            999: "00000_01000",
            1_000: None,
            9_999: None,
            10_000: "10000_11000",
            10_999: "10000_11000",
            11_000: None,
            20_000: "20000_21000",
            20_999: "20000_21000",
            21_000: None,
        }
        for step, name in expected.items():
            self.assertEqual(diagnostic_window_name(step, windows), name)

    def test_jsonl_is_append_only_strict_json_and_outside_window_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RewardDiagnosticWriter(directory, "0:2")
            self.assertIsNone(writer.append_step(2, {"value": 1.0}))
            self.assertFalse(list(Path(directory).glob("*.jsonl")))
            path = writer.append_step(0, {"value": np.nan})
            writer.append_step(1, {"value": 2.0})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            records = [json.loads(line) for line in lines]
            self.assertIsNone(records[0]["value"])
            self.assertEqual(records[1]["value"], 2.0)
            self.assertTrue(all(
                item["version"] == CONTEXTUAL_VERSION
                for item in records
            ))

    def test_wandb_only_writer_keeps_bounded_free_flow_summary_without_files(self):
        writer = RewardDiagnosticWriter(None, "0:10", cycle_buffer_size=2)
        for step, ratio in enumerate((1.5, 1.7, 2.0)):
            writer.append_cycle(step, {
                "rail_reward_mode": "free_flow_neutral_2",
                "rail_free_flow_neutral_ratio": 2.0,
                "reward_applied": True,
                "route_free_flow_available": True,
                "route_free_flow_ratio": ratio,
                "cycle_rail_reward_raw": 2.0 - ratio,
                "cycle_is_positive_reward": ratio < 2.0,
                "cycle_is_negative_reward": ratio > 2.0,
                "cycle_is_zero_reward": ratio == 2.0,
                "rail_attributions": [{
                    "raw_reward": 2.0 - ratio,
                    "weighted_preclip": 2.0 - ratio,
                    "postclip": 2.0 - ratio,
                }],
                "clip_assignment_ratio": 0.0,
                "clip_removed_ratio": 0.0,
            })
        summary = writer.cycle_summary(3)
        self.assertEqual(summary["reward/rail/mode"], "free_flow_neutral_2")
        self.assertEqual(summary["reward/rail/cycle_count"], 2.0)
        self.assertEqual(summary["reward/rail/free_flow_neutral_ratio"], 2.0)
        self.assertAlmostEqual(
            summary["reward/rail/route_ratio_p50"], 1.85
        )
        self.assertGreater(
            summary["reward/rail/nonzero_reward_abs_mean"], 0.0
        )

    def test_cycle_free_flow_counts_same_rail_occurrences_without_reward_change(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        with tempfile.TemporaryDirectory() as directory:
            step = {"value": 0}
            writer = RewardDiagnosticWriter(directory, "0:10")
            builder = ContextualRewardBuilder(
                topology,
                ContextualRewardConfig(),
                global_step_provider=lambda: step["value"],
                reward_diagnostic_writer=writer,
            )
            client = rail_tat_client(state=0)
            client.SimTime = 0.0
            client.RAILLINE_DIC = {
                1: SimpleNamespace(DistancePerVelocity=1.5)
            }
            builder._rail_tat_raw(client, env_step=0)
            oht = client.OHT_DIC[7]
            step["value"] = 1
            oht.State = 2
            oht.JobID = oht.DispatchedCommand = 1
            oht.PassTimes = [rail_pass(1, 2, 2.0)]
            oht.CmdCompleteTat = {1: completion(1, 10.0, 10.0)}
            builder._rail_tat_raw(client, env_step=1)
            step["value"] = 2
            oht.State = 4
            oht.PassTimes = [rail_pass(1, 4, 3.0)]
            oht.CmdCompleteTat = {1: completion(1, 100.0, 100.0)}
            builder._rail_tat_raw(client, env_step=2)
            step["value"] = 3
            oht.State = 5
            oht.PassTimes = []
            oht.CmdCompleteTat = {1: completion(1, 200.0, 200.0)}
            builder._rail_tat_raw(client, env_step=3)
            step["value"] = 4
            oht.State = 0
            oht.CmdCompleteTat = {}
            raw = builder._rail_tat_raw(client, env_step=4)

            self.assertLess(raw[0], 0.0)
            path = Path(directory) / "rail_cycle_window_00000_00010.jsonl"
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(record["route_free_flow_available"])
            self.assertEqual(record["route_free_flow_time"], 3.0)
            self.assertEqual(record["completed_rail_occurrence_count"], 2)
            self.assertEqual(record["same_rail_revisit_count"], 1)
            self.assertTrue(record["cycle_fully_observed"])
            self.assertEqual(record["path_occurrence_count"], 2)
            self.assertEqual(len(record["rail_occurrences"]), 2)
            self.assertAlmostEqual(record["elapsed_share_sum"], 1.0)
            self.assertLess(record["raw_attribution_sum_error"], 1e-12)
            self.assertLess(record["weighted_scaling_error_max"], 1e-12)
            self.assertLess(record["clip_contract_error_max"], 1e-12)

    def test_missing_and_invalid_free_flow_are_unavailable(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1, 2], dtype=np.int64)
        )
        with tempfile.TemporaryDirectory() as directory:
            writer = RewardDiagnosticWriter(directory, "0:10")
            builder = ContextualRewardBuilder(
                topology,
                ContextualRewardConfig(),
                reward_diagnostic_writer=writer,
            )
            client = rail_tat_client(state=0)
            client.SimTime = 0.0
            client.RAILLINE_DIC = {
                1: SimpleNamespace(DistancePerVelocity=0.0),
            }
            builder._rail_tat_raw(client, env_step=0)
            oht = client.OHT_DIC[7]
            oht.State = 2
            oht.JobID = oht.DispatchedCommand = 1
            oht.PassTimes = [
                rail_pass(1, 2, 1.0), rail_pass(2, 2, 1.0)
            ]
            oht.CmdCompleteTat = {1: completion(1, 10.0, 10.0)}
            builder._rail_tat_raw(client, env_step=1)
            oht.State = 5
            oht.PassTimes = []
            oht.CmdCompleteTat = {1: completion(1, 200.0, 200.0)}
            builder._rail_tat_raw(client, env_step=2)
            oht.State = 0
            oht.CmdCompleteTat = {}
            builder._rail_tat_raw(client, env_step=3)
            path = Path(directory) / "rail_cycle_window_00000_00010.jsonl"
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(record["route_free_flow_available"])
            self.assertEqual(record["free_flow_invalid_count"], 1)
            self.assertEqual(record["free_flow_missing_count"], 1)
            self.assertIsNone(record["route_delay_ratio"])

    def test_repeated_inflight_packets_and_completion_record_one_occurrence(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        with tempfile.TemporaryDirectory() as directory:
            writer = RewardDiagnosticWriter(directory, "0:10")
            builder = ContextualRewardBuilder(
                topology,
                ContextualRewardConfig(),
                reward_diagnostic_writer=writer,
            )
            client = rail_tat_client(state=0)
            client.SimTime = 0.0
            client.RAILLINE_DIC = {
                1: SimpleNamespace(DistancePerVelocity=1.25)
            }
            builder._rail_tat_raw(client, env_step=0)
            oht = client.OHT_DIC[7]
            oht.State = 2
            oht.JobID = oht.DispatchedCommand = 1
            oht.PassTimes = []
            oht.NotPassTimes = [rail_pass(1, 2, 1.0)]
            oht.CmdCompleteTat = {1: completion(1, 10.0, 10.0)}
            builder._rail_tat_raw(client, env_step=1)
            oht.NotPassTimes = [rail_pass(1, 2, 2.0)]
            builder._rail_tat_raw(client, env_step=2)
            oht.PassTimes = [rail_pass(1, 2, 3.0)]
            oht.NotPassTimes = []
            builder._rail_tat_raw(client, env_step=3)
            oht.State = 5
            oht.PassTimes = []
            oht.CmdCompleteTat = {1: completion(1, 200.0, 200.0)}
            builder._rail_tat_raw(client, env_step=4)
            oht.State = 0
            oht.CmdCompleteTat = {}
            builder._rail_tat_raw(client, env_step=5)

            path = Path(directory) / "rail_cycle_window_00000_00010.jsonl"
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["completed_rail_occurrence_count"], 1)
            self.assertEqual(record["route_free_flow_time"], 1.25)
            self.assertEqual(len(record["rail_occurrences"]), 1)

    def test_free_flow_reward_tracks_occurrences_without_json_writer(self):
        topology = SimpleNamespace(
            controlled_rail_ids=np.asarray([1], dtype=np.int64)
        )
        builder = ContextualRewardBuilder(
            topology,
            ContextualRewardConfig(),
        )
        client = rail_tat_client(state=0)
        client.RAILLINE_DIC = {
            1: SimpleNamespace(DistancePerVelocity=1.5)
        }
        builder._rail_tat_raw(client, env_step=0)
        oht = client.OHT_DIC[7]
        oht.State = 2
        oht.JobID = oht.DispatchedCommand = 1
        oht.PassTimes = [rail_pass(1, 2, 2.0)]
        oht.CmdCompleteTat = {1: completion(1, 10.0, 10.0)}
        builder._rail_tat_raw(client, env_step=1)
        oht.State = 4
        oht.PassTimes = [rail_pass(1, 4, 3.0)]
        builder._rail_tat_raw(client, env_step=2)
        oht.State = 5
        oht.PassTimes = []
        builder._rail_tat_raw(client, env_step=3)
        oht.State = 0
        oht.CmdCompleteTat = {}
        raw = builder._rail_tat_raw(client, env_step=4)
        np.testing.assert_allclose(raw, [-1.0 / 3.0])
        self.assertEqual(builder._last_rail_tat_cycle_count, 1)
        self.assertEqual(
            builder._last_rail_tat_controlled_assignment_count, 1
        )

    def test_reward_config_rejects_negative_rail_weight(self):
        with self.assertRaisesRegex(ValueError, "rail_tat_weight"):
            ContextualRewardConfig(rail_tat_weight=-1.0)


if __name__ == "__main__":
    unittest.main()
