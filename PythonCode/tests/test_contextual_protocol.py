import json
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import PClient
from main_contextual import SmokeReporter, handle_command


class DummySocket:
    def settimeout(self, value):
        self.timeout = value


class ContextualProtocolTests(unittest.TestCase):
    def test_smoke_reporter_keeps_bounded_rows_and_writes_periodically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/smoke.json"
            reporter = SmokeReporter(path, "training", write_interval=10)
            for step in range(25):
                reporter.record_tick({
                    "runtime/total_algorithm_ms": float(step),
                    "runtime/send_cost_ms": 1.0,
                    "runtime/nonfinite_count": 0.0,
                })
            self.assertFalse(hasattr(reporter, "rows"))
            self.assertEqual(len(reporter.totals), 25)
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["tick_count"], 20)
            reporter.record_reset()
            with open(path, encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertEqual(payload["tick_count"], 25)
            self.assertEqual(payload["episode_reset_count"], 1)

    def test_pclient_sends_custom_end_time_once_per_handshake(self):
        noops = (
            "SendConnectMessage",
            "RecieveInitializeMessage",
            "RecieveInitializeMessage2",
            "SendStartPythonStartTime",
            "SendDijkstraUpdateTime",
            "SendReroutingUpdateTime",
            "RecieveSetPara",
        )
        with ExitStack() as stack:
            for name in noops:
                stack.enter_context(
                    patch.object(PClient.PClient, name, return_value=None)
                )
            send_end = stack.enter_context(
                patch.object(PClient.PClient, "SendEndTime", return_value=None)
            )
            client = PClient.PClient(DummySocket(), sim_end_time=12_345)
            send_end.assert_called_once_with(12_345)

            send_end.reset_mock()
            stack.enter_context(
                patch.object(client, "RecieveEndSim", return_value=2)
            )
            self.assertEqual(client.RecieveSimulationStandardData(), 2)
            send_end.assert_called_once_with(12_345)

    def test_command_two_does_not_append_duplicate_end_time_bytes(self):
        pclient = SimpleNamespace(SendEndTime=Mock())
        client = SimpleNamespace(Reset=Mock())
        reporter = SimpleNamespace(record_reset=Mock())
        handle_command(
            2,
            pclient,
            client,
            reporter=reporter,
            sim_end_time=12_345,
        )
        pclient.SendEndTime.assert_not_called()
        client.Reset.assert_called_once_with(pclient)
        reporter.record_reset.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
