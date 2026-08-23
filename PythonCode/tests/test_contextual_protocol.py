import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simulator import client as PClient
from simulator.oht import OHTState
from main import handle_command
from oht_routing.runtime.config import ContextualRuntimeConfig
from oht_routing.runtime.server import _run_session


class DummySocket:
    def settimeout(self, value):
        self.timeout = value


class ContextualProtocolTests(unittest.TestCase):
    def test_oht_state_contract_and_packet_tat_entries_are_preserved(self):
        self.assertEqual(int(OHTState.IDLE), 0)
        self.assertEqual(int(OHTState.MOVE_TO_LOAD), 2)
        self.assertEqual(int(OHTState.LOADING), 3)
        self.assertEqual(int(OHTState.MOVE_TO_UNLOAD), 4)
        self.assertEqual(int(OHTState.UNLOADING), 5)

        oht = SimpleNamespace(CmdCompleteTat={})
        first = SimpleNamespace(CmdID=10)
        second = SimpleNamespace(CmdID=11)
        PClient.PClient._store_oht_command_tat(oht, 10, first)
        PClient.PClient._store_oht_command_tat(oht, 11, second)
        self.assertEqual(oht.CmdCompleteTat, {10: first, 11: second})

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
        client = SimpleNamespace(
            config=ContextualRuntimeConfig(sim_end_time=12_345),
            Reset=Mock(),
        )
        handle_command(2, pclient, client)
        pclient.SendEndTime.assert_not_called()
        client.Reset.assert_called_once_with(pclient)

    @patch("oht_routing.runtime.protocol.send_active_data")
    def test_command_zero_uses_config_console_interval(self, send_active_data):
        pclient = SimpleNamespace(WriteAdminLog=Mock())
        client = SimpleNamespace(
            config=ContextualRuntimeConfig(console_log_interval=3),
            total_steps=6,
        )

        handle_command(0, pclient, client)
        pclient.WriteAdminLog.assert_called_once_with(
            "Contextual SendAndReceiveRailLineCost."
        )
        send_active_data.assert_called_once_with(pclient, client)

        client.total_steps = 7
        handle_command(0, pclient, client)
        self.assertEqual(pclient.WriteAdminLog.call_count, 1)
        self.assertEqual(send_active_data.call_count, 2)

    def test_runtime_config_validates_server_controls(self):
        config = ContextualRuntimeConfig()
        self.assertEqual(config.console_log_interval, 100)
        self.assertEqual(config.sim_end_time, 45_000)
        for name, invalid in (
            ("console_log_interval", 0),
            ("console_log_interval", 1.5),
            ("sim_end_time", -1),
            ("sim_end_time", True),
        ):
            with self.subTest(name=name, invalid=invalid):
                with self.assertRaisesRegex(ValueError, "must be"):
                    ContextualRuntimeConfig(**{name: invalid})

    @patch("oht_routing.runtime.server.handle_command")
    @patch("oht_routing.runtime.server.PClient.PClient")
    def test_session_uses_runtime_config_server_controls(
        self, pclient_type, dispatch_command
    ):
        connection = object()
        pclient = pclient_type.return_value
        pclient.RecieveSimulationStandardData.side_effect = [0, 99]
        pclient.PeekPending.return_value = b""
        client = SimpleNamespace(
            config=ContextualRuntimeConfig(
                console_log_interval=2,
                sim_end_time=12_345,
            ),
            on_new_connection=Mock(),
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected v=99"):
            _run_session(connection, ("127.0.0.1", 1234), 9100, client)

        pclient_type.assert_called_once_with(
            connection, sim_end_time=12_345
        )
        dispatch_command.assert_called_once_with(0, pclient, client)

    def test_assign_command_reader_advances_to_the_next_fixed_buffer(self):
        client = object.__new__(PClient.PClient)
        client.BUFFER_SIZE = 32
        client.JOB_DIC = {}
        first = bytearray(client.BUFFER_SIZE)
        second = bytearray(client.BUFFER_SIZE)
        client.SetByteHexa_2Legnth(2, first, 0)

        def write_job(buffer, index, job_id, carriers, areas):
            client.SetByteHexa_3Legnth(job_id, buffer, index)
            index += 3
            client.SetByteHexa_2Legnth(9, buffer, index)
            index += 2
            client.SetByteHexa_2Legnth(10, buffer, index)
            index += 2
            buffer[index] = 1
            index += 1
            client.SetByteHexa_2Legnth(0, buffer, index)
            index += 2
            buffer[index] = len(carriers)
            index += 1
            for carrier in carriers:
                buffer[index] = carrier
                index += 1
            buffer[index] = len(areas)
            index += 1
            for area in areas:
                buffer[index] = area
                index += 1
            return index

        self.assertEqual(write_job(first, 2, 100, [1, 2, 3], [1, 2]), 19)
        write_job(second, 0, 200, [1], [1])
        client.RecieveMessage = Mock(
            side_effect=[bytes(first), bytes(second)]
        )

        jobs = client.GetAssignCommand()

        self.assertEqual([job.ID for job in jobs], [100, 200])
        self.assertEqual(jobs[1].FromNode, 9)
        self.assertEqual(jobs[1].ToNode, 10)
        self.assertEqual(client.RecieveMessage.call_count, 2)

    def test_assign_response_clears_each_continuation_buffer(self):
        client = object.__new__(PClient.PClient)
        client.BUFFER_SIZE = 32
        client.JOB_DIC = {
            job_id: SimpleNamespace(RouteList=[job_id, job_id + 1, job_id + 2])
            for job_id in range(1, 6)
        }
        client.SendMessage = Mock()

        client.SendAssignOht({job_id: job_id + 100 for job_id in range(1, 6)})

        sent = [bytes(call.args[0]) for call in client.SendMessage.call_args_list]
        self.assertEqual(len(sent), 3)
        self.assertEqual(client.GetBase10Value_2(sent[0], 0), 5)
        self.assertTrue(all(value == 0 for value in sent[1][26:]))

    def test_assign_response_sends_explicit_pickup_path(self):
        client = object.__new__(PClient.PClient)
        client.BUFFER_SIZE = 32
        client.JOB_DIC = {
            7: SimpleNamespace(RouteList=[99]),
        }
        client.SendMessage = Mock()

        client.SendAssignOht({
            7: {
                "oht_id": 42,
                "pickup_path": [1, 2, 9],
            },
        })

        sent = bytes(client.SendMessage.call_args.args[0])
        self.assertEqual(client.GetBase10Value_2(sent, 0), 1)
        self.assertEqual(client.GetBase10Value_3(sent, 2), 7)
        self.assertEqual(client.GetBase10Value_2(sent, 5), 42)
        self.assertEqual(client.GetBase10Value_2(sent, 7), 3)
        self.assertEqual(
            [client.GetBase10Value_2(sent, index) for index in (9, 11, 13)],
            [1, 2, 9],
        )


if __name__ == "__main__":
    unittest.main()
