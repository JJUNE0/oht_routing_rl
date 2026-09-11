"""Regress the native GUI's empty connect -> delayed batch model reset."""
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simulator.client import PClient
from oht_routing.runtime.server import _run_session


class PacketSocket:
    def __init__(self, packets):
        self.packets = list(packets)
        self.sent = []

    def settimeout(self, value):
        pass

    def sendall(self, message):
        self.sent.append(bytes(message))

    def recv(self, size):
        if not self.packets:
            return b""
        value = self.packets.pop(0)
        if isinstance(value, Exception):
            raise value
        if len(value) > size:
            self.packets.insert(0, value[size:])
        return value[:size]


def header(rails):
    data = bytearray(39)
    data[4:8] = (64).to_bytes(4, "big")
    data[8:11] = rails.to_bytes(3, "big")
    return bytes(data)


class GuiHandshakeTests(unittest.TestCase):
    def test_empty_connect_waits_over_180_seconds_then_loads_model_reset(self):
        rail = bytearray(64)
        rail[0:2] = (1).to_bytes(2, "big")
        rail[2:5] = (101).to_bytes(3, "big")
        rail[5] = 1
        rail[7:10] = (1000).to_bytes(3, "big")
        peer = PacketSocket([
            header(0), bytes(64), bytes(16),
            *[socket.timeout() for _ in range(4)],
            b"\x02", header(1), bytes(64), bytes(rail), bytes(16),
        ])
        with patch.object(PClient, "WriteAdminLog"), patch("builtins.print"):
            client = PClient(peer, sim_end_time=2000)
            self.assertEqual(client.RAILLINE_DIC, {})
            self.assertEqual(client.RecieveSimulationStandardData(), 2)
        self.assertEqual(client.RAILINE_COUNT, 1)
        self.assertEqual(client.RAILLINE_DIC[1].SimID, 101)
        self.assertEqual(peer.sent.count((2000).to_bytes(3, "big")), 2)
        self.assertEqual(peer.packets, [])

    def receive_client(self, packets):
        client = object.__new__(PClient)
        client.client_socket = PacketSocket(packets)
        client.TOTAL_BYTES_READ = 0
        client.secData = {}
        client.LAST_V = 0
        return client

    def test_partial_packet_timeout_still_fails_even_when_idle_allowed(self):
        client = self.receive_client([b"a", *[socket.timeout() for _ in range(3)]])
        with patch.object(client, "WriteAdminLog"), patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "180s"):
                client.RecieveMessage(2, allow_idle=True)

    def test_active_episode_command_timeout_is_still_bounded(self):
        client = self.receive_client([socket.timeout() for _ in range(3)])
        with patch.object(client, "WriteAdminLog"), patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "180s"):
                client.RecieveEndSim()

    def test_disconnect_never_returns_a_zero_or_partial_packet(self):
        for packets in ([b""], [b"a", b""]):
            with self.subTest(packets=packets):
                with self.assertRaises(ConnectionError):
                    self.receive_client(packets).RecieveMessage(2)

    def test_normal_run_without_model_never_calls_the_policy(self):
        pclient = Mock(RAILINE_COUNT=0)
        pclient.RecieveSimulationStandardData.return_value = 0
        client = SimpleNamespace(
            config=SimpleNamespace(sim_end_time=2000, console_log_interval=100),
            on_new_connection=Mock(),
        )
        with patch("oht_routing.runtime.server.PClient.PClient", return_value=pclient), \
             patch("oht_routing.runtime.server.handle_command") as handle:
            with self.assertRaisesRegex(ConnectionError, "MODEL NOT LOADED"):
                _run_session(object(), ("127.0.0.1", 1000), 9102, client)
        handle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
