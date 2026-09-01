import socket
import threading
import unittest
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import main as main_module
from simulator import client as PClient
from simulator.oht import OHTState
from main import handle_command
from oht_routing.runtime.config import ContextualRuntimeConfig
from oht_routing.runtime.server import (
    _configure_listener_socket,
    _run_session,
    serve_contextual,
)
from oht_routing.runtime.distributed.server import DistributedServerGroup
from oht_routing.runtime.distributed.coordinator import (
    CentralTrainingCoordinator,
    _AdvanceRequest,
    _RegistrationRequest,
    _TransitionRequest,
)
from oht_routing.runtime.distributed.types import CoordinatorSnapshot


class DummySocket:
    def settimeout(self, value):
        self.timeout = value


class ContextualProtocolTests(unittest.TestCase):
    def test_main_routes_single_and_multi_simulator_launches(self):
        parsed = object()
        with (
            patch.object(main_module, "_configure_console_encoding"),
            patch.object(main_module, "parse_args", return_value=parsed),
            patch.object(main_module, "seed_everything") as seed,
            patch.object(main_module, "print_runtime_summary") as summary,
            patch.object(main_module, "ClientAlgorithm") as client_type,
            patch.object(main_module, "serve_contextual") as serve_single,
            patch(
                "oht_routing.runtime.distributed.server.serve_distributed"
            ) as serve_multi,
        ):
            single = SimpleNamespace(
                seed=11, num_sim=1, sim_ports=(9_123,)
            )
            with patch.object(
                main_module,
                "runtime_config_from_args",
                return_value=single,
            ):
                main_module.main()

            seed.assert_called_once_with(11)
            client_type.assert_called_once_with(single)
            summary.assert_called_once_with(client_type.return_value)
            serve_single.assert_called_once_with(
                client_type.return_value, port=9_123
            )
            serve_multi.assert_not_called()

            seed.reset_mock()
            client_type.reset_mock()
            summary.reset_mock()
            serve_single.reset_mock()
            multi = SimpleNamespace(
                seed=17,
                num_sim=4,
                sim_ports=(9_100, 9_101, 9_102, 9_103),
            )
            with patch.object(
                main_module,
                "runtime_config_from_args",
                return_value=multi,
            ):
                main_module.main()

            seed.assert_called_once_with(17)
            serve_multi.assert_called_once_with(multi)
            client_type.assert_not_called()
            summary.assert_not_called()
            serve_single.assert_not_called()

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

    def test_pclient_session_state_is_isolated_before_first_reset(self):
        noops = (
            "SendConnectMessage",
            "RecieveInitializeMessage",
            "RecieveInitializeMessage2",
            "SendStartPythonStartTime",
            "SendDijkstraUpdateTime",
            "SendReroutingUpdateTime",
            "SendEndTime",
            "RecieveSetPara",
        )
        with ExitStack() as stack:
            for name in noops:
                stack.enter_context(
                    patch.object(PClient.PClient, name, return_value=None)
                )
            first = PClient.PClient(DummySocket())
            second = PClient.PClient(DummySocket())

        for name in (
            "RAILLINECOST_DIC",
            "RAILINE_SIM_ID_DIC",
            "RAILLINE_DIC",
            "ReRoueDic",
            "OHT_DIC",
            "JOB_DIC",
            "secData",
        ):
            with self.subTest(name=name):
                self.assertIsNot(getattr(first, name), getattr(second, name))
        first.JOB_DIC[7] = object()
        self.assertNotIn(7, second.JOB_DIC)
        self.assertNotEqual(first.file_path, second.file_path)
        self.assertNotEqual(first.file_path_TCP, second.file_path_TCP)

    def test_listener_socket_uses_platform_safe_ownership(self):
        server = Mock()

        _configure_listener_socket(server)

        option = (
            socket.SO_EXCLUSIVEADDRUSE
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE")
            else socket.SO_REUSEADDR
        )
        server.setsockopt.assert_called_once_with(
            socket.SOL_SOCKET, option, 1
        )

    def test_distributed_prebind_closes_every_socket_on_failure(self):
        first = Mock()
        second = Mock()
        second.bind.side_effect = OSError("occupied")
        coordinator = SimpleNamespace(
            stop_event=threading.Event(),
            request_stop=Mock(),
        )
        group = DistributedServerGroup(
            (9_100, 9_101), (object(), object()), coordinator
        )

        with patch(
            "oht_routing.runtime.distributed.server.socket.socket",
            side_effect=(first, second),
        ):
            with self.assertRaisesRegex(OSError, "occupied"):
                group._bind_all()

        first.close.assert_called_once_with()
        second.close.assert_called_once_with()
        self.assertEqual(group._listeners, [])

    def test_distributed_registration_warm_starts_stage2_policy_once(self):
        config = ContextualRuntimeConfig(
            mode="training",
            action_enabled=True,
            stage=2,
            load_stage1_policy_path="stage1.pt",
            num_sim=2,
            sim_ports=(9_100, 9_101),
            device="cpu",
            wandb_enabled=False,
            sale_enabled=False,
            lap_enabled=False,
        )
        coordinator = CentralTrainingCoordinator(config)
        topology = SimpleNamespace(
            topology_hash="topology-hash",
            mapping_hash="mapping-hash",
            all_rail_ids=np.asarray([10, 11], dtype=np.int64),
            controlled_rail_ids=np.asarray([10], dtype=np.int64),
            incoming_neighbor_ids=np.zeros((1, 15), dtype=np.int64),
        )

        def normalizer_builder():
            normalizers = []
            for name in ("local", "global", "critic"):
                normalizer = Mock()
                normalizer.state_dict.return_value = {
                    "name": name,
                    "frozen": True,
                }
                normalizers.append(normalizer)
            return SimpleNamespace(
                local_normalizer=normalizers[0],
                global_normalizer=normalizers[1],
                critic_normalizer=normalizers[2],
            )

        central_builder = normalizer_builder()
        worker_builders = (normalizer_builder(), normalizer_builder())
        replay = SimpleNamespace(size_env_steps=0)
        learner = SimpleNamespace(learner_update_count=0)
        frozen_stage1 = SimpleNamespace(
            action_mode=config.action_mode,
            applied_action_scale=0.73,
            checkpoint_path=Path("stage1.pt"),
            checkpoint_sha256="a" * 64,
        )
        responses = (Future(), Future())

        with (
            patch(
                "oht_routing.runtime.distributed.coordinator."
                "ContextualObservationBuilder",
                return_value=central_builder,
            ) as observation_type,
            patch(
                "oht_routing.runtime.distributed.coordinator."
                "ContextualStepReplayBuffer",
                return_value=replay,
            ) as replay_type,
            patch(
                "oht_routing.runtime.distributed.coordinator."
                "ContextualTD7Learner",
                return_value=learner,
            ) as learner_type,
            patch(
                "oht_routing.runtime.distributed.coordinator."
                "load_frozen_contextual_policy",
                return_value=frozen_stage1,
            ) as load_stage1,
            patch(
                "oht_routing.runtime.distributed.coordinator."
                "ContextualRewardBuilder",
                return_value=SimpleNamespace(reward_steps=0),
            ),
        ):
            for worker_id in range(2):
                coordinator._handle_registration(_RegistrationRequest(
                    worker_id=worker_id,
                    topology=topology,
                    observation_builder=worker_builders[worker_id],
                    cache_path=Path(f"cache_{worker_id}.npz"),
                    response=responses[worker_id],
                ))

        observation_type.assert_called_once()
        replay_type.assert_called_once()
        learner_type.assert_called_once()
        load_stage1.assert_called_once_with(
            "stage1.pt",
            learner,
            observation_builder=central_builder,
            expected_reward_version=config.reward_version,
            initialize_fresh_learner_policy=True,
        )
        self.assertTrue(all(response.done() for response in responses))
        self.assertTrue(all(
            response.result().stage1_policy.checkpoint_sha256 == "a" * 64
            for response in responses
        ))

    def test_distributed_tick_commit_has_exact_aggregate_cadence(self):
        coordinator = object.__new__(CentralTrainingCoordinator)
        coordinator.num_workers = 4
        coordinator._worker_generations = [0] * 4
        coordinator._state_lock = threading.Lock()
        coordinator._transition_advanced_ticks = {}
        coordinator._worker_states = {}
        coordinator._schedule_step = 0
        coordinator._action_enabled_env_steps = 0
        coordinator._latest_diagnostics = {}
        coordinator.config = SimpleNamespace(
            learn_every_env_steps=4,
            updates_per_env_step=1,
        )
        coordinator.replay = SimpleNamespace(
            push_transition=Mock(),
            diagnostics=Mock(return_value={}),
        )
        coordinator.reward_builder = SimpleNamespace(reward_steps=0)
        coordinator._training_gate_open = Mock(return_value=True)
        coordinator._update_learner = Mock(
            return_value={"learner/critic_loss": 1.0}
        )
        coordinator._maybe_checkpoint = Mock()
        coordinator._publish_snapshot = Mock(
            side_effect=lambda: CoordinatorSnapshot(
                schedule_step=coordinator._schedule_step
            )
        )
        transition = SimpleNamespace(
            env_step=2_000,
            reward=SimpleNamespace(
                total=np.asarray([1.0, 2.0], dtype=np.float32)
            ),
            action=np.asarray([-0.1, 0.1], dtype=np.float32),
            applied_action=np.asarray([-0.05, 0.05], dtype=np.float32),
            done=False,
        )

        transition_futures = []
        for worker_id in range(4):
            response = Future()
            transition_futures.append(response)
            coordinator._handle_transition(_TransitionRequest(
                worker_id=worker_id,
                generation=0,
                transition=transition,
                action_scale=0.2,
                response=response,
            ))

        self.assertEqual(coordinator._schedule_step, 4)
        self.assertEqual(coordinator._update_learner.call_count, 1)
        self.assertTrue(all(future.done() for future in transition_futures))

        for worker_id in range(4):
            response = Future()
            coordinator._handle_advance(_AdvanceRequest(
                worker_id=worker_id,
                generation=0,
                episode_id=1,
                episode_step=2_001,
                action_scale=0.2,
                response=response,
            ))
            self.assertTrue(response.done())

        self.assertEqual(coordinator._schedule_step, 4)
        self.assertEqual(coordinator._update_learner.call_count, 1)

        # A first Stage 2 tick has no previous transition, but still consumes
        # exactly one aggregate clock/update-cadence slot.
        response = Future()
        coordinator._handle_advance(_AdvanceRequest(
            worker_id=0,
            generation=0,
            episode_id=2,
            episode_step=2_000,
            action_scale=0.2,
            response=response,
        ))
        self.assertEqual(coordinator._schedule_step, 5)
        self.assertEqual(coordinator._update_learner.call_count, 2)
        self.assertEqual(coordinator._maybe_checkpoint.call_count, 5)

        # A reconnect may occur after the transition commit but before its
        # matching advance RPC. The old marker is already counted and must not
        # contaminate the new generation's first tick.
        coordinator._transition_advanced_ticks[1] = (0, 2_002)
        coordinator._worker_generations[1] = 1
        response = Future()
        coordinator._handle_advance(_AdvanceRequest(
            worker_id=1,
            generation=1,
            episode_id=2,
            episode_step=2_000,
            action_scale=0.2,
            response=response,
        ))
        self.assertEqual(coordinator._schedule_step, 6)
        self.assertNotIn(1, coordinator._transition_advanced_ticks)

    def test_worker_failure_bypasses_saturated_normal_event_queue(self):
        config = ContextualRuntimeConfig(
            mode="training",
            action_enabled=True,
            stage=2,
            load_stage1_policy_path="stage1.pt",
            num_sim=2,
            sim_ports=(9_100, 9_101),
            device="cpu",
            wandb_enabled=False,
        )
        coordinator = CentralTrainingCoordinator(config)
        while not coordinator._event_queue.full():
            coordinator._event_queue.put_nowait(object())

        coordinator.report_worker_failure(1, RuntimeError("boom"))

        self.assertEqual(coordinator.run_once(timeout=0.0), "failure")
        self.assertTrue(coordinator.stop_event.is_set())
        self.assertTrue(coordinator.snapshot().stopped)
        self.assertIn("collector 1 failed", str(coordinator._failure))
        coordinator.close(status="failed")

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

    @patch("oht_routing.runtime.server._accept_sessions")
    @patch("oht_routing.runtime.server.socket.socket")
    @patch("oht_routing.runtime.server.read_port")
    def test_server_uses_wpconfig_port_when_explicit_port_is_absent(
        self, read_port, socket_type, accept_sessions
    ):
        read_port.return_value = 9_100
        server = socket_type.return_value
        client = SimpleNamespace(
            wandb_logger=SimpleNamespace(
                finish_failed=Mock(),
                finish_interrupted=Mock(),
                finish_success=Mock(),
            ),
            training_failed=False,
            close_environment_capture=Mock(),
        )

        serve_contextual(client)

        read_port.assert_called_once_with()
        server.bind.assert_called_once_with(("127.0.0.1", 9_100))
        accept_sessions.assert_called_once_with(server, 9_100, client)
        server.close.assert_called_once_with()

    @patch("oht_routing.runtime.server._accept_sessions")
    @patch("oht_routing.runtime.server.socket.socket")
    @patch("oht_routing.runtime.server.read_port")
    def test_server_explicit_port_bypasses_wpconfig(
        self, read_port, socket_type, accept_sessions
    ):
        server = socket_type.return_value
        client = SimpleNamespace(
            wandb_logger=SimpleNamespace(
                finish_failed=Mock(),
                finish_interrupted=Mock(),
                finish_success=Mock(),
            ),
            training_failed=False,
            close_environment_capture=Mock(),
        )

        serve_contextual(client, port=9_123)

        read_port.assert_not_called()
        server.bind.assert_called_once_with(("127.0.0.1", 9_123))
        accept_sessions.assert_called_once_with(server, 9_123, client)
        server.close.assert_called_once_with()

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
