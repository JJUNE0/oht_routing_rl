import json
from contextlib import closing
from pathlib import Path
import socket
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from run_headless import (parse_options, result_name, require_free_port,
                          read_results, verify_result, clean_work, server_command, require_native_paths,
                          prepare_inference_checkpoint)
from oht_routing.runtime.client import ClientAlgorithm
from oht_routing.runtime.server import _run_session, _accept_sessions


class HeadlessTests(unittest.TestCase):
    def test_headless_closes_after_terminal_and_gui_session_can_continue(self):
        for headless in (True, False):
            with self.subTest(headless=headless), patch.dict("os.environ", {"PINOKIO_HEADLESS": "1" if headless else "0"}), \
                    patch("oht_routing.runtime.server.PClient.PClient") as pclient_type, \
                    patch("oht_routing.runtime.server.handle_command") as dispatch:
                pclient = pclient_type.return_value
                pclient.RecieveSimulationStandardData.side_effect = [1, ConnectionError("closed")]
                client = SimpleNamespace(config=SimpleNamespace(sim_end_time=60, console_log_interval=100),
                                         on_new_connection=Mock())
                if headless:
                    _run_session(object(), ("127.0.0.1", 1234), 9118, client)
                    self.assertEqual(pclient.RecieveSimulationStandardData.call_count, 1)
                else:
                    with self.assertRaises(ConnectionError):
                        _run_session(object(), ("127.0.0.1", 1234), 9118, client)
                dispatch.assert_called_once_with(1, pclient, client)

    def test_idle_server_exits_on_its_stop_file_without_a_tcp_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "stop.request"
            stop.touch()
            server = Mock()
            with patch.dict("os.environ", {"PINOKIO_STOP_FILE": str(stop)}):
                _accept_sessions(server, 9118, object())
            server.accept.assert_not_called()

    def test_result_names_and_path_escape_rejected(self):
        self.assertEqual(result_name("{input}_{episode:04d}", Path("input.db"), 2), "input_0002")
        for name in ("../x", "C:\\x", "x/y", "CON", "NUL.db", "", "x."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                result_name(name, Path("input.db"), 1)

    def test_invalid_horizon_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.db"
            source.touch()
            for extra in (["--end-time", "0"], ["--episodes", "2", "--name", "same"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    parse_options(["--mode", "baseline", "--input", str(source), *extra])

    def test_port_conflict_fails_before_starting(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            with self.assertRaises(OSError):
                require_free_port(listener.getsockname()[1])

    def test_long_native_config_path_is_rejected_before_startup(self):
        options = SimpleNamespace(inputs=[Path("input.db").resolve()], episodes=1, name="result")
        short = Path(tempfile.gettempdir()) / "pinokio_test"
        require_native_paths(short, options)
        output = short
        while len(str(output / "runtime/Pinokio.Headless.exe.config")) < 260:
            output = output / "nested"
        with self.assertRaisesRegex(ValueError, "260-character limit"):
            require_native_paths(output, options)

    def test_terminal_summary_is_once_per_episode_and_handles_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "episodes.jsonl"
            client = ClientAlgorithm.__new__(ClientAlgorithm)
            client.config = SimpleNamespace(stage=None, mode="baseline_only", sim_end_time=60,
                                            episode_summary_path=str(target))
            client.episode_id, client.total_steps, client.episode_steps = 1, 60, 60
            client.training_failed = False
            client.transition_aligner = None
            client.last_diagnostics = {"env/tat": float("nan"), "env/sim_time": 59,
                                       "env/completed": 12, "termination/done": 0}
            client.on_terminal()
            client.on_terminal()
            records = read_results(target)
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["full_horizon"])
            self.assertIsNone(records[0]["tat_s"])
            client.episode_id, client.total_steps = 2, 80
            client.last_diagnostics = {"env/tat": 50, "env/sim_time": 19, "termination/done": 1}
            client.on_terminal()
            self.assertTrue(read_results(target)[1]["early_termination"])
            self.assertFalse(read_results(target)[1]["full_horizon"])
            with target.open("a") as stream:
                stream.write('{"episode_id":')
            self.assertEqual(len(read_results(target)), 2)

    def test_native_caught_error_is_not_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            result, log = root / "result.db", root / "sim.log"
            with closing(sqlite3.connect(result)) as database:
                database.execute("CREATE TABLE test (id INTEGER)")
                database.commit()
            log.write_text("[headless] completed", encoding="utf-8")
            verify_result(result, log)
            log.write_text("[ERROR] swallowed native exception\n[headless] completed", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_result(result, log)

    def test_work_cleanup_cannot_delete_run_or_other_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            with self.assertRaises(ValueError):
                clean_work(root, root)
            with self.assertRaises(ValueError):
                clean_work(root, root.parent)
            work = root / "episode_work"
            work.mkdir()
            (work / "input.db").write_bytes(b"temporary")
            clean_work(root, work)
            self.assertFalse(work.exists())
            self.assertTrue(root.exists())

    def test_launch_preserves_training_stage_and_explicit_port(self):
        options = SimpleNamespace(port=9119, no_wandb=True, mode="stage1")
        command = server_command(options, Path("metrics.jsonl"))
        self.assertTrue(any(p.endswith("run_ud7_stage1.py") for p in command))
        self.assertEqual(command[command.index("--port") + 1], "9119")
        self.assertNotIn("--sim-end-time", command)

    def test_stage2_inference_preserves_its_frozen_prefix(self):
        options = SimpleNamespace(port=9119, no_wandb=True, mode="inference", device="cuda",
                                  end_time=45000, checkpoint=Path("stage2.pt"), stage1_policy=Path("prefix.pt"))
        command = server_command(options, Path("metrics.jsonl"))
        self.assertEqual(command[command.index("--stage") + 1], "2")
        self.assertEqual(command[command.index("--load-stage1-policy") + 1], "prefix.pt")
        self.assertEqual(command[command.index("--resume-checkpoint") + 1], "stage2.pt")
        self.assertNotIn("training", command)

    def test_inference_strength_override_is_explicit_and_bounded(self):
        options = SimpleNamespace(port=9119, no_wandb=True, mode="inference", device="cuda",
                                  end_time=45000, checkpoint=Path("stage2.pt"), stage1_policy=Path("prefix.pt"),
                                  rl_cost_lambda=0.45)
        command = server_command(options, Path("metrics.jsonl"))
        self.assertEqual(command[command.index("--rl-cost-lambda") + 1], "0.45")
        options.rl_cost_lambda = None
        self.assertNotIn("--rl-cost-lambda", server_command(options, Path("metrics.jsonl")))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.db"
            source.touch()
            for value in ("nan", "inf", "-0.1", "1.1"):
                with self.subTest(value=value), self.assertRaises(SystemExit):
                    parse_options(["--mode", "inference", "--input", str(source),
                                   "--checkpoint", str(source), "--rl-cost-lambda", value])

    def test_stage2_checkpoint_cannot_silently_skip_prefix_and_policy_is_pinned_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "latest.pt"
            source.write_bytes(b"selected policy")
            output = root / "run"
            output.mkdir()
            options = SimpleNamespace(checkpoint=source, stage1_policy=None)
            with patch("oht_routing.algorithms.rl.contextual_td7.checkpoint.read_contextual_runtime_config",
                       return_value=({"stage": 2}, True)):
                with self.assertRaisesRegex(ValueError, "requires --stage1-policy"):
                    prepare_inference_checkpoint(options, output)
            with patch("oht_routing.algorithms.rl.contextual_td7.checkpoint.read_contextual_runtime_config",
                       return_value=({"stage": 1}, True)):
                identity = prepare_inference_checkpoint(options, output)
            source.write_bytes(b"newer training policy")
            self.assertEqual(options.checkpoint.read_bytes(), b"selected policy")
            self.assertEqual(identity["source"], str(source))


if __name__ == "__main__":
    unittest.main()
