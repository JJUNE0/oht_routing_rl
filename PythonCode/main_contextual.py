"""Contextual baseline, inference, and synchronous training runtime."""

import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import PClient
import numpy as np
from ClientAlgorithm_contextual import (
    ClientAlgorithm,
    ContextualRuntimeConfig,
    ContextualTrainingFailure,
)
from contextual_action import ACTION_MODES, REGION_B_RL


HOST = "127.0.0.1"
PORT = 9100
EXPECTED_SIMULATION_STATES = {0, 1, 2, 3, 4, 5, 6}
RUNTIME_PERFORMANCE_VERSION = "contextual_runtime_bounded_reporting_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contextual baseline, inference, and Phase 7 training runtime"
    )
    parser.add_argument(
        "--mode",
        choices=("baseline_only", "actor_inference", "training"),
        default="baseline_only",
    )
    parser.add_argument("--action-enabled", action="store_true")
    parser.add_argument(
        "--action-mode", choices=ACTION_MODES, default=REGION_B_RL
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.05,
        help="Fixed applied-action scale for exp_residual mode only.",
    )
    parser.add_argument("--curriculum-end-step", type=int, default=40_000)
    parser.add_argument("--curriculum-scale-start", type=float, default=0.05)
    parser.add_argument("--curriculum-scale-end", type=float, default=1.0)
    parser.add_argument(
        "--curriculum-shape",
        choices=("geometric", "linear"),
        default="geometric",
    )
    parser.add_argument("--smooth-b-rl-weight", type=float, default=0.05)
    parser.add_argument(
        "--smooth-exp-residual-weight", type=float, default=0.5
    )
    parser.add_argument("--exploration-noise-std", type=float, default=0.10)
    parser.add_argument("--exploration-noise-clip", type=float, default=0.20)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--normalizer-freeze-steps", type=int, default=10_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-capacity-env-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--minimum-replay-env-steps", type=int, default=100)
    parser.add_argument(
        "--minimum-action-enabled-env-steps", type=int, default=100
    )
    parser.add_argument("--updates-per-env-step", type=int, default=1)
    parser.add_argument("--learn-every-env-steps", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--sale", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--lap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--critic-loss-mode", choices=("auto", "huber", "mse"), default="auto"
    )
    parser.add_argument("--wandb-log-interval", type=int, default=10)
    parser.add_argument("--console-log-interval", type=int, default=100)
    parser.add_argument("--early-stop-queued-threshold", type=float, default=500.0)
    parser.add_argument("--early-stop-tat-threshold", type=float, default=500.0)
    parser.add_argument("--early-stop-min-episode-steps", type=int, default=100)
    parser.add_argument("--max-stale-sim-time-ticks", type=int, default=5)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--sim-end-time",
        type=int,
        default=45_000,
        help="Override PClient's simulator episode end time after every init/reset.",
    )
    parser.add_argument(
        "--smoke-report",
        type=Path,
        default=None,
        help="Continuously write aggregated live smoke metrics as JSON.",
    )
    parser.add_argument(
        "--smoke-report-interval",
        type=int,
        default=100,
        help="Write the smoke summary every N active ticks.",
    )
    return parser.parse_args()


def read_port(path="wpconfig.json"):
    base = Path(__file__).resolve().parent.parent.parent
    cfg_file = base / path
    if os.path.isfile(cfg_file):
        with cfg_file.open(encoding="utf-8") as stream:
            return int(json.load(stream)["MyProcessListenPort"])
    return PORT


class SmokeReporter:
    _MAX_KEYS = (
        "cost/all_baseline_abs_error_max",
        "boundary/cost_baseline_abs_error_max",
        "runtime/observation_build_calls_per_tick",
        "runtime/nonfinite_count",
    )

    def __init__(self, path, mode, write_interval=100):
        if int(write_interval) <= 0:
            raise ValueError("smoke report interval must be positive")
        self.path = Path(path)
        self.mode = mode
        self.write_interval = int(write_interval)
        self.tick_count = 0
        self.totals = []
        self.last = {}
        self.maxima = {key: 0.0 for key in self._MAX_KEYS}
        self.reset_count = 0
        self.socket_send_count = 0

    def record_tick(self, diagnostics):
        self.tick_count += 1
        self.last = dict(diagnostics)
        total = (
            self.last.get("runtime/total_algorithm_ms", np.nan)
            + self.last.get("runtime/send_cost_ms", 0.0)
        )
        if np.isfinite(total):
            self.totals.append(float(total))
        for key in self._MAX_KEYS:
            self.maxima[key] = max(
                self.maxima[key], float(self.last.get(key, 0.0))
            )
        if self.tick_count % self.write_interval == 0:
            self._write()

    def record_reset(self):
        self.reset_count += 1
        self._write()

    def record_send(self):
        self.socket_send_count += 1

    def _write(self):
        totals = np.asarray(self.totals, dtype=np.float64)
        payload = {
            "mode": self.mode,
            "tick_count": self.tick_count,
            "episode_reset_count": self.reset_count,
            "socket_send_count": self.socket_send_count,
            "runtime_total_ms": {
                "median": float(np.median(totals)) if totals.size else None,
                "p95": float(np.percentile(totals, 95)) if totals.size else None,
                "p99": float(np.percentile(totals, 99)) if totals.size else None,
            },
            "last": self.last,
            "max": dict(self.maxima) if self.tick_count else {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def send_active_data(pclient, client, reporter=None, log_interval=100):
    pclient.RecieveSimulationActiveData()
    client.Algorithm(pclient)
    started = time.perf_counter()
    pclient.SendRailLineCostMessage()
    client.record_send_cost_ms((time.perf_counter() - started) * 1000.0)
    client.log_wandb_tick()
    if reporter is not None:
        reporter.record_send()
        reporter.record_tick(client.last_diagnostics)
    client.AlgorithmAfter(pclient)
    diagnostics = client.last_diagnostics
    step = int(diagnostics.get("env/step", client.total_steps))
    warmup_remaining = max(0, int(client.config.warmup_steps) - step)
    replay_steps = int(diagnostics.get("replay/size_env_steps", 0))
    learner_updates = int(diagnostics.get("update/learner_count", 0))
    action_state = (
        "enabled"
        if client.config.action_enabled and warmup_remaining == 0
        else "warmup"
    )
    if step % int(log_interval) == 0:
        print(
            "[main-contextual] "
            f"step={step}, episode={client.episode_id}, "
            f"episode_step={client.episode_steps}, action={action_state}, "
            f"warmup_remaining={warmup_remaining}, "
            f"replay_steps={replay_steps}, learner_updates={learner_updates}",
            flush=True,
        )


def handle_command(
    command,
    pclient,
    client,
    reporter=None,
    sim_end_time=45_000,
    console_log_interval=100,
):
    if command == 0:
        if client.total_steps % int(console_log_interval) == 0:
            pclient.WriteAdminLog("Contextual SendAndReceiveRailLineCost.")
        send_active_data(
            pclient, client, reporter, log_interval=console_log_interval
        )
    elif command == 1:
        pclient.WriteAdminLog("Contextual simulation end signal (v=1).")
        client.on_terminal()
    elif command == 2:
        client.Reset(pclient)
        if reporter is not None:
            reporter.record_reset()
    elif command == 3:
        pclient.RecieveSimulationSnapshotData()
        client.UpdateDatas(pclient)
    elif command == 4:
        key, current, destination = pclient.RecieveSingleOHTData()
        client.UpdateOHTRoute(
            pclient,
            key,
            current,
            destination,
            pclient.OHT_DIC[key].RouteList,
        )
        pclient.SendSingleRoute(key)
    elif command == 5:
        ohts, from_nodes, to_nodes = pclient.GetNeedReRouteOht()
        pclient.SendReRoute(client.ReRoute(pclient, ohts, from_nodes, to_nodes))
    elif command == 6:
        pclient.RecieveSimulationSnapshotData()
        jobs = pclient.GetAssignCommand()
        pclient.SendAssignOht(client.Assign(pclient, jobs))


def main():
    try:
        sys.stdout.reconfigure(
            encoding="utf-8", errors="replace", line_buffering=True
        )
        sys.stderr.reconfigure(
            encoding="utf-8", errors="replace", line_buffering=True
        )
    except (AttributeError, ValueError):
        pass

    args = parse_args()
    if args.sim_end_time <= 0:
        raise ValueError("--sim-end-time must be positive")
    if args.console_log_interval <= 0:
        raise ValueError("--console-log-interval must be positive")
    config_kwargs = {
        "mode": args.mode,
        "action_enabled": args.action_enabled,
        "action_mode": args.action_mode,
        "action_scale": args.action_scale,
        "curriculum_end_step": args.curriculum_end_step,
        "curriculum_scale_start": args.curriculum_scale_start,
        "curriculum_scale_end": args.curriculum_scale_end,
        "curriculum_shape": args.curriculum_shape,
        "smooth_b_rl_weight": args.smooth_b_rl_weight,
        "smooth_exp_residual_weight": args.smooth_exp_residual_weight,
        "warmup_steps": args.warmup_steps,
        "normalizer_freeze_steps": args.normalizer_freeze_steps,
        "seed": args.seed,
        "exploration_noise_std": args.exploration_noise_std,
        "exploration_noise_clip": args.exploration_noise_clip,
        "replay_capacity_env_steps": args.replay_capacity_env_steps,
        "batch_size": args.batch_size,
        "minimum_replay_env_steps": args.minimum_replay_env_steps,
        "minimum_action_enabled_env_steps": (
            args.minimum_action_enabled_env_steps
        ),
        "updates_per_env_step": args.updates_per_env_step,
        "learn_every_env_steps": args.learn_every_env_steps,
        "wandb_enabled": args.wandb,
        "sale_enabled": args.sale,
        "lap_enabled": args.lap,
        "critic_loss_mode": args.critic_loss_mode,
        "wandb_log_interval": args.wandb_log_interval,
        "early_stop_queued_threshold": args.early_stop_queued_threshold,
        "early_stop_tat_threshold": args.early_stop_tat_threshold,
        "early_stop_min_episode_steps": args.early_stop_min_episode_steps,
        "max_stale_sim_time_ticks": args.max_stale_sim_time_ticks,
        "checkpoint_root": args.checkpoint_root,
        "resume_checkpoint_path": args.resume_checkpoint,
    }
    if args.device:
        config_kwargs["device"] = args.device
    client = ClientAlgorithm(ContextualRuntimeConfig(**config_kwargs))
    reporter = (
        SmokeReporter(
            args.smoke_report, args.mode, args.smoke_report_interval
        )
        if args.smoke_report is not None
        else None
    )
    print(
        "[main-contextual] "
        f"mode={args.mode}, action_enabled={args.action_enabled}, "
        f"action_mode={args.action_mode}, "
        f"action_scale={client._action_scale()}, "
        f"warmup_steps={args.warmup_steps}, "
        f"device={client.device}"
    )

    port = read_port()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, port))
    server.listen(1)
    try:
        while True:
            print(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))
            print(f"[main-contextual] waiting on {HOST}:{port}")
            connection, address = server.accept()
            with connection:
                print(
                    "[main-contextual] TCP accepted; "
                    f"starting PClient initialization handshake: {address}"
                )
                # PClient owns the fixed-width initialization/reset handshake.
                # Passing the override avoids appending duplicate raw bytes.
                pclient = PClient.PClient(
                    connection, sim_end_time=args.sim_end_time
                )
                client.on_new_connection()
                print(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))
                print("[main-contextual] PClient initialization completed")
                print("Socket 서버와 연결되었습니다.")
                print(HOST)
                print(port)
                pclient.WriteAdminLog("Socket 서버와 연결되었습니다. ")
                command_count = 0
                try:
                    while True:
                        command = pclient.RecieveSimulationStandardData()
                        command_count += 1
                        if (
                            command != 0
                            or command_count % args.console_log_interval == 0
                        ):
                            print(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))
                            print(
                                "[main-contextual] "
                                f"command={command}, count={command_count}"
                            )
                            pclient.WriteAdminLog(
                                "RecieveSimulationStandardData."
                            )
                        if command not in EXPECTED_SIMULATION_STATES:
                            pending = pclient.PeekPending(64)
                            preview = " ".join(
                                f"{value:02X}" for value in pending[:32]
                            )
                            raise RuntimeError(
                                f"[DESYNC] unexpected v={command}; "
                                f"pending={len(pending)}B: {preview}"
                            )
                        handle_command(
                            command,
                            pclient,
                            client,
                            reporter,
                            sim_end_time=args.sim_end_time,
                            console_log_interval=args.console_log_interval,
                        )
                except ContextualTrainingFailure:
                    raise
                except (ConnectionError, IndexError):
                    traceback.print_exc()
                    print("[main-contextual] disconnected; waiting for reconnect")
                except FloatingPointError:
                    raise
                except Exception:
                    traceback.print_exc()
                    if client.training_failed:
                        raise ContextualTrainingFailure(
                            "training failure latched; operator restart required"
                        )
                    print("[main-contextual] session failed; waiting for reconnect")
    except ContextualTrainingFailure as error:
        client.wandb_logger.finish_failed(error, client.failure_env_step)
        raise
    except KeyboardInterrupt:
        client.wandb_logger.finish_interrupted()
        raise
    finally:
        if not client.training_failed:
            client.wandb_logger.finish_success()
        server.close()


if __name__ == "__main__":
    main()
