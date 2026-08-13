"""Contextual baseline, inference, and synchronous training runtime."""

import argparse
import json
import os
import random
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import PClient
import numpy as np
import torch
from ClientAlgorithm_contextual import (
    ClientAlgorithm,
    ContextualRuntimeConfig,
    ContextualTrainingFailure,
)
from contextual_dispatch import (
    DISPATCH_FIRST_MATCH,
    DISPATCH_MODES,
)
from cocel_rl.algorithms.contextual_td7 import (
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    read_contextual_runtime_config,
)
from contextual_action import ACTION_MODES, REGION_B_RL
from contextual_reward import (
    RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
    RAIL_REWARD_MODES,
)


HOST = "127.0.0.1"
PORT = 9100
EXPECTED_SIMULATION_STATES = {0, 1, 2, 3, 4, 5, 6}
RUNTIME_PERFORMANCE_VERSION = "contextual_runtime_bounded_reporting_v1"
RESUME_RUNTIME_CONFIG_VERSION = "contextual_resume_full_runtime_config_v1"
RESUME_LAUNCH_CONTROL_FIELDS = {
    "mode",
    "action_enabled",
    "device",
    "topology_cache_path",
    "topology_audit_path",
    "checkpoint_root",
    "resume_checkpoint_path",
    "rail_tat_diagnostic_path",
    "rail_tat_diagnostic_max_step",
    "reward_diagnostic_dir",
    "reward_diagnostic_windows",
    "wandb_enabled",
    "dispatch_mode",
    "tat_confidence_ramp",
}


def seed_everything(seed):
    seed = int(seed)
    if seed < 0:
        raise ValueError("--seed must be non-negative")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def restore_checkpoint_runtime_config(config_kwargs, checkpoint_path):
    """Apply saved experiment settings while preserving launch controls."""
    saved, complete = read_contextual_runtime_config(checkpoint_path)
    valid_fields = set(ContextualRuntimeConfig.__dataclass_fields__)
    restored = []
    for key, value in saved.items():
        if key not in valid_fields or key in RESUME_LAUNCH_CONTROL_FIELDS:
            continue
        config_kwargs[key] = value
        restored.append(key)
    source = "full" if complete else "legacy-partial"
    print(
        "[checkpoint-config] "
        f"version={RESUME_RUNTIME_CONFIG_VERSION}, source={source}, "
        f"restored={','.join(sorted(restored)) or 'none'}"
    )
    if not complete:
        print(
            "[checkpoint-config] v3 checkpoint does not contain every runtime "
            "setting; unavailable fields retain the current CLI/default value."
        )
    return config_kwargs


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
        "--dispatch-mode",
        choices=DISPATCH_MODES,
        default=DISPATCH_FIRST_MATCH,
        help=(
            "OHT selection for command 6: preserve iteration-order "
            "first-match, or minimize cumulative live command-0 rail cost."
        ),
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.05,
        help="Fixed applied-action scale for exp_residual mode only.",
    )
    parser.add_argument(
        "--num-stacks",
        type=int,
        default=1,
        help="Total observation frames including the current frame.",
    )
    parser.add_argument(
        "--stack-interval",
        type=int,
        default=1,
        help="Environment-step spacing between stacked frames.",
    )
    parser.add_argument("--curriculum-end-step", type=int, default=20_000)
    parser.add_argument("--curriculum-scale-start", type=float, default=0.05)
    parser.add_argument("--curriculum-scale-end", type=float, default=1.0)
    parser.add_argument(
        "--curriculum-shape",
        choices=("geometric", "linear"),
        default="geometric",
    )
    parser.add_argument("--smooth-b-rl-weight", type=float, default=0.25)
    parser.add_argument(
        "--smooth-exp-residual-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--exploration-noise-std",
        type=float,
        default=0.10,
        help="Exploration noise std at global environment step 0.",
    )
    parser.add_argument(
        "--exploration-noise-final-std",
        type=float,
        default=0.02,
        help="Final exploration noise std after annealing.",
    )
    parser.add_argument(
        "--exploration-noise-anneal-steps",
        type=int,
        default=100_000,
        help="Post-warmup env steps over which noise reaches its final std.",
    )
    parser.add_argument("--exploration-noise-clip", type=float, default=0.20)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--episode-burnin-steps", type=int, default=0)
    parser.add_argument("--normalizer-freeze-steps", type=int, default=10_000)
    parser.add_argument(
        "--rail-reward-mode",
        choices=RAIL_REWARD_MODES,
        default=RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
    )
    parser.add_argument(
        "--rail-free-flow-neutral-ratio",
        type=float,
        default=2.0,
        help="Route/free-flow ratio that receives zero rail reward.",
    )
    parser.add_argument(
        "--reward-diagnostic-dir",
        default=None,
        help="Opt-in directory for contextual reward step/cycle JSONL diagnostics.",
    )
    parser.add_argument(
        "--reward-diagnostic-windows",
        default="0:1000,10000:11000,20000:21000",
        help="Start-inclusive:end-exclusive global-step windows.",
    )
    parser.add_argument("--tat-confidence-n0", type=float, default=500.0)
    parser.add_argument(
        "--use-tat-confidence",
        "--use-tat-cofidence",
        dest="use_tat_cofidence",
        action="store_true",
        default=False,
        help=(
            "Apply completed/(completed + n0) confidence ramp to the TAT "
            "reward term. Disabled by default."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-capacity-env-steps", type=int, default=50_000)
    replay_sampling = parser.add_mutually_exclusive_group()
    replay_sampling.add_argument(
        "--replay-buffer-rail",
        dest="replay_sampling_mode",
        action="store_const",
        const=REPLAY_SAMPLING_RAIL,
        help=(
            "Sample batch-size independent (environment step, controlled "
            "rail) transitions. This is the default and original behavior."
        ),
    )
    replay_sampling.add_argument(
        "--replay-buffer-snapshot",
        dest="replay_sampling_mode",
        action="store_const",
        const=REPLAY_SAMPLING_SNAPSHOT,
        help=(
            "Sample batch-size distinct environment steps and include every "
            "controlled rail from each selected snapshot. Requires --no-lap."
        ),
    )
    replay_sampling.add_argument(
        "--replay-buffer-random-rail",
        "--random-rail-mode",
        dest="replay_sampling_mode",
        action="store_const",
        const=REPLAY_SAMPLING_RANDOM_RAIL,
        help=(
            "Uniformly sample batch-size unique centered-rail transitions "
            "directly from the complete (environment step, controlled rail) "
            "replay pool. Requires --no-lap."
        ),
    )
    parser.set_defaults(replay_sampling_mode=REPLAY_SAMPLING_RAIL)
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
    parser.add_argument("--early-stop-tat-threshold", type=float, default=200.0)
    parser.add_argument("--tat-termination-grace-steps", type=int, default=10_000)
    parser.add_argument("--tat-above-threshold-patience", type=int, default=300)
    parser.add_argument("--max-stale-sim-time-ticks", type=int, default=5)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--rail-tat-diagnostic",
        default=None,
        help=(
            "Opt in to the legacy rail-TAT completion JSONL at this path."
        ),
    )
    parser.add_argument(
        "--rail-tat-diagnostic-max-step",
        type=int,
        default=1_000,
        help=(
            "Write rail-TAT JSONL only while global_step is below this "
            "exclusive cutoff. Zero disables event logging."
        ),
    )
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
        started = time.perf_counter()
        pclient.RecieveSimulationSnapshotData()
        snapshot_done = time.perf_counter()
        jobs = pclient.GetAssignCommand()
        jobs_done = time.perf_counter()
        assigned = client.Assign(pclient, jobs)
        assign_done = time.perf_counter()
        pclient.SendAssignOht(assigned)
        send_done = time.perf_counter()
        print(
            f"[cmd6] jobs={len(jobs)} assigned={len(assigned)} snapshot_ms={(snapshot_done - started) * 1000:.1f} get_jobs_ms={(jobs_done - snapshot_done) * 1000:.1f} assign_ms={(assign_done - jobs_done) * 1000:.1f} send_ms={(send_done - assign_done) * 1000:.1f} total_ms={(send_done - started) * 1000:.1f}",
            flush=True,
        )


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
        "dispatch_mode": args.dispatch_mode,
        "action_scale": args.action_scale,
        "num_stacks": args.num_stacks,
        "stack_interval": args.stack_interval,
        "curriculum_end_step": args.curriculum_end_step,
        "curriculum_scale_start": args.curriculum_scale_start,
        "curriculum_scale_end": args.curriculum_scale_end,
        "curriculum_shape": args.curriculum_shape,
        "smooth_b_rl_weight": args.smooth_b_rl_weight,
        "smooth_exp_residual_weight": args.smooth_exp_residual_weight,
        "warmup_steps": args.warmup_steps,
        "episode_burnin_steps": args.episode_burnin_steps,
        "normalizer_freeze_steps": args.normalizer_freeze_steps,
        "rail_reward_mode": args.rail_reward_mode,
        "rail_free_flow_neutral_ratio": args.rail_free_flow_neutral_ratio,
        "reward_diagnostic_dir": args.reward_diagnostic_dir,
        "reward_diagnostic_windows": args.reward_diagnostic_windows,
        "tat_confidence_n0": args.tat_confidence_n0,
        "tat_confidence_ramp": args.use_tat_cofidence,
        "seed": args.seed,
        "exploration_noise_std": args.exploration_noise_std,
        "exploration_noise_final_std": args.exploration_noise_final_std,
        "exploration_noise_anneal_steps": (
            args.exploration_noise_anneal_steps
        ),
        "exploration_noise_clip": args.exploration_noise_clip,
        "replay_capacity_env_steps": args.replay_capacity_env_steps,
        "replay_sampling_mode": args.replay_sampling_mode,
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
        "tat_termination_grace_steps": args.tat_termination_grace_steps,
        "tat_above_threshold_patience": args.tat_above_threshold_patience,
        "max_stale_sim_time_ticks": args.max_stale_sim_time_ticks,
        "checkpoint_root": args.checkpoint_root,
        "resume_checkpoint_path": args.resume_checkpoint,
        "rail_tat_diagnostic_path": args.rail_tat_diagnostic,
        "rail_tat_diagnostic_max_step": (
            args.rail_tat_diagnostic_max_step
        ),
    }
    if args.device:
        config_kwargs["device"] = args.device
    if args.resume_checkpoint:
        config_kwargs = restore_checkpoint_runtime_config(
            config_kwargs, args.resume_checkpoint
        )
    seed_everything(config_kwargs["seed"])
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
        f"dispatch_mode={client.config.dispatch_mode}, "
        f"action_scale={client._action_scale()}, "
        f"num_stacks={client.config.num_stacks}, "
        f"stack_interval={client.config.stack_interval}, "
        f"warmup_steps={client.config.warmup_steps}, "
        f"episode_burnin_steps={client.config.episode_burnin_steps}, "
        f"reward_diagnostic_dir={client.config.reward_diagnostic_dir}, "
        f"rail_reward_mode={client.config.rail_reward_mode}, "
        "rail_free_flow_neutral_ratio="
        f"{client.config.rail_free_flow_neutral_ratio}, "
        f"tat_weight={client.config.tat_weight}, "
        f"op_weight={client.config.op_weight}, "
        f"backlog_weight={client.config.backlog_weight}, "
        f"backlog_growth_weight={client.config.backlog_growth_weight}, "
        f"idle_reserve_weight={client.config.idle_reserve_weight}, "
        "local_predicted_oht_weight="
        f"{client.config.local_predicted_oht_weight}, "
        f"local_reward_scale={client.config.local_reward_scale}, "
        f"rail_tat_weight={client.config.reward_rail_tat_weight}, "
        f"rail_tat_clip={client.config.reward_rail_tat_clip}, "
        "tat_termination="
        f"{client.config.early_stop_tat_threshold:g}x"
        f"{client.config.tat_above_threshold_patience} after "
        f"{client.config.tat_termination_grace_steps}, "
        f"terminal_tat_penalty={client.config.terminal_tat_penalty:g}, "
        f"use_tat_confidence={client.config.tat_confidence_ramp}, "
        f"replay_sampling={client.config.replay_sampling_mode}, "
        f"batch_size={client.config.batch_size}, "
        f"exploration_noise={client.config.exploration_noise_std}->"
        f"{min(client.config.exploration_noise_std, client.config.exploration_noise_final_std)}"
        f"@global[{client.config.warmup_steps},"
        f"{client.config.warmup_steps + client.config.exploration_noise_anneal_steps}], "
        f"seed={client.config.seed}, "
        f"device={client.device}"
    )
    print(
        "[main-contextual] rail_tat_diagnostic="
        f"{client.rail_tat_diagnostic_path}, "
        f"global_step<"
        f"{client.config.rail_tat_diagnostic_max_step}"
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
