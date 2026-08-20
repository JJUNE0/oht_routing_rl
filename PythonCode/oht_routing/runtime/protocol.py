"""Simulator command dispatch and lightweight smoke reporting."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


DEFAULT_PORT = 9100


def read_port(path="wpconfig.json"):
    base = Path(__file__).resolve().parent.parent.parent
    cfg_file = base / path
    if os.path.isfile(cfg_file):
        with cfg_file.open(encoding="utf-8") as stream:
            return int(json.load(stream)["MyProcessListenPort"])
    return DEFAULT_PORT


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
                "p95": (
                    float(np.percentile(totals, 95)) if totals.size else None
                ),
                "p99": (
                    float(np.percentile(totals, 99)) if totals.size else None
                ),
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
    warmup_remaining = max(
        0, int(client.config.effective_warmup_steps) - step
    )
    replay_steps = int(diagnostics.get("replay/size_env_steps", 0))
    learner_updates = int(diagnostics.get("update/learner_count", 0))
    action_state = (
        "enabled"
        if client.config.action_enabled and warmup_remaining == 0
        else "warmup"
    )
    if step % int(log_interval) == 0:
        print(
            "[contextual-runtime] "
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
            "[cmd6] "
            f"jobs={len(jobs)} assigned={len(assigned)} "
            f"snapshot_ms={(snapshot_done - started) * 1000:.1f} "
            f"get_jobs_ms={(jobs_done - snapshot_done) * 1000:.1f} "
            f"assign_ms={(assign_done - jobs_done) * 1000:.1f} "
            f"send_ms={(send_done - assign_done) * 1000:.1f} "
            f"total_ms={(send_done - started) * 1000:.1f}",
            flush=True,
        )


__all__ = (
    "DEFAULT_PORT",
    "SmokeReporter",
    "handle_command",
    "read_port",
    "send_active_data",
)
