"""Simulator command dispatch for the contextual runtime."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


DEFAULT_PORT = 9100


def read_port(path="wpconfig.json"):
    base = Path(__file__).resolve().parent.parent.parent
    cfg_file = base / path
    if os.path.isfile(cfg_file):
        with cfg_file.open(encoding="utf-8") as stream:
            return int(json.load(stream)["MyProcessListenPort"])
    return DEFAULT_PORT


def send_active_data(pclient, client, log_interval=100):
    pclient.RecieveSimulationActiveData()
    client.Algorithm(pclient)
    started = time.perf_counter()
    pclient.SendRailLineCostMessage()
    client.record_send_cost_ms((time.perf_counter() - started) * 1000.0)
    client.capture_environment_tick(pclient)
    client.log_wandb_tick()
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
    sim_end_time=45_000,
    console_log_interval=100,
):
    if command == 0:
        if client.total_steps % int(console_log_interval) == 0:
            pclient.WriteAdminLog("Contextual SendAndReceiveRailLineCost.")
        send_active_data(pclient, client, log_interval=console_log_interval)
    elif command == 1:
        pclient.WriteAdminLog("Contextual simulation end signal (v=1).")
        client.on_terminal()
    elif command == 2:
        client.Reset(pclient)
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
    "handle_command",
    "read_port",
    "send_active_data",
)
