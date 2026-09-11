"""Terminal metrics for automated simulator evaluation; no new W&B metrics."""
import json
import math
import os
from pathlib import Path


def write_episode_result(path, config, diagnostics, episode_id, step,
                         episode_steps, training_failed=False):
    def metric(key):
        value = diagnostics.get(key)
        return float(value) if value is not None and math.isfinite(float(value)) else None

    sim_time = metric("env/sim_time")
    record = {
        "episode_id": episode_id, "step": step, "episode_steps": episode_steps,
        "mode": config.mode, "stage": config.stage,
        "sim_time": sim_time, "end_time": config.sim_end_time,
        "full_horizon": sim_time is not None and sim_time >= config.sim_end_time - 1,
        "tat_s": metric("env/tat"),
        "completed": metric("env/completed"),
        "queued": metric("env/queued"), "waiting": metric("env/waiting"),
        "early_termination": bool(diagnostics.get("termination/done", 0)),
        "training_failed": bool(training_failed),
        "metric_source": "last_active_packet_before_episode_end",
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record
