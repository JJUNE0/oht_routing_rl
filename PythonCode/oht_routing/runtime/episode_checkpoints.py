"""Episode snapshots and best observed terminal TAT, without replay payloads."""
import json
import hashlib
import math
from pathlib import Path
import shutil


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def episode_selection(*, diagnostics, episode_steps, total_steps, warmup_steps,
                      sim_end_time, normalizers_ready, learner_updates):
    tat = float(diagnostics.get("env/tat", float("nan")))
    sim_time = float(diagnostics.get("env/sim_time", -1))
    full_horizon = math.isfinite(sim_time) and sim_time >= sim_end_time - 1
    trained_episode = total_steps - episode_steps >= warmup_steps and learner_updates > 0
    eligible = (full_horizon and trained_episode and normalizers_ready
                and not diagnostics.get("termination/done", 0)
                and math.isfinite(tat) and tat > 0)
    return {
        "tat_s": tat if math.isfinite(tat) else None,
        "sim_time": sim_time if math.isfinite(sim_time) else None,
        "episode_steps": episode_steps,
        "full_horizon": full_horizon,
        "trained_episode": trained_episode,
        "normalizers_ready": normalizers_ready,
        "early_termination": bool(diagnostics.get("termination/done", 0)),
        "eligible_for_best": bool(eligible),
        "metric_source": "last_active_packet_before_episode_end",
    }


def save_episode(root, metadata, save_checkpoint):
    """Save every episode once; promote only eligible, strictly lower TAT.

    save_checkpoint(path, metadata) writes the normal compatible full checkpoint.
    A terminal packet contains no final observation, so selection records the
    last active packet explicitly instead of pretending it is a final metric.
    """
    root = Path(root)
    name = f"episode_{metadata['episode_id']:06d}_step_{metadata['runtime_env_step']:09d}"
    path = root / "episodes" / (name + ".pt")
    record_path = path.with_suffix(".json")
    if record_path.exists():
        return path
    best_record = root / "best_tat" / "selection.json"
    best = json.loads(best_record.read_text(encoding="utf-8")) if best_record.exists() else None
    selection = metadata["episode_end"]
    promote = selection["eligible_for_best"] and (
        best is None or selection["tat_s"] < best["episode_end"]["tat_s"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, metadata)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {**metadata, "source_checkpoint": str(path.resolve()), "sha256": digest}
    if promote:
        target = root / "best_tat" / "checkpoint.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".pt.tmp")
        shutil.copyfile(path, temporary)
        temporary.replace(target)
        atomic_json(best_record, record)
    atomic_json(record_path, record)
    print(f"[stage1-episode-saved] episode={metadata['episode_id']} "
          f"step={metadata['runtime_env_step']} tat={selection['tat_s']} "
          f"eligible={selection['eligible_for_best']} best_updated={bool(promote)} "
          f"path={path}", flush=True)
    return path
