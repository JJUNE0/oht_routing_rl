"""Wait for an existing Stage 1 batch, audit its best checkpoint, then evaluate it."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from oht_routing.runtime.episode_checkpoints import atomic_json
from run_headless import ROOT, read_results, result_name, stop_owned, verify_result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluation_root_for(training_run):
    # Keep the native CLR executable/config paths below MAX_PATH. Nesting a
    # second named run inside the training run can prevent CLR startup.
    identity = hashlib.sha256(str(Path(training_run).resolve()).encode("utf-8")).hexdigest()[:12]
    return ROOT / "results" / ("eval_" + identity)


def wait_for_training(state):
    """Hold a handle to the exact original process, including creation time."""
    if os.name != "nt":
        raise RuntimeError("This installed simulator requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, int(state["training_pid"]))
    manifest_path = Path(state["training_run"]) / "run.json"
    if not handle:
        error = ctypes.get_last_error()
        if error == 87 and read_json(manifest_path)["status"] == "complete":
            return
        raise OSError(error, "Original training process unavailable; inspect its manifest/log")
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        created = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) / 10_000_000 - 11_644_473_600
        expected = datetime.fromisoformat(state["training_started_utc"].replace("Z", "+00:00")).timestamp()
        if abs(created - expected) > 0.01:
            if read_json(manifest_path)["status"] == "complete":
                return
            raise RuntimeError("Training PID was reused; refusing to wait on an unrelated process")
        print(f"[finalize] waiting for training PID {state['training_pid']}", flush=True)
        while True:
            result = kernel.WaitForSingleObject(handle, 1000)
            if result == 0:
                break
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if code.value != 0:
            raise RuntimeError(f"Training exited with code {code.value}; inspect {manifest_path}")
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def exclusive_finalizer(path):
    import msvcrt
    with Path(path).open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def audit_training(state, *, check_databases=True):
    run = Path(state["training_run"]).resolve()
    root = Path(state["checkpoint_root"]).resolve()
    manifest = read_json(run / "run.json")
    count = int(state["planned_episodes"])
    if (manifest["status"] != "complete" or manifest["mode"] != "stage1"
            or manifest["episodes"] != count or manifest["completed_episodes"] != count
            or manifest["end_time"] != 2000):
        raise RuntimeError("Training has not completed the requested Stage 1 batch")
    input_path = Path(state["input_path"]).resolve()
    if ([Path(p).resolve() for p in manifest["inputs"]] != [input_path]
            or sha256(input_path) != state["input_sha256"]):
        raise RuntimeError("Training input identity changed")
    rows = read_results(run / "episodes.jsonl")
    files = sorted((root / "episodes").glob("*.json"))
    if len(rows) != count or len(files) != count:
        raise RuntimeError("Missing or extra episode metrics/checkpoints")
    records, candidates = [], []
    previous_step = 0
    for episode, (row, file) in enumerate(zip(rows, files), 1):
        record = read_json(file)
        end = record["episode_end"]
        source = Path(record["source_checkpoint"]).resolve()
        if (record["episode_id"] != episode or row["episode_id"] != episode
                or row["mode"] != "training" or row["stage"] != 1 or record["stage"] != 1
                or row["step"] != record["runtime_env_step"]
                or row["step"] - previous_step != row["episode_steps"]
                or row["episode_steps"] != end["episode_steps"]
                or row["tat_s"] != end["tat_s"] or row["sim_time"] != end["sim_time"]
                or row["early_termination"] != end["early_termination"]
                or row["training_failed"] or record["training_failed"]):
            raise RuntimeError(f"Episode {episode} metrics and checkpoint disagree")
        previous_step = row["step"]
        if source != file.with_suffix(".pt").resolve() or sha256(source) != record["sha256"]:
            raise RuntimeError(f"Episode {episode} checkpoint identity mismatch")
        full = end["sim_time"] is not None and math.isfinite(end["sim_time"]) and end["sim_time"] >= 1999
        tat = end["tat_s"]
        eligible = bool(full and not end["early_termination"]
                        and record["runtime_env_step"] - end["episode_steps"] >= 10000
                        and end["trained_episode"] and end["normalizers_ready"]
                        and record["normalizers_frozen"]
                        and tat is not None and math.isfinite(tat) and tat > 0)
        if full != end["full_horizon"] or full != row["full_horizon"] or eligible != end["eligible_for_best"]:
            raise RuntimeError(f"Episode {episode} eligibility is inconsistent")
        if check_databases:
            name = result_name(manifest["name"], input_path, episode)
            verify_result(run / (name + ".db"), run / (name + ".log"))
        records.append(record)
        if eligible:
            candidates.append(record)
    if not candidates:
        raise RuntimeError("No eligible post-warmup full-horizon Stage 1 checkpoint")
    best = min(candidates, key=lambda record: (record["episode_end"]["tat_s"], record["episode_id"]))
    selected = read_json(root / "best_tat/selection.json")
    if (selected["episode_id"] != best["episode_id"] or selected["sha256"] != best["sha256"]
            or selected["episode_end"] != best["episode_end"]
            or sha256(root / "best_tat/checkpoint.pt") != best["sha256"]):
        raise RuntimeError("Saved best is not the lowest eligible episode (earliest tie wins)")
    return best, records


def validate_policy(path, best):
    import torch
    from oht_routing.version import is_compatible_contextual_version
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (not is_compatible_contextual_version(payload["version"])
            or payload["num_critics"] != 5 or payload["critic_target_mode"] != "uboc"
            or payload["reward_version"] != "Q" or payload["action_mode"] != "region_b_rl"
            or not payload["sale_enabled"] or not payload["lap_enabled"]
            or payload["runtime_config"]["stage"] != 1
            or payload["learner_update_count"] <= 0):
        raise RuntimeError("Selected policy is not a trained compatible UD7 Stage 1 checkpoint")
    for key in ("episode_id", "runtime_env_step", "stage", "episode_end", "normalizers_frozen", "training_failed"):
        if payload["runtime_metadata"][key] != best[key]:
            raise RuntimeError(f"Selected checkpoint payload/provenance mismatch: {key}")
    for key in ("observation_local_normalizer", "observation_global_normalizer", "observation_critic_total_tat_normalizer"):
        normalizer = payload[key]
        if not normalizer["frozen"] or normalizer["count"] <= 0:
            raise RuntimeError(f"Selected policy has unusable normalizers: {key}")
    scale = float(payload["applied_action_scale"])
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise RuntimeError("Invalid saved policy action scale")
    return {"checkpoint_version": payload["version"], "applied_action_scale": scale,
            "learner_updates": int(payload["learner_update_count"])}


def verify_inference(run, pinned, input_path):
    manifest = read_json(run / "run.json")
    rows = read_results(run / "episodes.jsonl")
    if (manifest["status"] != "complete" or manifest["mode"] != "inference"
            or manifest["completed_episodes"] != 1 or manifest["episodes"] != 1
            or manifest["end_time"] != 2000 or Path(manifest["checkpoint"]).resolve() != pinned.resolve()
            or [Path(p).resolve() for p in manifest["inputs"]] != [Path(input_path).resolve()]
            or len(rows) != 1):
        raise RuntimeError("Inference batch is incomplete or used another checkpoint")
    row = rows[0]
    tat = row["tat_s"]
    # Standalone actor inference leaves runtime stage unset; the pinned
    # checkpoint's Stage 1 identity is checked separately by validate_policy.
    if (row["mode"] != "actor_inference" or row["stage"] not in (None, 1) or row["end_time"] != 2000 or not row["full_horizon"]
            or row["early_termination"] or row["training_failed"]
            or row["sim_time"] is None or not math.isfinite(row["sim_time"]) or row["sim_time"] < 1999
            or tat is None or not math.isfinite(tat) or tat <= 0):
        raise RuntimeError("Inference did not produce a valid full-horizon TAT")
    log = (run / "python.log").read_text(encoding="utf-8", errors="replace")
    if "[checkpoint-loaded]" not in log:
        raise RuntimeError("No confirmation that inference loaded its checkpoint")
    verify_result(run / "best_stage1_inference.db", run / "best_stage1_inference.log")
    return row


def finalize(state):
    run = Path(state["training_run"])
    status_path = run / "finalizer.json"
    with exclusive_finalizer(run / "finalizer.lock"):
        status = {"pid": os.getpid(), "phase": "waiting_for_training", "started_utc": datetime.now().astimezone().isoformat()}
        atomic_json(status_path, status)
        inference_process = None
        try:
            wait_for_training(state)
            status["phase"] = "auditing_training"
            atomic_json(status_path, status)
            best, records = audit_training(state)
            destination = run / "final_stage1"
            destination.mkdir(exist_ok=True)
            pinned = destination / "checkpoint.pt"
            if pinned.exists():
                if sha256(pinned) != best["sha256"]:
                    raise RuntimeError("An existing pinned checkpoint differs from the audited best")
            else:
                temporary = pinned.with_suffix(".pt.tmp")
                shutil.copyfile(best["source_checkpoint"], temporary)
                if sha256(temporary) != best["sha256"]:
                    raise RuntimeError("Policy changed during pinning")
                temporary.replace(pinned)
            policy = validate_policy(pinned, best)
            atomic_json(destination / "selection.json", best)
            with (destination / "episode_ranking.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("episode", "step", "tat_s", "eligible_for_best", "checkpoint"))
                for record in sorted(records, key=lambda r: (not r["episode_end"]["eligible_for_best"], r["episode_end"]["tat_s"] or math.inf, r["episode_id"])):
                    writer.writerow((record["episode_id"], record["runtime_env_step"], record["episode_end"]["tat_s"], record["episode_end"]["eligible_for_best"], record["source_checkpoint"]))
            evaluation_root = evaluation_root_for(run)
            status["inference_output_root"] = str(evaluation_root)
            existing = list(evaluation_root.glob("*/run.json"))
            if existing:
                if len(existing) != 1:
                    raise RuntimeError("Multiple existing inference runs; inspect before retrying")
                inference_run = existing[0].parent
                evaluation = verify_inference(inference_run, pinned, state["input_path"])
            else:
                status.update(phase="inference", selected_episode=best["episode_id"], selected_sha256=best["sha256"])
                atomic_json(status_path, status)
                command = [sys.executable, "-u", str(ROOT / "PythonCode/run_headless.py"),
                           "--mode", "inference", "--checkpoint", str(pinned), "--input", state["input_path"],
                           "--episodes", "1", "--end-time", "2000", "--port", str(state["port"]),
                           "--output-dir", str(evaluation_root), "--name", "best_stage1_inference", "--device", "cuda"]
                with (destination / "inference_launcher.log").open("w", encoding="utf-8") as log:
                    inference_process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                         creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
                    status["inference_pid"] = inference_process.pid
                    atomic_json(status_path, status)
                    if inference_process.wait() != 0:
                        raise RuntimeError(f"Inference failed; see {destination / 'inference_launcher.log'}")
                manifests = list(evaluation_root.glob("*/run.json"))
                if len(manifests) != 1:
                    raise RuntimeError("Cannot identify the inference output")
                inference_run = manifests[0].parent
                evaluation = verify_inference(inference_run, pinned, state["input_path"])
            if sha256(pinned) != best["sha256"] or sha256(state["input_path"]) != state["input_sha256"]:
                raise RuntimeError("Evaluation input or pinned policy identity changed")
            result = {"status": "complete", "training_run": str(run), "episodes_audited": len(records),
                      "eligible_episodes": sum(r["episode_end"]["eligible_for_best"] for r in records),
                      "selected_episode": best["episode_id"], "selected_step": best["runtime_env_step"],
                      "training_tat_s": best["episode_end"]["tat_s"], "checkpoint": str(pinned),
                      "sha256": best["sha256"], "input_sha256": state["input_sha256"], **policy,
                      "inference_runtime_version": read_json(inference_run / "run.json")["EXP_META"]["version"],
                      "inference_run": str(inference_run), "inference": evaluation}
            atomic_json(destination / "result.json", result)
            report = (f"# UD7 Stage 1 result\n\n"
                      f"Audited {len(records)} episodes and their checkpoint hashes/results.\n\n"
                      f"| Measurement | Result |\n|---|---:|\n"
                      f"| Selected episode | {best['episode_id']} |\n"
                      f"| Selected training step | {best['runtime_env_step']} |\n"
                      f"| Lowest eligible training TAT (s) | {best['episode_end']['tat_s']} |\n"
                      f"| Deterministic inference TAT (s) | {evaluation['tat_s']} |\n"
                      f"| Fixed action scale | {policy['applied_action_scale']} |\n\n"
                      f"[Checkpoint]({pinned.as_posix()}) · [Inference results]({(inference_run / 'evaluation.csv').as_posix()})\n\n"
                      f"SHA-256: `{best['sha256']}`\n\n"
                      "TAT is the last active packet's TotalTat before termination. Both selected training and inference completed the 2000-second horizon. Warmup and early-stop episodes are excluded from selection. Training TAT includes exploration; inference is a separate zero-noise evaluation.\n")
            (destination / "STAGE1_RESULT.md").write_text(report, encoding="utf-8")
            status.update(phase="complete", result=str(destination / "result.json"), report=str(destination / "STAGE1_RESULT.md"))
            atomic_json(status_path, status)
            print(json.dumps(result, indent=2), flush=True)
        except BaseException as error:
            status.update(phase="failed", error=str(error))
            atomic_json(status_path, status)
            raise
        finally:
            stop_owned(inference_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-state", type=Path, default=ROOT / "results/ud7_stage1_goal.json")
    parser.add_argument("--audit-only", action="store_true", help="Read-only audit of an already completed training batch")
    options = parser.parse_args()
    state = read_json(options.goal_state)
    if options.audit_only:
        best, records = audit_training(state)
        print(json.dumps({"best": best, "episodes": len(records)}, indent=2))
    else:
        finalize(state)
