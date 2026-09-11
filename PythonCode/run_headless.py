"""Launch Pinokio and its Python controller without opening the simulator GUI."""
from __future__ import annotations

import argparse
import csv
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT.parent / "Simulator"
DEFAULT_INPUT = ROOT.parent / "db/base/AICC_Input_260403.db"
BUILD = ROOT / "tools/pinokio"
RUNTIME_FILES = ("Pinokio.Headless.exe", "Pinokio.Headless.exe.config",
                 "Pinokio.TCP.IP.dll", "Pinokio.Utill.Log.dll")
# Controller options this launcher already owns. A sweep that sets them again
# through the override list would desynchronize the native simulator, the
# result collector or the run's own output layout, so name the launcher flag
# that owns each one instead of silently letting the last value win.
RESERVED_TRAINING_OPTIONS = {
    "--port": "pass --port before the --",
    "--ports": "pass --port before the --",
    "--num-sim": "one simulator per launch; pass --port before the --",
    "--mode": "pass --mode before the --",
    "--stage": "pass --mode stage1 or --mode stage2 before the --",
    "--sim-end-time": "pass --end-time before the --, so the native simulator agrees",
    "--device": "pass --device before the --",
    "--wandb": "pass --no-wandb before the --",
    "--no-wandb": "pass --no-wandb before the --",
    "--episode-summary-path": "this launcher collects the per-episode metrics itself",
    "--checkpoint-root": "the training runner names each run's checkpoint root",
    "--resume-checkpoint": "pass --checkpoint before the --",
    "--load-stage1-policy": "pass --stage1-policy before the --",
}


# Stage 1 training targets. Both runners share every setting except the critic
# contract, so a UD7 run and a TD7 run form a controlled comparison. Each keeps
# its own "current Stage 1 run" pointer so neither selects the other's policy.
STAGE1_RUNNERS = {"ud7": "run_ud7_stage1.py", "td7": "run_td7_stage1.py"}
STAGE2_RUNNERS = {"ud7": "run_ud7_best_stage2.py"}


def stage1_pointer(algorithm):
    return ROOT / f"{algorithm}_stage1_current.json"


def split_training_overrides(argv=None):
    """Split launcher arguments from the training overrides after a bare --."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        return argv, []
    boundary = argv.index("--")
    return argv[:boundary], argv[boundary + 1:]


def result_name(pattern, input_path, episode):
    name = pattern.format(input=input_path.stem, episode=episode)
    if (not name or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.endswith((".", " ")) or len(name) > 120
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", name)):
        raise ValueError(f"Invalid result filename: {name!r}")
    return name


def parse_options(argv=None):
    argv, overrides = split_training_overrides(argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=("Everything after a bare -- is passed to the Python controller's "
                "own CLI, so any training option can be swept: "
                "run_headless.py --mode stage1 --episodes 50 -- --seed 3 "
                "--exploration-noise-std 0.05"),
    )
    parser.add_argument("--mode", choices=("stage1", "stage2", "inference", "baseline", "simulator"), default="stage1")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path)
    inputs.add_argument("--input-dir", type=Path, help="Run all .db files in sorted order, cycling if needed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/headless")
    parser.add_argument("--name", default="{input}_{episode:04d}", help="Filename template: {input}, {episode:04d}")
    parser.add_argument("--episodes", type=int, help="Default: 100 for training, one pass over inputs for evaluation")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--end-time", type=int, help="Simulation seconds; training fixes Stage 1=2000 / Stage 2=45000")
    parser.add_argument("--checkpoint", type=Path, help="Inference checkpoint; default: fresh Stage 1 lowest-TAT policy")
    parser.add_argument("--stage1-policy", type=Path, help="Frozen prefix required when evaluating a Stage 2 checkpoint")
    parser.add_argument("--rl-cost-lambda", type=float, help="Optional inference cost-strength experiment (0..1); default: saved checkpoint value")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Training and inference device")
    parser.add_argument("--note", help="One-word run tag for the run/checkpoint name; default: each runner's own tag")
    parser.add_argument("--algorithm", choices=tuple(STAGE1_RUNNERS), default="ud7",
                        help="Training target: ud7 (UBOC ensemble) or td7 (clipped double-Q baseline)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate paths/options without starting any process or run")
    parser.add_argument("--keep-work", action="store_true", help="Keep decrypted per-episode working DBs for debugging")
    parser.add_argument("--timeout", type=float, default=21600, help="Maximum wall seconds per episode (default: 6h)")
    options = parser.parse_args(argv)
    options.train_args = tuple(overrides)
    if options.note is not None and not re.fullmatch(r"[A-Za-z0-9._-]{1,32}", options.note):
        parser.error("--note must be 1..32 characters of letters, digits, '.', '_' or '-'")
    if options.mode == "stage2" and options.algorithm not in STAGE2_RUNNERS:
        parser.error(f"no Stage 2 runner for --algorithm {options.algorithm}; "
                     "train its Stage 1 and evaluate with --mode inference")
    if options.train_args:
        if options.mode == "simulator":
            parser.error("simulator mode starts no Python controller; training overrides need "
                         "--mode stage1/stage2/inference/baseline")
        if not options.train_args[0].startswith("--"):
            parser.error("training overrides must begin with an option, for example: -- --seed 3")
        for token in options.train_args:
            name = token.split("=", 1)[0]
            if name in RESERVED_TRAINING_OPTIONS:
                parser.error(f"{name} is set by this launcher; {RESERVED_TRAINING_OPTIONS[name]}")
            if name == "--rl-cost-lambda" and options.rl_cost_lambda is not None:
                parser.error("--rl-cost-lambda was given twice; keep either the launcher option or the override")
    options.inputs = (sorted(options.input_dir.resolve().glob("*.db"))
                      if options.input_dir else [(options.input or DEFAULT_INPUT).resolve()])
    if not options.inputs or any(not path.is_file() for path in options.inputs):
        parser.error("Input DB not found")
    options.episodes = options.episodes if options.episodes is not None else (100 if options.mode.startswith("stage") else len(options.inputs))
    if options.episodes < 1 or not 1 <= options.port <= 65535 or options.timeout <= 0:
        parser.error("episodes and timeout must be positive; port must be 1..65535")
    if options.mode.startswith("stage") and options.end_time is not None:
        parser.error("Training uses the fixed stage horizon; omit --end-time")
    if options.stage1_policy is not None:
        if options.mode != "inference" or options.checkpoint is None:
            parser.error("--stage1-policy requires --mode inference and an explicit Stage 2 --checkpoint")
        options.stage1_policy = options.stage1_policy.resolve()
        if not options.stage1_policy.is_file():
            parser.error(f"Stage 1 prefix checkpoint not found: {options.stage1_policy}")
    if options.end_time is None:
        options.end_time = 45000 if options.mode == "stage2" or options.stage1_policy else 2000
    if not 1 <= options.end_time <= 999999:
        parser.error("end-time must be 1..999999 (native TCP encoding)")
    if options.mode != "inference" and options.checkpoint is not None:
        parser.error("--checkpoint is only for inference")
    if options.rl_cost_lambda is not None and (
        options.mode != "inference" or not 0.0 <= options.rl_cost_lambda <= 1.0
    ):
        parser.error("--rl-cost-lambda requires inference and a finite value in [0, 1]")
    if options.mode in ("inference", "stage2"):
        if options.checkpoint is None:
            pointer = stage1_pointer(options.algorithm)
            if not pointer.is_file():
                parser.error(f"Train a fresh {options.algorithm} Stage 1 first, or provide "
                             "--checkpoint for inference")
            options.checkpoint = Path(json.loads(pointer.read_text(encoding="utf-8"))["best_policy"])
        options.checkpoint = options.checkpoint.resolve()
        if not options.checkpoint.is_file():
            parser.error(f"No eligible Stage 1 policy yet: {options.checkpoint}")
    names = [result_name(options.name, options.inputs[i % len(options.inputs)], i + 1)
             for i in range(options.episodes)]
    if len(set(name.casefold() for name in names)) != len(names):
        parser.error("Result names repeat; include {episode:04d} in --name")
    if not (SIM / "Simulation.Model.dll").is_file():
        parser.error(f"Simulator binaries not found: {SIM}")
    return options


def require_free_port(port):
    with socket.socket() as probe:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", port))


def require_native_paths(output, options):
    """Fail clearly before starting Python when legacy CLR paths are too long."""
    paths = list(options.inputs)
    for episode in range(1, options.episodes + 1):
        source = options.inputs[(episode - 1) % len(options.inputs)]
        name = result_name(options.name, source, episode)
        paths.extend((output / "runtime" / "Pinokio.Headless.exe.config",
                      output / (name + ".db"), output / (name + "_work") / "input.db"))
    for path in paths:
        if len(str(path.resolve()).encode("utf-16-le")) // 2 >= 260:
            raise ValueError(f"Native simulator path reaches the 260-character limit: {path}. "
                             "Use a shorter --output-dir / --name or input path.")


def server_command(options, summary_path):
    common = ["--port", str(options.port), "--episode-summary-path", str(summary_path)]
    if options.no_wandb or options.mode == "baseline":
        common.append("--no-wandb")
    if options.mode in ("stage1", "stage2"):
        script = (STAGE1_RUNNERS if options.mode == "stage1" else STAGE2_RUNNERS)[options.algorithm]
        # main.py has no run tag of its own, so --note reaches only the two
        # training runners that name a checkpoint root. Overrides come last so
        # a swept value wins over the runner's default.
        tag = ["--note", options.note] if options.note else []
        return [sys.executable, "-u", str(ROOT / "PythonCode" / script),
                "--device", options.device, *tag, *common, *options.train_args]
    arguments = ["--mode", "actor_inference" if options.mode == "inference" else "baseline_only",
                 "--reward-version", "Q", "--sim-end-time", str(options.end_time),
                 "--device", options.device if options.mode == "inference" else "cpu"]
    if options.mode == "inference":
        arguments += ["--action-enabled", "--resume-checkpoint", str(options.checkpoint)]
        if getattr(options, "stage1_policy", None):
            arguments += ["--stage", "2", "--load-stage1-policy", str(options.stage1_policy)]
        if getattr(options, "rl_cost_lambda", None) is not None:
            arguments += ["--rl-cost-lambda", str(options.rl_cost_lambda)]
    return [sys.executable, "-u", str(ROOT / "PythonCode/run_headless_server.py"),
            *arguments, *common, *options.train_args]


def stop_owned(process, stop_file=None):
    if process is None or process.poll() is not None:
        return
    if stop_file is not None:
        stop_file.touch()
        try:
            process.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    except OSError:
        pass  # A hidden Windows child has no console to receive CTRL_BREAK.
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            process.kill()
        process.wait(timeout=10)


def wait_for_server(process, log, port, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Python exited ({process.returncode}); see {log}")
        # Do not connect to test readiness: that would consume a protocol session.
        if f"waiting on 127.0.0.1:{port}" in log.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(0.2)
    raise TimeoutError(f"Python did not become ready; see {log}")


def read_results(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]


def verify_result(path, simulator_log):
    text = simulator_log.read_text(encoding="utf-8", errors="replace")
    if (any(marker in text for marker in ("[ERROR]", "HEADLESS ERROR", " Engine Error", "Licence limited"))
            or "[headless] completed" not in text):
        raise RuntimeError(f"Simulator reported an error or incomplete execution: {simulator_log}")
    if not path.is_file():
        raise RuntimeError(f"Simulator result DB missing: {path}")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
        if database.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise RuntimeError(f"Invalid result DB: {path}")


def clean_work(output, work):
    """Remove only generated work beneath this unique invocation's output."""
    root, target = output.resolve(), work.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Refusing to remove work outside run: {target}")
    if target.exists():
        shutil.rmtree(target)


def prepare_inference_checkpoint(options, output):
    """Pin a policy once for all rollouts and enforce the saved stage contract."""
    from oht_routing.algorithms.rl.contextual_td7.checkpoint import read_contextual_runtime_config

    source = options.checkpoint
    pinned = output / "policy.pt"
    shutil.copyfile(source, pinned)
    saved, _ = read_contextual_runtime_config(pinned)
    saved_stage = saved.get("stage")
    if saved_stage == 2 and options.stage1_policy is None:
        raise ValueError("Stage 2 inference requires --stage1-policy with the matching frozen prefix; "
                         "running its Stage 2 actor from time zero changes the evaluation contract")
    if options.stage1_policy is not None and saved_stage != 2:
        raise ValueError("--stage1-policy expects a Stage 2 --checkpoint")
    identity = {"source": str(source), "pinned": str(pinned), "saved_stage": saved_stage,
                "sha256": hashlib.sha256(pinned.read_bytes()).hexdigest()}
    options.checkpoint = pinned
    if options.stage1_policy is not None:
        prefix_source = options.stage1_policy
        prefix = output / "prefix.pt"
        shutil.copyfile(prefix_source, prefix)
        options.stage1_policy = prefix
        identity["prefix_source"] = str(prefix_source)
        identity["prefix_pinned"] = str(prefix)
        identity["prefix_sha256"] = hashlib.sha256(prefix.read_bytes()).hexdigest()
    return identity


def run(options):
    if options.mode != "simulator":
        require_free_port(options.port)
    if options.check:
        print(json.dumps({"mode": options.mode, "inputs": [str(p) for p in options.inputs],
                          "episodes": options.episodes, "end_time": options.end_time,
                          "port": options.port, "name": options.name,
                          "checkpoint": str(options.checkpoint) if options.checkpoint else None,
                          "stage1_policy": str(options.stage1_policy) if options.stage1_policy else None,
                          "rl_cost_lambda_override": options.rl_cost_lambda,
                          "note": options.note, "algorithm": options.algorithm,
                          "train_args": list(options.train_args),
                          "headless_built": all((BUILD / "bin" / f).is_file() for f in RUNTIME_FILES)}, indent=2))
        return
    if os.name != "nt":
        raise RuntimeError("This installed Pinokio engine requires Windows/.NET Framework 4.8")
    from oht_routing.utils.wandb_logging import EXP_META, _make_run_name
    overrides = " ".join(options.train_args)
    EXP_META.update(note=options.note or "headless", reward_version="Q",
                    description=(f"Headless {options.mode}: {options.episodes} episodes, "
                                 "original Pinokio TCP protocol."
                                 + (f" Training overrides: {overrides}." if overrides else "")))
    base = options.output_dir.resolve() / _make_run_name(EXP_META)
    output, suffix = base, 0
    while True:
        try:
            output.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            output = base.with_name(base.name + f"_{suffix}")
    from oht_routing.runtime.episode_checkpoints import atomic_json
    manifest = {"EXP_META": EXP_META, "mode": options.mode, "status": "running",
                "inputs": [str(p) for p in options.inputs], "episodes": options.episodes,
                "port": options.port, "end_time": options.end_time, "name": options.name,
                "checkpoint": str(options.checkpoint) if options.checkpoint else None,
                "stage1_policy": str(options.stage1_policy) if options.stage1_policy else None,
                "rl_cost_lambda_override": options.rl_cost_lambda,
                "note": options.note, "algorithm": options.algorithm,
                "train_args": list(options.train_args),
                "completed_episodes": 0}
    atomic_json(output / "run.json", manifest)
    print(f"[batch] output={output}", flush=True)
    summary = output / "episodes.jsonl"
    server, simulator, server_stream = None, None, None
    creation = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    try:
        require_native_paths(output, options)
        if options.mode == "inference":
            manifest["policy_identity"] = prepare_inference_checkpoint(options, output)
        # Build once into this run. Later builds cannot change an active run's
        # executable, and sequential episodes reuse these immutable binaries.
        runtime = output / "runtime"
        runtime.mkdir()
        preparation_started = time.monotonic()
        subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-File", str(BUILD / "build.ps1"), "-OutputDirectory", str(runtime)], check=True)
        manifest["runtime_preparation_s"] = round(time.monotonic() - preparation_started, 3)
        manifest["runtime_sha256"] = {
            name: hashlib.sha256((runtime / name).read_bytes()).hexdigest() for name in RUNTIME_FILES
        }
        atomic_json(output / "run.json", manifest)
        if options.mode != "simulator":
            server_log = output / "python.log"
            server_stream = server_log.open("w", encoding="utf-8")
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
                   "PINOKIO_HEADLESS": "1", "PINOKIO_STOP_FILE": str(output / "stop.request")}
            server = subprocess.Popen(server_command(options, summary), cwd=ROOT,
                                      stdout=server_stream, stderr=subprocess.STDOUT, env=env,
                                      creationflags=creation)
            wait_for_server(server, server_log, options.port)
        for episode in range(1, options.episodes + 1):
            input_path = options.inputs[(episode - 1) % len(options.inputs)]
            name = result_name(options.name, input_path, episode)
            sim_log = output / (name + ".log")
            command = [str(runtime / "Pinokio.Headless.exe"), "--sim-dir", str(SIM),
                       "--input", str(input_path), "--output-dir", str(output), "--name", name,
                       "--end-time", str(options.end_time)]
            command += ["--no-python"] if server is None else ["--port", str(options.port)]
            started = time.monotonic()
            print(f"[batch] episode={episode}/{options.episodes} input={input_path.name}", flush=True)
            with sim_log.open("w", encoding="utf-8") as stream:
                simulator = subprocess.Popen(command, cwd=runtime, stdout=stream, stderr=subprocess.STDOUT,
                                             creationflags=creation)
                next_progress = started + 30
                while simulator.poll() is None:
                    if server is not None and server.poll() is not None:
                        raise RuntimeError(f"Python server exited; see {server_log}")
                    now = time.monotonic()
                    if now - started > options.timeout:
                        raise TimeoutError(f"Episode exceeded {options.timeout}s: {sim_log}")
                    if now >= next_progress:
                        progress = [line for line in sim_log.read_text(encoding="utf-8", errors="replace").splitlines()
                                    if "[headless]" in line]
                        print(f"[batch] elapsed={now-started:.0f}s " + (progress[-1] if progress else "starting"), flush=True)
                        next_progress = now + 30
                    time.sleep(0.2)
            if simulator.returncode != 0:
                raise RuntimeError(f"Simulator exited ({simulator.returncode}); see {sim_log}")
            verify_result(output / (name + ".db"), sim_log)
            record = {}
            if server is not None:
                deadline = time.monotonic() + 180
                while len(read_results(summary)) < episode:
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(f"Missing terminal metrics/checkpoint; see {server_log}")
                    time.sleep(0.2)
                record = read_results(summary)[episode - 1]
                if record["training_failed"]:
                    raise RuntimeError(f"Learner failed; see {server_log}")
            row = {"episode": episode, "input": str(input_path), "result_db": str(output / (name + ".db")),
                   "wall_s": round(time.monotonic() - started, 2), **record}
            from simulator.final_window import score_final_window, score_meeting_window
            row.update(score_final_window(
                output / (name + ".db"),
                full_rollout=options.end_time >= 45000 and record.get("full_horizon", server is None),
            ))
            meeting_score = score_meeting_window(
                output / (name + ".db"),
                full_rollout=options.end_time >= 45000 and record.get("full_horizon", server is None),
            )
            row.update({"meeting_" + key: value for key, value in meeting_score.items()})
            csv_path = output / "evaluation.csv"
            with csv_path.open("a", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if episode == 1:
                    writer.writeheader()
                writer.writerow(row)
            manifest["completed_episodes"] = episode
            atomic_json(output / "run.json", manifest)
            if not options.keep_work:
                clean_work(output, output / (name + "_work"))
            print(f"[batch] saved={name}.db wall_s={row['wall_s']} tat_s={row.get('tat_s')} full_horizon={row.get('full_horizon')}", flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        stop_owned(simulator)
        stop_owned(server, output / "stop.request")
        if server_stream is not None:
            server_stream.close()
        atomic_json(output / "run.json", manifest)
    print(f"[batch] complete: {output / 'evaluation.csv'}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    try:
        run(parse_options())
    except KeyboardInterrupt:
        print("[batch] interrupted; completed results retained", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"[batch] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
