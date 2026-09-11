"""Start Stage 2 using the new Stage 1 run's lowest endpoint-TAT checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def launch_args(source):
    return [
        "--mode", "training", "--action-enabled", "--stage", "2",
        "--reward-version", "Q", "--load-stage1-policy", str(source),
        "--stage1-policy-warm-start", "--critic-target-mode", "uboc",
        "--num-critics", "5", "--replay-eviction-mode", "random",
        "--replay-capacity-env-steps", "200000",
        "--exploration-noise-std", "0.05",
        "--exploration-noise-final-std", "0.05", "--device", "cuda",
        "--checkpoint-root", str(source.parent.parent / "stage2"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Validate launch settings without loading pickle or starting a server")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--episode-summary-path", type=Path)
    parser.add_argument("--no-wandb", action="store_true")
    options = parser.parse_args()
    pointer = ROOT / "ud7_stage1_current.json"
    if not pointer.is_file():
        raise RuntimeError("Start run_ud7_stage1.py first; no new Stage 1 run is selected.")
    source = Path(json.loads(pointer.read_text(encoding="utf-8"))["best_policy"])
    if not source.is_file():
        raise RuntimeError(f"No eligible Stage 1 episode saved yet: {source}")
    selection = json.loads(source.with_name("selection.json").read_text(encoding="utf-8"))
    os.chdir(ROOT)
    from oht_routing.utils.wandb_logging import EXP_META
    EXP_META.update(
        note="besttat",
        description=(
            "UD7 Stage 2 from the fresh Stage 1 run's lowest eligible "
            "episode-end TAT checkpoint. Frozen deterministic "
            "Stage 1 inference for the first 2000 ticks of every episode, then "
            "fresh UD7 Stage 2 training with policy warm start, LAP, random "
            "replay eviction and flat exploration std 0.05."
        ),
    )
    # Freeze the selected artifact per launch; subsequent Stage 1 improvements
    # must not silently change the prefix required by a Stage 2 resume.
    if not options.check:
        import shutil
        from datetime import datetime
        launch_dir = source.parent.parent / ("stage2_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        launch_dir.mkdir()
        pinned = launch_dir / "stage1_policy.pt"
        shutil.copyfile(source, pinned)
        source = pinned
    args = launch_args(source)
    if not options.check:
        args[-1] = str(source.parent / "checkpoints")
    args += ["--port", str(options.port)]
    if options.episode_summary_path:
        args += ["--episode-summary-path", str(options.episode_summary_path.resolve())]
    if options.no_wandb:
        args += ["--no-wandb"]
    from oht_routing.runtime.cli import parse_args
    from oht_routing.runtime.config import runtime_config_from_args
    sys.argv = [str(Path(__file__).resolve()), *args]
    config = runtime_config_from_args(parse_args())
    with source.open("rb") as source_file:
        fingerprint = hashlib.file_digest(source_file, "sha256").hexdigest()
    if fingerprint != selection["sha256"]:
        raise RuntimeError("Stage 1 best policy changed during selection; stop Stage 1 and retry.")
    print(json.dumps({
        "source": str(source),
        "sha256": fingerprint,
        "selected_checkpoint_step": selection["runtime_env_step"],
        "selected_episode": selection["episode_id"],
        "selection_basis": "lowest_eligible_episode_end_tat",
        "best_episode_tat_s": selection["episode_end"]["tat_s"],
        "best_episode_sim_time": selection["episode_end"]["sim_time"],
        "stage": config.stage,
        "sim_end_time": config.sim_end_time,
        "warmup_steps": config.effective_warmup_steps,
        "arguments": args,
        "EXP_META": {k: EXP_META[k] for k in ("version", "note", "description")},
    }, indent=2, ensure_ascii=True), flush=True)
    if options.check:
        return
    # Keep the existing runtime's episode boundary and prefix replay exclusion.
    # Stage 2 itself performs frozen Stage 1 inference before learning begins.
    from main import main as run
    run()


if __name__ == "__main__":
    main()
