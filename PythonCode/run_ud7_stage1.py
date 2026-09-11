"""Fresh UD7 Stage 1, with episode snapshots and lowest terminal TAT selection."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--episode-summary-path", type=Path)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--note", default="episodebest",
                        help="Run tag used by the run/checkpoint name")
    # Unrecognized arguments are main.py training options. They are appended
    # after this runner's defaults, so a swept value wins over the default.
    options, overrides = parser.parse_known_args()
    os.chdir(ROOT)
    from oht_routing.utils.wandb_logging import EXP_META, _make_run_name
    EXP_META.update(
        reward_version="Q", note=options.note,
        description=("Fresh UD7 Stage 1 (UBOC 5, reward Q, SALE/LAP, random eviction). "
                     "Save a full checkpoint at each episode end; retain lowest "
                     "last-packet TAT among full episodes after warmup with frozen "
                     "normalizers. No checkpoint or normalizer warm start."
                     + (f" Training overrides: {' '.join(overrides)}." if overrides else "")),
    )
    destination = ROOT / "checkpoints" / _make_run_name(EXP_META)
    if not options.check:
        # Never mix separate fresh learners, even when started within one minute.
        base = destination
        suffix = 1
        while True:
            try:
                destination.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                destination = base.with_name(base.name + f"_{suffix}")
                suffix += 1
    args = [
        "--mode", "training", "--action-enabled", "--stage", "1",
        "--reward-version", "Q", "--critic-target-mode", "uboc",
        "--num-critics", "5", "--replay-eviction-mode", "random",
        "--replay-capacity-env-steps", "100000", "--device", options.device,
        "--checkpoint-root", str(destination),
        "--port", str(options.port),
    ]
    if options.episode_summary_path:
        args += ["--episode-summary-path", str(options.episode_summary_path.resolve())]
    if options.no_wandb:
        args += ["--no-wandb"]
    args += overrides
    from oht_routing.runtime.cli import parse_args
    from oht_routing.runtime.config import runtime_config_from_args
    sys.argv = [str(Path(__file__).resolve()), *args]
    config = runtime_config_from_args(parse_args())
    print(json.dumps(asdict(config), indent=2, ensure_ascii=True), flush=True)
    if options.check:
        return
    from oht_routing.runtime.episode_checkpoints import atomic_json
    atomic_json(ROOT / "ud7_stage1_current.json", {
        "checkpoint_root": str(destination),
        "best_policy": str(destination / "best_tat/checkpoint.pt"),
    })
    from main import main as run
    run()


if __name__ == "__main__":
    main()
