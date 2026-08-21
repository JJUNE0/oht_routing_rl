"""Export the compact contextual training metric schema from W&B to CSV."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_CODE_DIR = PROJECT_ROOT / "PythonCode"
if str(PYTHON_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_DIR))

from contextual_wandb import (
    EXP_META,
    REMOVED_WANDB_PREFIXES,
    WANDB_METRIC_KEYS,
)

REWARD_U_EXPORT_COLUMNS = {
    "reward/global/tat_component_raw",
    "reward/global/backlog_component_raw",
    "reward/global/raw",
    "reward/global/normalized",
    "reward/global/component",
    "reward/completion/count",
    "reward/completion/valid_count",
    "reward/completion/invalid_count",
    "reward/completion/tat_mean",
    "reward/completion/tat_std",
    "reward/completion/tat_min",
    "reward/completion/tat_max",
    "reward/completion/raw",
    "reward/completion/weighted_raw",
    "reward/local/raw_mean",
    "reward/local/raw_std",
    "reward/local/normalized_mean",
    "reward/local/normalized_std",
    "reward/local/component_mean",
    "reward/local/component_std",
    "reward/contribution/completion_tat_abs",
    "reward/contribution/tat_abs",
    "reward/contribution/backlog_abs",
    "reward/contribution/local_abs",
    "reward/budget/rail_abs",
    "reward/budget/smooth_abs",
    "reward/budget/completion_tat_share",
    "reward/budget/tat_share",
    "reward/budget/backlog_share",
    "reward/budget/local_share",
    "reward/budget/rail_share",
    "reward/budget/smooth_share",
    "reward/budget/share_sum_error",
    "trace/command/id",
    "trace/command/lifetime",
    "trace/command/available",
    "trace/command/cmd_tat",
    "trace/command/oht_tat",
    "trace/command/oht_id",
    "trace/command/oht_state",
    "trace/command/completed",
    "local/oht_abs_mean",
    "local/oht_std",
    "local/pred_abs_mean",
    "local/pred_std",
    "local/stop_abs_mean",
    "local/stop_std",
    "local/idle_abs_mean",
    "local/idle_std",
    "local/capacity_abs_mean",
    "local/capacity_std",
    "reward/rail_tat_mean",
    "reward/smooth_penalty_mean",
    "reward/total_mean",
    "reward/total_std",
    "reward/global_normalizer_mean",
    "reward/global_normalizer_std",
    "reward/local_normalizer_mean",
    "reward/local_normalizer_std",
}
WARMUP_BOUNDARY_EXPORT_COLUMNS = {
    "termination/by_warmup",
    "warmup/episode_boundary_sent",
}


if len(WANDB_METRIC_KEYS) != len(set(WANDB_METRIC_KEYS)):
    raise RuntimeError("compact W&B export schema contains duplicate keys")
if any(key.startswith(REMOVED_WANDB_PREFIXES) for key in WANDB_METRIC_KEYS):
    raise RuntimeError("removed metric group leaked into W&B export schema")
if not REWARD_U_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("Reward U diagnostics are missing from W&B export schema")
if not WARMUP_BOUNDARY_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("warm-up boundary diagnostics are missing from W&B export schema")

EXPORT_COLUMNS = ("_step",) + WANDB_METRIC_KEYS
EXPORT_SCHEMA_VERSION = EXP_META["wandb_metric_schema_version"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="entity/project/run_id")
    parser.add_argument("--output", default="contextual_wandb_run.csv")
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help=(
            "Export every history key present in the run. Use this when an "
            "older run does not contain every key in the current schema."
        ),
    )
    args = parser.parse_args()
    import wandb

    run = wandb.Api().run(args.run_path)
    if args.all_keys:
        rows = list(run.scan_history())
        keys = {key for row in rows for key in row}
        leading = [
            key for key in ("_step", "_timestamp", "_runtime")
            if key in keys
        ]
        columns = tuple(leading + sorted(keys.difference(leading)))
    else:
        rows = run.scan_history(keys=list(EXPORT_COLUMNS))
        columns = EXPORT_COLUMNS
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


if __name__ == "__main__":
    main()
