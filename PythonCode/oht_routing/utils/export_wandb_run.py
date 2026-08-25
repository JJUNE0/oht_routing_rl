"""Export the compact contextual training metric schema from W&B to CSV."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON_CODE_DIR = PROJECT_ROOT / "PythonCode"
if str(PYTHON_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_DIR))

from oht_routing.utils.wandb_logging import (
    EXP_META,
    REMOVED_WANDB_PREFIXES,
    WANDB_METRIC_KEYS,
)

REWARD_P_EXPORT_COLUMNS = {
    "env/recent_completed_tat_300s_mean",
    "env/recent_completed_tat_300s_p90",
    "env/recent_completed_tat_300s_count",
    "env/recent_completed_tat_300s_available",
    "observation/recent_completed_tat_300s_mean",
    "observation/recent_completed_tat_300s_available",
    "reward/global/tat_raw",
    "reward/global/tat_component_raw",
    "reward/global/total_tat",
    "reward/global/recent_completed_tat_300s_mean",
    "reward/global/cumulative_total_tat",
    "reward/global/backlog_component_raw",
    "reward/global/raw",
    "reward/global/component",
    "reward/local/raw_mean",
    "reward/local/raw_std",
    "reward/local/normalized_mean",
    "reward/local/normalized_std",
    "reward/local/component_mean",
    "reward/local/component_std",
    "reward/contribution/tat_abs",
    "reward/contribution/backlog_abs",
    "reward/contribution/local_abs",
    "reward/budget/rail_abs",
    "reward/budget/smooth_abs",
    "reward/budget/tat_share",
    "reward/budget/backlog_share",
    "reward/budget/local_share",
    "reward/budget/rail_share",
    "reward/budget/smooth_share",
    "reward/budget/share_sum_error",
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
} | {
    f"reward/term_scale/{name}_abs_mean"
    for name in (
        "tat", "op", "backlog_level", "backlog_growth", "idle_reserve",
        "current_oht", "predicted_oht", "stop_time", "local_idle",
        "capacity", "rail_outcome", "smooth",
    )
} | {
    f"reward/term_share/{name}"
    for name in (
        "tat", "op", "backlog_level", "backlog_growth", "idle_reserve",
        "current_oht", "predicted_oht", "stop_time", "local_idle",
        "capacity", "rail_outcome", "smooth",
    )
}
WARMUP_BOUNDARY_EXPORT_COLUMNS = {
    "termination/by_warmup",
    "warmup/episode_boundary_sent",
}
RUNTIME_SAFETY_EXPORT_COLUMNS = {
    "cost/all_baseline_abs_error_max",
    "runtime/state_collection_ms",
    "runtime/observation_build_ms",
    "runtime/tensor_conversion_ms",
    "runtime/host_to_device_ms",
    "runtime/encoder_actor_ms",
    "runtime/actor_inference_ms",
    "runtime/device_to_host_ms",
    "runtime/cost_apply_ms",
    "runtime/checkpoint_ms",
    "runtime/send_cost_ms",
    "runtime/total_algorithm_ms",
    "runtime/total_ms",
    "runtime/data_capture_ms",
    "runtime/observation_build_calls_per_tick",
    "runtime/nonfinite_count",
}
OBSERVATION_EXPORT_COLUMNS = {
    "observation/predicted_route10_pearson",
    "observation/predicted_route10_mae",
    "observation/predicted_route10_nonzero_agreement",
    "observation/predicted_route10_both_nonzero",
    "observation/predicted_route10_pearson_available",
    "observation/predicted_route10_union_nonzero_pearson",
    "observation/predicted_route10_union_nonzero_pearson_available",
    "observation/predicted_route10_union_nonzero_ratio",
}


if len(WANDB_METRIC_KEYS) != len(set(WANDB_METRIC_KEYS)):
    raise RuntimeError("compact W&B export schema contains duplicate keys")
if any(key.startswith(REMOVED_WANDB_PREFIXES) for key in WANDB_METRIC_KEYS):
    raise RuntimeError("removed metric group leaked into W&B export schema")
if not REWARD_P_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("Reward P diagnostics are missing from W&B export schema")
if not WARMUP_BOUNDARY_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("warm-up boundary diagnostics are missing from W&B export schema")
if not RUNTIME_SAFETY_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("runtime safety diagnostics are missing from W&B export schema")
if not OBSERVATION_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("observation diagnostics are missing from W&B export schema")

EXPORT_COLUMNS = ("_step",) + WANDB_METRIC_KEYS
EXPORT_SCHEMA_VERSION = EXP_META["version"]


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
