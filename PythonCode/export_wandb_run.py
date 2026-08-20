"""Canonical compact contextual W&B export schema.

The downloader can import ``EXPORT_COLUMNS`` to request a stable column order,
and tests keep this schema synchronized with the runtime logger.
"""

from contextual_wandb import EXP_META, WANDB_METRIC_KEYS

REWARD_N_EXPORT_COLUMNS = {
    "reward/global/tat_component_raw",
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
}
WARMUP_BOUNDARY_EXPORT_COLUMNS = {
    "termination/by_warmup",
    "warmup/episode_boundary_sent",
}
if not REWARD_N_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("Reward N diagnostics are missing from W&B export schema")
if not WARMUP_BOUNDARY_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("warm-up boundary diagnostics are missing from W&B export schema")


EXPORT_COLUMNS = ("_step", *WANDB_METRIC_KEYS)
EXPORT_SCHEMA_VERSION = EXP_META["wandb_metric_schema_version"]
