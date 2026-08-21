"""Canonical compact contextual W&B export schema.

The downloader can import ``EXPORT_COLUMNS`` to request a stable column order,
and tests keep this schema synchronized with the runtime logger.
"""

from contextual_wandb import EXP_META, WANDB_METRIC_KEYS

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
if not REWARD_U_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("Reward U diagnostics are missing from W&B export schema")
if not WARMUP_BOUNDARY_EXPORT_COLUMNS <= set(WANDB_METRIC_KEYS):
    raise RuntimeError("warm-up boundary diagnostics are missing from W&B export schema")


EXPORT_COLUMNS = ("_step", *WANDB_METRIC_KEYS)
EXPORT_SCHEMA_VERSION = EXP_META["wandb_metric_schema_version"]
