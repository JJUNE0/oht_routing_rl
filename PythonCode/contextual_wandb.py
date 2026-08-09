"""Training-only W&B schema and lightweight logger."""

from __future__ import annotations

from datetime import datetime

from cocel_rl.algorithms.contextual_td7 import (
    ALGORITHM_VERSION,
    CHECKPOINT_VERSION,
    CRITIC_INITIALIZATION,
    LAP_VERSION,
    REPLAY_SAMPLING_VERSION,
    SALE_VERSION,
    contextual_algorithm_variant,
)
from contextual_action import (
    EXPLORATION_SCHEDULE_VERSION,
    REGION_B_RL,
    action_version,
)
from contextual_dispatch import DISPATCH_SELECTION_VERSION
from contextual_reward import ContextualRewardConfig, REWARD_VERSION

EXP_META = {
    "cost_structure": "b_rl",
    "action_range": "b_rl_0.0-1.0",
    "topology": "directed_10in_10out_controlled_centers_v2",
    "observation": "contextual_obs_controlled_v1",
    "reward_version": "N",
    "reward_contract_version": REWARD_VERSION,
    "calibration_status": "provisional",
    "reward_global_alpha": 0.5,
    "reward_local_alpha": 0.5,
    "tat_reference": 165.0,
    "tat_weight": 11.0,
    "tat_penalty_start": 160.0,
    "op_reference": 0.80,
    "op_weight": 4.0,
    "backlog_weight": 0.0004,
    "backlog_growth_enabled": True,
    "backlog_growth_horizon": 300,
    "backlog_growth_scale": 30.0,
    "backlog_growth_weight": 0.16,
    "idle_reserve_target": 200.0,
    "idle_reserve_scale": 50.0,
    "idle_reserve_weight": 0.20,
    "local_predicted_oht_weight": 0.075,
    "local_reward_scale": 2.0,
    "rail_reward_mode": "free_flow_neutral_2",
    "rail_free_flow_neutral_ratio": 2.0,
    "rail_tat_weight": 30.0,
    "rail_tat_clip": 1.0,
    "tat_raw_clip": None,
    "tat_confidence_ramp": False,
    "action_scale": 0.05,
    "exploration_noise_std": 0.10,
    "exploration_noise_final_std": 0.02,
    "exploration_noise_anneal_steps": 100_000,
    "exploration_schedule_version": EXPLORATION_SCHEDULE_VERSION,
    "resume_refill_version": "global_step_v2",
    "checkpoint_rng_version": "exploration_rng_v2",
    "checkpoint_directory_version": "contextual_checkpoint_dir_slug_v1",
    "episode_burnin_version": "deterministic_policy_burnin_v1",
    "send_cost_logging_version": "post_send_v2",
    "protocol_version": "single_end_time_v2",
    "diagnostic_schema_version": (
        "contextual_reward_diagnostic_v16_leading_indicators"
    ),
    "wandb_metric_schema_version": "contextual_wandb_compact_v1",
    "rail_budget_calibration": "median_abs_nonzero_active_postclip",
    "arrival_tracking_source": "PClient.JOB_DIC.Job.ID_command_id",
    "arrival_tracking_reuse_fail_safe": True,
    "diagnostic_only_leading_indicators": (
        "predicted_oht_distribution,backlog_idle_op_rollups,route_ratio_tail,"
        "arrival_completion_flow,oht_utilization,leadlag,composite_pressure"
    ),
    "centering": False,
    "replay_sampling_version": REPLAY_SAMPLING_VERSION,
    "tat_termination_grace_steps": 10_000,
    "early_stop_tat_threshold": 170.0,
    "tat_above_threshold_patience": 300,
    "terminal_tat_penalty": -20.0,
    "note": "tatonesided_unbounded_grace10k_patience300_terminal20",
    "description": (
        "Reward N changes only Reward M continuous TAT behavior: global TAT "
        "is a one-sided linear penalty above 160 with no saturation and no "
        "tat_raw_clip. The 10,000-step grace, 170 threshold, 300-step "
        "patience, replayed terminal -20, and all non-TAT Reward M terms are "
        "unchanged."
    ),
}


def _make_run_name(exp_meta):
    stamp = datetime.now().strftime("%m%d_%H%M")
    return (
        f"run_{stamp}_{exp_meta['cost_structure']}_"
        f"{exp_meta['action_range']}_{exp_meta['reward_version']}_"
        f"{exp_meta['note']}"
    )


WANDB_METRIC_KEYS = (
    "env/step", "env/episode", "env/sim_time", "env/tat", "env/operation_rate",
    "env/queued", "env/waiting", "env/transferring", "env/completed",
    "env/termination_reason", "episode/step",
    "burnin/active", "burnin/remaining_steps",
    "burnin/has_trained_policy", "burnin/action_source",
    "termination/done", "termination/by_queue", "termination/by_tat",
    "protocol/stale_sim_time_ticks",
    "action/policy_mean", "action/policy_std", "action/policy_min",
    "action/policy_max", "action/policy_saturation_ratio",
    "action/cross_rail_policy_std", "action/cross_rail_applied_std",
    "action/cross_rail_noise_std",
    "action/cross_rail_noise_residual_std",
    "action/cross_rail_policy_range",
    "action/policy_to_noise_variance_ratio",
    "action/noise_abs_mean", "action/noise_residual_abs_mean",
    "action/clipped_fraction", "action/positive_clip_ratio",
    "action/negative_clip_ratio",
    "action/noise_suppressed_by_clip_mean",
    "action/policy_positive_saturation_ratio",
    "action/policy_negative_saturation_ratio",
    "action/policy_temporal_delta_mean",
    "action/policy_temporal_delta_std",
    "action/exploratory_mean", "action/exploratory_std",
    "action/exploratory_temporal_delta_mean",
    "action/exploratory_temporal_delta_std",
    "action/exploration_noise_std",
    "action/applied_mean", "action/applied_std",
    "action/applied_temporal_std",
    "action/applied_controlled_mean", "action/applied_controlled_std",
    "action/applied_controlled_min", "action/applied_controlled_max",
    "action/applied_scale", "action/mode_region_b_rl",
    "action/version_region_b_rl_v1",
    "b_rl/mean", "b_rl/std", "b_rl/min", "b_rl/max",
    "boundary/action_abs_max", "boundary/cost_baseline_abs_error_max",
    "reward/global_raw",
    "reward/global_component", "reward/local_raw_mean",
    "reward/local_raw_std", "reward/local_scaled_mean",
    "reward/local_scaled_std", "reward/local_component_mean",
    "reward/local_component_std", "reward/total_tat_level",
    "reward/rail_tat_penalty_mean", "reward/smooth_penalty_mean",
    "reward/rail_tat_raw_mean", "reward/rail_tat_weighted_preclip_mean",
    "reward/rail_tat_cycle_count",
    "reward/rail_tat_controlled_assignment_count",
    "reward/rail_tat_uncontrolled_assignment_count",
    "reward/total_mean", "reward/total_std", "reward/finite_ratio",
    "reward/smooth_control_delta_mean", "reward/smooth_control_delta_max",
    "reward/step_reward", "reward/global_norm", "reward/local_norm",
    "reward/rail_tat", "reward/rail_tat_event_count",
    "reward/rail_tat_sum", "reward/rail_tat_mean",
    "reward/rail_tat_vector_mean", "reward/rail_tat_event_mean",
    "reward/rail_tat_max",
    "reward/alpha", "reward/backlog",
    "reward/smooth_mean",
    "curriculum/action_scale",
    "global/tat", "global/op_rate", "global/queued",
    "global/queued_jobs", "global/waiting", "global/transfer",
    "global/completed",
    "job/mean_wait_priority", "job/mean_reassign", "job/queued",
    "dispatch/mode_first_match", "dispatch/mode_neutral_path_cost",
    "dispatch/mode_live_td7_path_cost",
    "dispatch/cost_mode_active",
    "dispatch/cost_snapshot_ready",
    "dispatch/cost_snapshot_ready_ratio",
    "dispatch/cost_snapshot_age_steps",
    "dispatch/eligible_job_count_total",
    "dispatch/candidate_count_mean",
    "dispatch/candidate_count_mean_per_eligible_job",
    "dispatch/zero_candidate_ratio_per_eligible_job",
    "dispatch/selected_ratio_per_eligible_job",
    "dispatch/selected_pickup_hops_mean",
    "dispatch/first_match_pickup_hops_mean",
    "dispatch/first_match_path_cost_mean",
    "dispatch/selected_path_cost_mean",
    "dispatch/cost_saving_vs_first_mean",
    "dispatch/cost_saving_vs_first_ratio_mean",
    "dispatch/candidate_cost_spread_mean",
    "dispatch/multi_candidate_ratio",
    "dispatch/changed_from_first_ratio",
    "dispatch/strict_cost_improvement_ratio",
    "dispatch/cost_snapshot_fallback_ratio",
    "dispatch/selection_margin_mean",
    "dispatch/selection_changed_from_first_match_ratio",
    "dispatch/best_second_margin_mean",
    "dispatch/cost_snapshot_fallback_count",
    "dispatch/invalid_cost_count",
    "oht/idle_count", "oht/move_to_load", "oht/move_to_unload",
    "oht/loading", "oht/unloading",
    "replay/size", "replay/size_env_steps", "replay/size_logical_transitions",
    "replay/action_enabled_env_steps", "replay/reward_mean",
    "replay/reward_std", "replay/policy_action_std",
    "replay/applied_action_std", "replay/done_ratio",
    "learner/critic_loss",
    "learner/actor_loss_last", "learner/actor_grad_norm_last",
    "learner/actor_last_update_step", "learner/actor_updates_total",
    "learner/actor_updated_this_step",
    "learner/encoder_joint_critic_loss",
    "learner/applied_action_scale",
    "learner/replay_policy_action_std",
    "learner/replay_applied_action_std",
    "learner/critic_action_matches_applied",
    "critic/q1_mean", "critic/q2_mean", "critic/q_min", "critic/q_max",
    "critic/q_mean_delta_100", "critic/q_mean_delta_1000",
    "critic/q_abs_diff_mean", "critic/q_abs_diff_max",
    "critic/parameter_l2_distance", "critic/parameter_max_abs_diff",
    "critic/q1_loss", "critic/q2_loss",
    "critic/q1_grad_norm", "critic/q2_grad_norm",
    "critic/target_q_mean", "critic/td_error_mean", "critic/td_error_max",
    "grad/encoder_norm", "grad/critic_norm",
    "update/learner_count", "update/actor_count", "update/target_count",
    "learner/updates",
    "numeric/learner_finite_ratio",
    "sale/enabled", "sale/loss", "sale/zs_mean", "sale/zs_std",
    "sale/zs_abs_mean", "sale/zsa_mean", "sale/zsa_std",
    "sale/prediction_error_mean", "sale/prediction_error_max",
    "sale/online_grad_norm", "sale/fixed_grad_norm",
    "sale/target_fixed_grad_norm", "sale/fixed_online_distance",
    "sale/target_fixed_distance", "sale/update_count",
    "sale/fixed_generation",
    "lap/enabled", "lap/priority_mean", "lap/priority_max",
    "lap/priority_std", "lap/priority_min",
    "lap/td_error_mean", "lap/td_error_max",
    "lap/effective_sample_size", "lap/sample_probability_max",
    "lap/stale_update_reject_count",
    "lap/new_transition_max_priority",
    "value/current_target_q_min", "value/current_target_q_max",
    "value/fixed_target_q_min", "value/fixed_target_q_max",
    "timing/sale_forward_ms", "timing/sale_backward_ms",
    "attention/incoming_entropy", "attention/outgoing_entropy",
    "attention/incoming_max_weight", "attention/outgoing_max_weight",
    "runtime/observation_ms", "runtime/actor_inference_ms",
    "runtime/replay_push_ms", "runtime/replay_sample_ms",
    "runtime/learner_update_ms", "runtime/cost_apply_ms",
    "runtime/send_cost_ms", "runtime/checkpoint_ms",
    "runtime/total_algorithm_ms", "runtime/total_ms",
)

WANDB_METRIC_KEYS += (
    "reward/rail/mode", "reward/rail/free_flow_neutral_ratio",
    "reward/global/total_tat", "reward/global/tat_reference",
    "reward/global/tat_error", "reward/global/tat_weight",
    "reward/global/tat_signal_available",
    "reward/global/tat_raw_preclip", "reward/global/tat_raw_postclip",
    "reward/global/tat_confidence",
    "reward/global/tat_confidence_n0", "reward/global/tat_raw_ramped",
    "reward/global/completed_episode", "reward/global/completed_delta",
    "reward/global/op_rate", "reward/global/op_reference",
    "reward/global/op_error", "reward/global/op_delta",
    "reward/global/op_weight",
    "reward/global/op_raw", "reward/global/backlog",
    "reward/global/backlog_weight", "reward/global/backlog_raw",
    "reward/global/backlog_growth_enabled",
    "reward/global/backlog_growth_horizon",
    "reward/global/backlog_growth_scale",
    "reward/global/backlog_growth_weight",
    "reward/global/backlog_growth_signal",
    "reward/global/backlog_growth_raw",
    "reward/global/backlog_growth_component",
    "reward/global/idle_oht_count", "reward/global/idle_reserve_target",
    "reward/global/idle_reserve_scale", "reward/global/idle_reserve_weight",
    "reward/global/idle_reserve_signal", "reward/global/idle_reserve_raw",
    "reward/global/idle_reserve_component",
    "reward/global/raw_sum", "reward/global/raw_decomposition_error",
    "reward/local/raw_decomposition_error_max",
    "reward/contribution/global_mean", "reward/contribution/local_mean",
    "reward/contribution/rail_tat_mean",
    "reward/contribution/smooth_mean",
    "reward/contribution/total_mean", "reward/contribution/sum_error",
    "reward/scale/global_abs_mean", "reward/scale/local_abs_mean",
    "reward/scale/rail_tat_abs_mean", "reward/scale/smooth_abs_mean",
    "reward/scale/total_abs_mean", "reward/scale/global_abs_share",
    "reward/scale/local_abs_share", "reward/scale/rail_tat_abs_share",
    "reward/scale/smooth_abs_share", "reward/scale/abs_share_sum_error",
    "reward/config/global_alpha", "reward/config/local_alpha",
    "reward/config/local_reward_scale", "reward/config/rail_tat_weight",
    "reward/config/rail_tat_clip", "reward/config/tat_raw_clip",
    "reward/config/smooth_weight_effective",
    "reward/terminal_penalty",
    "reward/global/tat_raw", "reward/global/tat_component",
    "reward/global/backlog_component", "reward/global/op_component",
    "reward/global/global_component",
    "reward/local/predicted_oht_weight",
    "reward/local/predicted_oht_component_abs_mean",
    "reward/local/local_reward_scale",
    "reward/local/local_component_abs_mean",
    "reward/local/predicted_abs_share", "reward/local/oht_abs_share",
    "reward/local/stop_abs_share", "reward/local/capacity_abs_share",
    "reward/rail/neutral_ratio", "reward/rail/weight", "reward/rail/clip",
    "reward/rail/nonzero_raw_abs_mean",
    "reward/rail/nonzero_weighted_abs_mean",
    "reward/rail/nonzero_postclip_abs_mean",
    "reward/scale/global_representative",
    "reward/scale/local_representative",
    "reward/scale/rail_active_representative",
    "reward/scale/global_to_local", "reward/scale/global_to_rail",
    "reward/scale/local_to_rail", "reward/scale/main_balance_error",
    "reward/scale/smooth_excess_warning",
    "reward/sign/good_state_sample_count",
    "reward/sign/good_state_total_mean",
    "reward/sign/bad_state_sample_count",
    "reward/sign/bad_state_total_mean", "reward/sign/good_minus_bad",
    "reward/sign/good_state_sufficient", "reward/sign/bad_state_sufficient",
) + tuple(
    f"reward/budget/{name}"
    for name in (
        "tat_abs", "op_abs", "backlog_level_abs", "backlog_growth_abs",
        "backlog_flow_abs", "idle_reserve_abs", "current_oht_abs",
        "predicted_oht_abs", "stop_abs", "capacity_abs", "rail_active_abs",
        "rail_overall_abs", "smooth_abs", "tat_share",
        "predicted_oht_share", "backlog_flow_share", "idle_reserve_share",
        "rail_active_share", "op_share", "other_share", "share_sum_error",
        "target_band_violation_count",
    )
) + tuple(
    f"reward/local/{term}_raw_{stat}"
    for term in ("oht", "predicted", "stop", "idle", "capacity")
    for stat in ("mean", "std", "abs_mean")
)

WANDB_METRIC_KEYS += (
    "reward/rail/cycle_count",
    "reward/rail/reward_applied_cycle_count",
    "reward/rail/skipped_cycle_count",
    "reward/rail_tat/effective_oht_tat_mean",
    "reward/rail_tat/effective_oht_tat_p50",
    "reward/rail_tat/effective_oht_tat_p90",
    "reward/rail_tat/effective_oht_tat_p95",
    "reward/rail_tat/signed_excess_mean",
    "reward/rail_tat/signed_excess_p50",
    "reward/rail_tat/signed_excess_p95",
    "reward/rail_tat/reward_cycle_ratio",
    "reward/rail_tat/penalty_cycle_ratio",
    "reward/rail_tat/zero_cycle_ratio",
    "reward/rail_tat/route_time_mean",
    "reward/rail_tat/route_free_flow_available_ratio",
    "reward/rail_tat/route_free_flow_time_mean",
    "reward/rail_tat/route_delay_ratio_mean",
    "reward/rail_tat/route_delay_ratio_p95",
    "reward/rail_tat/effective_to_freeflow_ratio_mean",
    "reward/rail_tat/service_residual_time_mean",
    "reward/rail/clip_assignment_ratio", "reward/rail/clip_removed_ratio",
    "reward/rail/route_ratio_mean", "reward/rail/route_ratio_p50",
    "reward/rail/route_ratio_p90", "reward/rail/route_ratio_p95",
    "reward/rail/cycle_reward_raw_mean", "reward/rail/cycle_reward_raw_p10",
    "reward/rail/cycle_reward_raw_p50", "reward/rail/cycle_reward_raw_p90",
    "reward/rail/cycle_reward_raw_p95", "reward/rail/positive_cycle_ratio",
    "reward/rail/negative_cycle_ratio", "reward/rail/zero_cycle_ratio",
    "reward/rail/nonzero_reward_abs_mean", "reward/rail/nonzero_reward_abs_p50",
    "reward/rail/nonzero_reward_abs_p95",
    "reward/rail/weighted_preclip_abs_mean", "reward/rail/postclip_abs_mean",
)

WANDB_METRIC_KEYS += tuple(
    f"lead/{series}/delta_{horizon}"
    for series, horizons in (
        ("backlog", (100, 300, 500, 1_000)),
        ("queued", (100, 300, 500)),
        ("waiting", (100, 300, 500)),
        ("idle", (100, 300, 500, 1_000)),
        ("op", (100, 300, 500)),
    )
    for horizon in horizons
) + tuple(
    f"lead/op/ema_{horizon}" for horizon in (100, 300, 500)
) + tuple(
    f"lead/predicted_oht/{name}"
    for name in (
        "available", "mean", "p50", "p75", "p90", "p95", "p99",
        "max", "std", "top10_mean", "top50_mean", "top100_mean",
        "fraction_gt_5", "fraction_gt_10",
    )
) + tuple(
    f"lead/route_ratio/{name}"
    for name in (
        "available", "mean", "p50", "p75", "p90", "p95", "p99",
        "max", "ratio_gt_2", "negative_reward_cycle_ratio",
    )
) + tuple(
    f"lead/flow/{name}_{horizon}"
    for horizon in (100, 300, 500, 1_000)
    for name in (
        "arrival_count", "completion_count", "arrival_rate",
        "completion_rate", "imbalance_count", "imbalance_rate",
    )
) + tuple(
    f"leadlag/{feature}_vs_future_tat_{horizon}"
    for horizon in (200, 500, 1_000)
    for feature in (
        "predicted_mean", "predicted_p90", "predicted_p95", "backlog",
        "backlog_delta_300", "queue", "idle", "idle_delta_300", "op",
        "op_ema300", "transferring", "route_ratio_p90",
        "route_ratio_p95", "flow_imbalance", "composite",
    )
)

WANDB_METRIC_KEYS += (
    "lead/backlog/value", "lead/queued/value", "lead/waiting/value",
    "lead/idle/value", "lead/idle/reserve_signal", "lead/idle/below_target",
    "lead/op/value", "lead/op/above_078", "lead/op/above_080",
    "lead/op/consecutive_above_080", "lead/oht/idle",
    "lead/oht/move_to_load", "lead/oht/loading",
    "lead/oht/move_to_unload", "lead/oht/unloading",
    "lead/oht/transferring", "lead/oht/transferring_delta_300",
    "lead/oht/move_to_unload_delta_300", "lead/flow/arrival_available",
    "lead/flow/new_job_count", "lead/flow/duplicate_job_id_count",
    "lead/flow/job_id_reuse_count", "lead/composite_pressure",
    "lead/composite_available", "leadlag/sample_count_200",
    "leadlag/sample_count_500", "leadlag/sample_count_1000",
)

# Runtime diagnostics remain rich for JSONL and local debugging. W&B receives
# only this experiment-facing compact profile so charts stay interpretable and
# history exports remain small.
WANDB_METRIC_KEYS = (
    # Environment / performance state.
    "env/step",
    "env/episode",
    "episode/step",
    "env/tat",
    "env/operation_rate",
    "env/queued",
    "env/waiting",
    "env/transferring",
    "oht/idle_count",
    "termination/done",
    "termination/by_tat",
    "env/termination_reason",
    # Actor / applied control.
    "action/policy_mean",
    "action/policy_std",
    "action/cross_rail_policy_std",
    "action/policy_saturation_ratio",
    "action/clipped_fraction",
    "action/exploration_noise_std",
    "action/applied_mean",
    "action/applied_std",
    "curriculum/action_scale",
    "b_rl/mean",
    "b_rl/std",
    # Reward and representative contribution budget.
    "reward/total_mean",
    "reward/total_std",
    "reward/terminal_penalty",
    "reward/global/tat_component",
    "reward/global/op_component",
    "reward/global/backlog_component",
    "reward/global/backlog_growth_component",
    "reward/global/idle_reserve_component",
    "reward/local/predicted_oht_component_abs_mean",
    "reward/budget/tat_share",
    "reward/budget/op_share",
    "reward/budget/backlog_flow_share",
    "reward/budget/idle_reserve_share",
    "reward/budget/predicted_oht_share",
    "reward/budget/rail_active_share",
    "reward/budget/other_share",
    "reward/budget/share_sum_error",
    "reward/rail/route_ratio_mean",
    "reward/rail/route_ratio_p95",
    "reward/rail/positive_cycle_ratio",
    "reward/rail/negative_cycle_ratio",
    "reward/rail/clip_assignment_ratio",
    "reward/rail/clip_removed_ratio",
    "reward/scale/rail_active_representative",
    # Learner / critic / replay health.
    "learner/critic_loss",
    "learner/actor_loss_last",
    "learner/actor_grad_norm_last",
    "critic/q1_mean",
    "critic/q2_mean",
    "critic/target_q_mean",
    "critic/q_abs_diff_mean",
    "critic/td_error_mean",
    "critic/td_error_max",
    "grad/encoder_norm",
    "grad/critic_norm",
    "learner/actor_updates_total",
    "learner/updates",
    "replay/size_env_steps",
    "replay/reward_mean",
    "replay/reward_std",
    "numeric/learner_finite_ratio",
    # SALE essentials only.
    "sale/loss",
    "sale/prediction_error_mean",
    "sale/online_grad_norm",
    "sale/fixed_online_distance",
    # Compact leading state: 300-step change and 500-step future TAT.
    "lead/backlog/value",
    "lead/backlog/delta_300",
    "lead/idle/value",
    "lead/idle/delta_300",
    "lead/idle/reserve_signal",
    "lead/op/value",
    "lead/predicted_oht/mean",
    "lead/predicted_oht/p95",
    "lead/route_ratio/p95",
    "lead/flow/arrival_available",
    "lead/flow/imbalance_count_300",
    "lead/composite_pressure",
    "leadlag/predicted_p95_vs_future_tat_500",
    "leadlag/backlog_delta_300_vs_future_tat_500",
    "leadlag/idle_delta_300_vs_future_tat_500",
    "leadlag/flow_imbalance_vs_future_tat_500",
    "leadlag/composite_vs_future_tat_500",
    "leadlag/sample_count_500",
)

REMOVED_WANDB_PREFIXES = (
    "lap/",
    "dispatch/",
    "burnin/",
    "attention/",
    "boundary/",
    "protocol/",
    "value/",
)
if len(WANDB_METRIC_KEYS) != len(set(WANDB_METRIC_KEYS)):
    raise RuntimeError("compact W&B metric schema contains duplicate keys")
if any(key.startswith(REMOVED_WANDB_PREFIXES) for key in WANDB_METRIC_KEYS):
    raise RuntimeError("removed metric group leaked into compact W&B schema")


def runtime_exp_meta(config) -> dict:
    meta = dict(EXP_META)
    reward_config = ContextualRewardConfig(
        action_mode=config.action_mode,
        smooth_b_rl_weight=config.smooth_b_rl_weight,
        smooth_exp_residual_weight=config.smooth_exp_residual_weight,
        tat_confidence_n0=config.tat_confidence_n0,
        tat_confidence_ramp=config.tat_confidence_ramp,
        local_reward_scale=config.local_reward_scale,
        tat_weight=config.tat_weight,
        op_weight=config.op_weight,
        backlog_weight=config.backlog_weight,
        backlog_growth_enabled=config.backlog_growth_enabled,
        backlog_growth_horizon=config.backlog_growth_horizon,
        backlog_growth_scale=config.backlog_growth_scale,
        backlog_growth_weight=config.backlog_growth_weight,
        idle_reserve_target=config.idle_reserve_target,
        idle_reserve_scale=config.idle_reserve_scale,
        idle_reserve_weight=config.idle_reserve_weight,
        local_predicted_oht_weight=config.local_predicted_oht_weight,
        rail_reward_mode=config.rail_reward_mode,
        rail_free_flow_neutral_ratio=config.rail_free_flow_neutral_ratio,
        rail_baseline_ratio_reference=config.rail_baseline_ratio_reference,
        rail_tat_weight=config.reward_rail_tat_weight,
        rail_tat_clip=config.reward_rail_tat_clip,
        tat_raw_clip=config.tat_raw_clip,
    )
    sale = bool(config.sale_enabled)
    lap = bool(config.lap_enabled)
    action_mode = str(config.action_mode)
    mode_version = action_version(action_mode)
    meta["algorithm_version"] = ALGORITHM_VERSION
    meta["algorithm_variant"] = (
        f"{contextual_algorithm_variant(sale, lap)}_{mode_version}"
    )
    meta["critic_initialization"] = CRITIC_INITIALIZATION
    meta["checkpoint_version"] = CHECKPOINT_VERSION
    meta["action_mode"] = action_mode
    meta["action_version"] = mode_version
    meta["dispatch_mode"] = str(config.dispatch_mode)
    meta["dispatch_selection_version"] = DISPATCH_SELECTION_VERSION
    meta["note"] = (
        f"{meta['note']}_dispatch_{str(config.dispatch_mode).replace('-', '_')}"
    )
    meta["sale"] = sale
    meta["lap"] = lap
    meta["sale_version"] = SALE_VERSION
    meta["lap_version"] = LAP_VERSION
    meta["replay_sampling_version"] = REPLAY_SAMPLING_VERSION
    meta["replay_sampling_mode"] = config.replay_sampling_mode
    meta["critic_loss_mode"] = config.critic_loss_mode
    meta["action_scale"] = float(config.action_scale)
    meta["reward_global_alpha"] = float(reward_config.global_alpha)
    meta["reward_local_alpha"] = float(reward_config.local_alpha)
    meta["local_reward_scale"] = float(reward_config.local_reward_scale)
    meta["local_predicted_oht_weight"] = float(
        reward_config.local_predicted_oht_weight
    )
    meta["rail_tat_weight"] = float(reward_config.rail_tat_weight)
    meta["rail_tat_clip"] = float(reward_config.rail_tat_clip)
    meta["rail_reward_mode"] = reward_config.rail_reward_mode
    meta["rail_free_flow_neutral_ratio"] = float(
        reward_config.rail_free_flow_neutral_ratio
    )
    meta["tat_raw_clip"] = (
        None
        if reward_config.tat_raw_clip is None
        else float(reward_config.tat_raw_clip)
    )
    meta["tat_confidence_n0"] = float(reward_config.tat_confidence_n0)
    meta["tat_confidence_ramp"] = bool(reward_config.tat_confidence_ramp)
    meta["tat_reference"] = float(reward_config.tat_reference)
    meta["tat_weight"] = float(reward_config.tat_weight)
    meta["op_reference"] = float(reward_config.op_reference)
    meta["op_weight"] = float(reward_config.op_weight)
    meta["backlog_weight"] = float(reward_config.backlog_weight)
    meta["backlog_growth_enabled"] = bool(
        reward_config.backlog_growth_enabled
    )
    meta["backlog_growth_horizon"] = int(
        reward_config.backlog_growth_horizon
    )
    meta["backlog_growth_scale"] = float(
        reward_config.backlog_growth_scale
    )
    meta["backlog_growth_weight"] = float(
        reward_config.backlog_growth_weight
    )
    meta["idle_reserve_target"] = float(reward_config.idle_reserve_target)
    meta["idle_reserve_scale"] = float(reward_config.idle_reserve_scale)
    meta["idle_reserve_weight"] = float(reward_config.idle_reserve_weight)
    meta["early_stop_tat_threshold"] = float(
        config.early_stop_tat_threshold
    )
    meta["tat_termination_grace_steps"] = int(
        config.tat_termination_grace_steps
    )
    meta["tat_above_threshold_patience"] = int(
        config.tat_above_threshold_patience
    )
    meta["terminal_tat_penalty"] = float(config.terminal_tat_penalty)
    meta["curriculum_end_step"] = int(config.curriculum_end_step)
    meta["replay_capacity_env_steps"] = int(
        config.replay_capacity_env_steps
    )
    if action_mode == REGION_B_RL:
        meta["cost_structure"] = "b_rl"
        meta["action_range"] = (
            "b_rl_0.0-1.0_"
            f"curriculum_{float(config.curriculum_scale_start):g}-"
            f"{float(config.curriculum_scale_end):g}"
        )
    else:
        meta["cost_structure"] = "baseline_exp_residual"
        meta["action_range"] = f"{float(config.action_scale):g}-scaled"
    meta["exploration_noise_std"] = float(config.exploration_noise_std)
    meta["exploration_noise_final_std"] = float(
        min(config.exploration_noise_std, config.exploration_noise_final_std)
    )
    meta["exploration_noise_anneal_steps"] = int(
        config.exploration_noise_anneal_steps
    )
    meta["episode_burnin_steps"] = int(config.episode_burnin_steps)
    meta["exploration_noise_anneal_start_step"] = int(config.warmup_steps)
    meta["exploration_noise_anneal_end_step"] = int(
        config.warmup_steps + config.exploration_noise_anneal_steps
    )
    meta["exploration_schedule_version"] = EXPLORATION_SCHEDULE_VERSION
    replay_mode = "lap" if lap else "uniform"
    meta["replay"] = (
        f"{config.replay_sampling_mode}_{replay_mode}_"
        f"{int(config.replay_capacity_env_steps)}"
    )
    meta["description"] = (
        f"{EXP_META['description']} Directional contextual TD7 with "
        f"SALE={'on' if sale else 'off'}, LAP={'on' if lap else 'off'}, "
        f"replay_sampling={config.replay_sampling_mode}, "
        f"action_mode={action_mode}, dispatch_mode={config.dispatch_mode}, "
        f"critic_loss={config.critic_loss_mode}, "
        "independently initialized Q1/Q2 heads, "
        f"reward N ({REWARD_VERSION}, global/local="
        f"{reward_config.global_alpha:g}/{reward_config.local_alpha:g}), "
        "reward_normalizer=removed, fixed_scale=on, "
        f"replay_capacity={int(config.replay_capacity_env_steps)}, "
        f"curriculum_end_step={int(config.curriculum_end_step)}, "
        f"exploration={float(config.exploration_noise_std):g}->"
        f"{meta['exploration_noise_final_std']:g} over global steps "
        f"{int(config.warmup_steps)}.."
        f"{int(config.warmup_steps + config.exploration_noise_anneal_steps)}, "
        f"episode_burnin_steps={int(config.episode_burnin_steps)}, "
        "control-space smoothing, "
        "single-field simulator end-time handshake, and fail-closed SimTime "
        "progress validation."
    )
    return meta


class ContextualWandbLogger:
    def __init__(self, config):
        self.enabled = bool(config.wandb_enabled)
        self.run = None
        self._finished = False
        if not self.enabled:
            return
        import wandb

        meta = runtime_exp_meta(config)
        assert meta["action_scale"] == float(config.action_scale)
        assert meta["exploration_noise_std"] == float(
            config.exploration_noise_std
        )
        assert meta["exploration_noise_final_std"] == float(
            min(
                config.exploration_noise_std,
                config.exploration_noise_final_std,
            )
        )
        runtime_config = dict(vars(config))
        runtime_config["EXP_META"] = meta
        self.run = wandb.init(
            project="oht-routing-contextual-td7",
            config=runtime_config,
            name=_make_run_name(meta),
            notes=meta["description"],
        )

    def log(self, diagnostics, step):
        if self.run is None:
            return
        payload = {
            key: diagnostics[key]
            for key in WANDB_METRIC_KEYS
            if key in diagnostics
        }
        self.run.log(payload, step=int(step))

    def _finish(self, status, *, reason=None, step=None):
        if self.run is None or self._finished:
            return
        self._finished = True
        try:
            self.run.summary["run/status"] = status
            if reason is not None:
                self.run.summary["run/failure_reason"] = str(reason)
            if step is not None:
                self.run.summary["run/failure_step"] = int(step)
            self.run.finish()
        except Exception:
            # Logging must never mask the runtime/learner exception.
            pass

    def finish_success(self):
        self._finish("completed")

    def finish_failed(self, reason, step=None):
        self._finish("failed", reason=reason, step=step)

    def finish_interrupted(self):
        self._finish("interrupted")
