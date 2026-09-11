"""Mode-aware W&B schemas and lightweight contextual runtime logger."""

from __future__ import annotations

from datetime import datetime

from oht_routing.algorithms.rl.contextual_td7 import (
    CRITIC_TARGET_UBOC,
    ContextualLearnerConfig,
    REPLAY_EVICTION_FIFO,
    contextual_algorithm_variant,
)
from oht_routing.mdp.action import (
    FREE_FLOW_RESIDUAL,
    REGION_B_RL,
)
from oht_routing.mdp.observation import (
    ACTOR_GLOBAL_DIM,
    CRITIC_EXTRA_DIM,
    LOCAL_PHYSICAL_DIM,
    OBSERVATION_VERSION,
    RELATION_DIM,
)
from oht_routing.mdp.reward.builder import ContextualRewardConfig
from oht_routing.mdp.reward.config import (
    REWARD_VERSION,
    TAT_PENALTY_START,
    reward_contract,
)
from oht_routing.runtime.stages import (
    STAGE_TWO,
    STAGE_TWO_STAGE1_POLICY_STEPS,
)
from oht_routing.version import CONTEXTUAL_VERSION

_EXP_REWARD_VERSION = REWARD_VERSION
_EXP_REWARD_CONTRACT = reward_contract(_EXP_REWARD_VERSION)


def _tat_formula(reward_config) -> str:
    """One-sided TAT penalty, rendered from the live coefficients."""
    return (
        f"-{reward_config.tat_weight:g}*max(TotalTat-"
        f"{int(TAT_PENALTY_START)},0)/{reward_config.tat_reference:g}"
        "_for_TotalTat_gt_0"
    )

# Reward Q calibration targets, as projected onto the Stage 2 segment of run
# 017jxhcc (18,826 points, TotalTat p50 172.8) by rescaling its measured
# Reward P term magnitudes with the Reward Q coefficient factors. The density
# term has no Stage 2 measurement and is estimated from the
# density-to-predicted_oht magnitude ratio of 1.32 observed on
# results/environment_capture/capture_20260821_081812_v2.2.0_actor_inference.
# These are observed shares to check the next run against, not coefficient
# percentages. Below the 160 s TAT threshold the TAT term is zero by contract,
# so the same weights read differently there; see EXPERIMENTS_v9.md.
_REWARD_Q_TARGET_SHARES = {
    "rail_outcome": 0.3052,
    "tat": 0.2994,
    "density": 0.1891,
    "backlog_level": 0.1183,
    "backlog_growth": 0.0278,
    "predicted_oht": 0.0260,
    "idle_reserve": 0.0152,
    "smooth": 0.0123,
    "stop_time": 0.0067,
}
# Delay-carrying terms (stop_time + density + rail_outcome) hold 50.1% against
# 3.3% under Reward P; TAT drops from 49.7% to 29.9% purely by dilution, and
# the demand forecast from 20.1% to 2.6%.

EXP_META = {
    "version": CONTEXTUAL_VERSION,
    "cost_structure": "b_rl",
    "action_range": "0.0-1.0",
    "topology": "directed_15in_15out_controlled_centers_v3",
    "observation_version": OBSERVATION_VERSION,
    "observation": "compact_local14_actor_global6_shared_tat_v6",
    "local_physical_dim": LOCAL_PHYSICAL_DIM,
    "rail_embedding_dim": 8,
    "use_attention": False,
    "neighbor_aggregation": "directional_flat_projection",
    "actor_global_dim": ACTOR_GLOBAL_DIM,
    "global_dim": ACTOR_GLOBAL_DIM,
    "critic_extra_dim": CRITIC_EXTRA_DIM,
    "relation_dim": RELATION_DIM,
    "neighbor_count_per_direction": 15,
    "actor_has_total_tat": True,
    "critic_has_total_tat": True,
    "shared_total_tat_feature": "normalized_total_tat_s",
    "critic_direct_total_tat_feature": "normalized_total_tat_s",
    "obs/version": OBSERVATION_VERSION,
    "obs/local_dim": LOCAL_PHYSICAL_DIM,
    "obs/incoming_neighbors": 15,
    "obs/outgoing_neighbors": 15,
    "obs/relation_dim": RELATION_DIM,
    "obs/actor_global_dim": ACTOR_GLOBAL_DIM,
    "obs/critic_extra_dim": CRITIC_EXTRA_DIM,
    "obs/actor_has_total_tat": 1,
    "obs/critic_has_total_tat": 1,
    "previous_action_input": (
        "actor_and_critic_previous_applied_action_separate_from_encoder"
    ),
    "critic_action_input": "previous_applied_plus_current_candidate_action",
    "num_stacks": 1,
    "stack_interval": 1,
    "reward_version": _EXP_REWARD_VERSION,
    "tat_signal": _EXP_REWARD_CONTRACT.tat_signal_description,
    "calibration_status": (
        "v9_reward_q_delay_weighted_local_and_rail_budget"
    ),
    "reward_global_alpha": 0.5,
    "reward_local_alpha": 0.5,
    "tat_reference": 165.0,
    "tat_penalty_start": 160.0,
    "tat_penalty_threshold": 160.0,
    "tat_one_sided": True,
    "tat_formula": _tat_formula(
        ContextualRewardConfig.for_version(_EXP_REWARD_VERSION)
    ),
    "reward_tat_input": "pclient.TotalTat",
    "reward_recent_300_tat_used": False,
    "reward_tat_zero_policy": "unavailable_zero_contribution",
    "reward_tat_at_or_below_threshold_policy": "zero_contribution",
    "reward_tat_negative_policy": "invalid_fail_fast",
    "recent_tat_diagnostic_window_seconds": 300.0,
    "tat_weight": 4.0,
    "op_reference": 0.80,
    "op_weight": 0.0,
    "use_op": False,
    "backlog_weight": 0.0007,
    "backlog_growth_enabled": True,
    "backlog_growth_horizon": 300,
    "backlog_growth_scale": 30.0,
    "backlog_growth_weight": 0.17,
    "idle_reserve_target": 200.0,
    "idle_reserve_scale": 50.0,
    "idle_reserve_weight": 0.09,
    "local_oht_weight": 0.0,
    "local_predicted_oht_weight": 0.01,
    "local_stop_weight": 0.30,
    "local_stop_aggregation": "sum_unclipped",
    "local_density_weight": 5.5,
    "local_density_signal": "oht_count_per_rail_metre",
    "local_idle_weight": 0.0,
    "local_capacity_weight": 0.0,
    "local_reward_scale": 2.0,
    "rail_reward_mode": "free_flow_neutral_1_7",
    "rail_free_flow_neutral_ratio": 1.70,
    "rail_tat_weight": 660.0,
    "rail_tat_clip": 22.0,
    "reward_target_shares": _REWARD_Q_TARGET_SHARES,
    "action_scale": 0.05,
    "rl_cost_lambda": 0.5,
    "rail_cost_formula": "t_ff+d_w*(c+1)*b_rl",
    "congestion_count_offset": 1.0,
    "curriculum_scale_start": 0.05,
    "curriculum_scale_end": 1.0,
    "curriculum_shape": "geometric",
    "warmup_steps": 10_000,
    "state_normalizer_reuse": (
        "exact_contract_load_sets_effective_warmup_to_zero"
    ),
    "exploration_noise_std": 0.05,
    "exploration_noise_final_std": 0.05,
    "exploration_noise_anneal_steps": 100_000,
    "arrival_tracking_source": "PClient.JOB_DIC.Job.ID_command_id",
    "arrival_tracking_reuse_fail_safe": True,
    "diagnostic_only_leading_indicators": (
        "predicted_oht_distribution,backlog_idle_op_rollups,route_ratio_tail,"
        "arrival_completion_flow,oht_utilization,leadlag,composite_pressure"
    ),
    "centering": False,
    "replay_capacity_env_steps": 100_000,
    "replay_eviction_mode": REPLAY_EVICTION_FIFO,
    "replay_storage": (
        "packed_v5_local_u8_u16_critic_tat_f32_action_q15_lap_fp16"
    ),
    "replay_action_quantization_max_abs_error": 1.0 / (2.0 * 32_767.0),
    "batch_size": 1_024,
    "seed": 0,
    "tat_early_termination": True,
    "tat_termination_policy": "reward_profile",
    "tat_termination_configured_start_episode": 1,
    "tat_termination_start_episode": 1,
    "tat_termination_grace_steps": 10_000,
    "early_stop_tat_threshold": 200.0,
    "tat_above_threshold_patience": 300,
    "terminal_tat_penalty": -20.0,
    "stage2_contract": "per_episode_stage1_2000_then_stage2_to_45000",
    "stage1_policy_load": (
        "frozen_prefix_plus_fresh_stage2_policy_pipeline_warm_start"
    ),
    "stage2_policy_initialization": (
        "stage1_encoder_actor_sale_fixed_once_targets_synchronized"
    ),
    "stage2_checkpoint_clock": "stage2_env_steps",
    "distributed_runtime": "central_gpu_owner_with_independent_collectors",
    "distributed_live_replay": "central_packed_ram",
    "distributed_inference": "deadline_bounded_dynamic_microbatch",
    "note": "ud7",
    "description": (
        "UD7: the critic Bellman target drops the clipped double-Q minimum "
        "over two critics for the UBOC aggregation over N critics, "
        "mean - beta * unbiased_std with beta = 1/sqrt(pi) = 0.5641896 and "
        "N = 5. The penalty scales with how much the ensemble still "
        "disagrees, so it is largest early and fades as the critics "
        "converge; the value clip now applies to that aggregate rather than "
        "per critic, because the minimum commutes with a shared clamp but "
        "the mean and spread do not. The policy gradient reads the ensemble "
        "mean instead of the first head. Everything else is the v9 "
        "contract: SALE, LAP, decoupled encoders, delayed policy updates, "
        "target policy smoothing, observations, action mapping, replay and "
        "reward are unchanged. Pass --critic-target-mode cdq --num-critics "
        "2 for the TD7 baseline arm."
    ),
}


def _make_run_name(exp_meta):
    stamp = datetime.now().strftime("%m%d_%H%M")
    return (
        f"run_{stamp}_{exp_meta['version']}_{exp_meta['cost_structure']}_"
        f"{exp_meta['action_range']}_{exp_meta['reward_version']}_"
        f"{exp_meta['note']}"
    )


WANDB_METRIC_KEYS = (
    "env/step", "env/episode", "env/sim_time", "env/tat", "env/operation_rate",
    "eval/final_tat", "runtime/acceleration_rate",
    "env/queued", "env/waiting", "env/transferring", "env/completed",
    "env/termination_reason", "episode/step",
    "stage/id", "stage/active_policy",
    "stage1/active", "stage1/remaining_steps",
    "stage1/applied_action_scale", "stage1/policy_frozen",
    "stage2/active", "stage2/env_steps", "stage2/episode_step",
    "termination/done", "termination/by_queue", "termination/by_tat",
    "termination/by_warmup", "termination/by_resume_warmstart",
    "resume_warmstart/active", "resume_warmstart/remaining_steps",
    "warmup/episode_boundary_sent",
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
    "reward/config/rail_tat_clip",
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
    "reward/local/density_abs_share",
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
    for term in ("oht", "predicted", "stop", "density", "idle", "capacity")
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
    "eval/final_tat",
    "runtime/acceleration_rate",
    "stage/id",
    "stage/active_policy",
    "stage1/active",
    "stage1/remaining_steps",
    "stage1/applied_action_scale",
    "stage1/policy_frozen",
    "stage2/active",
    "stage2/env_steps",
    "stage2/episode_step",
    "env/sim_time",
    "env/tat",
    "env/recent_completed_tat_300s_mean",
    "env/recent_completed_tat_300s_p90",
    "env/recent_completed_tat_300s_count",
    "env/recent_completed_tat_300s_available",
    "env/recent_completed_tat_300s_window_age",
    "env/recent_completed_tat_events_added",
    "env/recent_completed_tat_duplicate_events",
    "env/recent_completed_tat_missing_state5",
    "env/recent_completed_tat_ambiguous_entries",
    "env/operation_rate",
    "env/queued",
    "env/waiting",
    "env/transferring",
    "env/completed",
    "oht/idle_count",
    "termination/done",
    "termination/by_queue",
    "termination/by_tat",
    "termination/by_warmup",
    "termination/by_resume_warmstart",
    "resume_warmstart/active",
    "resume_warmstart/remaining_steps",
    "warmup/episode_boundary_sent",
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
    "rl/action_mean",
    "rl/action_std",
    "rl/action_abs_mean",
    "rail_cost/t_ff_mean",
    "rail_cost/congestion_mean",
    "rail_cost/residual_mean",
    "rail_cost/residual_abs_mean",
    "rail_cost/final_cost_mean",
    "curriculum/action_scale",
    "b_rl/mean",
    "b_rl/std",
    "cost/all_baseline_abs_error_max",
    # Observation-contract diagnostics.
    "observation/predicted_route10_pearson",
    "observation/predicted_route10_mae",
    "observation/predicted_route10_nonzero_agreement",
    "observation/predicted_route10_both_nonzero",
    "observation/predicted_route10_pearson_available",
    "observation/predicted_route10_union_nonzero_pearson",
    "observation/predicted_route10_union_nonzero_pearson_available",
    "observation/predicted_route10_union_nonzero_ratio",
    "observation/recent_completed_tat_300s_mean",
    "observation/recent_completed_tat_300s_available",
    # Locked Reward P components and contribution budget.
    "reward/total_mean",
    "reward/total_std",
    "reward/terminal_penalty",
    "reward/global/tat_raw",
    "reward/global/tat_component_raw",
    "reward/global/recent_completed_tat_300s_mean",
    "reward/global/recent_completed_tat_300s_p90",
    "reward/global/recent_completed_tat_300s_count",
    "reward/global/recent_completed_tat_300s_window_age",
    "reward/global/recent_completed_tat_events_added",
    "reward/global/total_tat",
    "reward/global/cumulative_total_tat",
    "reward/global/tat_penalty_start",
    "reward/global/tat_excess",
    "reward/global/tat_signal_available",
    "reward/global/backlog_component_raw",
    "reward/global/raw",
    "reward/global/component",
    "reward/local/raw_mean",
    "reward/local/raw_std",
    "reward/local/normalized_mean",
    "reward/local/normalized_std",
    "reward/local/component_mean",
    "reward/local/component_std",
    "local/oht_abs_mean",
    "local/oht_std",
    "local/pred_abs_mean",
    "local/pred_std",
    "local/stop_abs_mean",
    "local/stop_std",
    "local/density_abs_mean",
    "local/density_std",
    "local/idle_abs_mean",
    "local/idle_std",
    "local/capacity_abs_mean",
    "local/capacity_std",
    "reward/rail_tat_mean",
    "reward/smooth_penalty_mean",
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
    "reward/term_scale/abs_sum",
    "reward/term_scale/share_sum_error",
    "reward/term_scale/reconstruction_error",
    *(
        f"reward/term_scale/{name}_abs_mean"
        for name in (
            "tat", "op", "backlog_level", "backlog_growth",
            "idle_reserve", "current_oht", "predicted_oht", "stop_time",
            "density", "local_idle", "capacity", "rail_outcome",
            "smooth",
        )
    ),
    *(
        f"reward/term_share/{name}"
        for name in (
            "tat", "op", "backlog_level", "backlog_growth",
            "idle_reserve", "current_oht", "predicted_oht", "stop_time",
            "density", "local_idle", "capacity", "rail_outcome",
            "smooth",
        )
    ),
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
    # UBOC mechanism: the ensemble spread is the uncertainty, beta times it is
    # the correction actually applied, and the gap against the ensemble
    # minimum is the optimism UBOC buys back over clipped double-Q.
    "critic/q_ensemble_std_mean",
    "critic/target_ensemble_std_mean",
    "critic/target_uboc_penalty_mean",
    "critic/target_vs_min_gap_mean",
    "critic/q_grad_norm_min",
    "critic/parameter_pair_l2_mean",
    "grad/encoder_norm",
    "grad/critic_norm",
    "learner/actor_updates_total",
    "learner/updates",
    "replay/size_env_steps",
    "replay/reward_mean",
    "replay/reward_std",
    "numeric/learner_finite_ratio",
    # Runtime stages and safety checks.
    "runtime/state_collection_ms",
    "runtime/observation_build_ms",
    "runtime/tensor_conversion_ms",
    "runtime/host_to_device_ms",
    "runtime/encoder_actor_ms",
    "runtime/actor_inference_ms",
    "runtime/device_to_host_ms",
    "runtime/inference_queue_ms",
    "runtime/inference_microbatch_size",
    "runtime/cost_apply_ms",
    "runtime/replay_push_ms",
    "runtime/replay_sample_ms",
    "runtime/learner_update_ms",
    "runtime/checkpoint_ms",
    "runtime/send_cost_ms",
    "runtime/total_algorithm_ms",
    "runtime/total_ms",
    "runtime/data_capture_ms",
    "runtime/observation_build_calls_per_tick",
    "runtime/nonfinite_count",
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


ACTOR_INFERENCE_EXCLUDED_WANDB_PREFIXES = (
    "learner/",
    "critic/",
    "grad/",
    "replay/",
    "sale/",
    "update/",
    "numeric/learner",
)
ACTOR_INFERENCE_EXCLUDED_WANDB_KEYS = {
    "runtime/replay_push_ms",
    "runtime/replay_sample_ms",
    "runtime/learner_update_ms",
}
ACTOR_INFERENCE_WANDB_METRIC_KEYS = tuple(
    key
    for key in WANDB_METRIC_KEYS
    if not key.startswith(ACTOR_INFERENCE_EXCLUDED_WANDB_PREFIXES)
    and key not in ACTOR_INFERENCE_EXCLUDED_WANDB_KEYS
)


def wandb_metric_keys_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "training":
        return WANDB_METRIC_KEYS
    return ACTOR_INFERENCE_WANDB_METRIC_KEYS


def runtime_exp_meta(config) -> dict:
    meta = dict(EXP_META)
    reward_config = ContextualRewardConfig.for_version(
        config.reward_version,
        action_mode=config.action_mode,
    )
    contract = reward_config.contract
    meta["version"] = CONTEXTUAL_VERSION
    meta["reward_version"] = contract.version
    meta["tat_signal"] = contract.tat_signal_description
    meta["calibration_status"] = (
        f"{EXP_META['calibration_status']}_"
        f"locked_reward_{contract.version.lower()}"
    )
    meta["tat_formula"] = _tat_formula(reward_config)
    # The calibrated shares describe Reward Q only; N predates them.
    if contract.version == "Q":
        meta["reward_target_shares"] = dict(_REWARD_Q_TARGET_SHARES)
    else:
        meta.pop("reward_target_shares", None)
        meta["note"] = "rewardn_1y9sx4a5_coeffs"
        meta["description"] = (
            "Fallback to the Reward N coefficient set that ran W&B run "
            "1y9sx4a5 on 2026-08-20: TAT weight 11, operation-rate term back "
            "on at 4.0, backlog 0.0004, idle reserve 0.2, local "
            "oht/predicted/stop/capacity 0.3/0.075/0.3/0.1, no density term, "
            "rail-cycle weight 30 with clip 1.0 at the 2.0 neutral point. "
            "Only the coefficients are restored - the observation, action "
            "mapping, Stage 1/2 contract, replay, and network are the "
            "current v9 ones, so this is not a reproduction of that run."
        )
    sale = bool(config.sale_enabled)
    lap = bool(config.lap_enabled)
    action_mode = str(config.action_mode)
    actor_q_aggregation = ContextualLearnerConfig(
        critic_target_mode=config.critic_target_mode,
        actor_q_aggregation=config.actor_q_aggregation,
    ).resolved_actor_q_aggregation
    meta["num_critics"] = int(config.num_critics)
    meta["critic_target_mode"] = str(config.critic_target_mode)
    meta["uboc_beta"] = float(config.uboc_beta)
    meta["actor_q_aggregation"] = actor_q_aggregation
    meta["actor_q_aggregation_configured"] = str(config.actor_q_aggregation)
    meta["algorithm_variant"] = contextual_algorithm_variant(
        sale, lap, config.critic_target_mode, config.num_critics
    )
    meta["use_attention"] = bool(config.use_attention)
    meta["neighbor_aggregation"] = (
        "directional_cross_attention"
        if config.use_attention
        else "directional_flat_projection"
    )
    meta["action_mode"] = action_mode
    meta["dispatch_mode"] = str(config.dispatch_mode)
    meta["num_sim"] = int(config.num_sim)
    meta["sim_ports"] = (
        list(config.sim_ports) if config.sim_ports is not None else None
    )
    meta["distributed_enabled"] = bool(config.num_sim > 1)
    meta["note"] = (
        f"{meta['note']}_{config.critic_target_mode}"
        f"{int(config.num_critics)}"
        f"_r{contract.version}_s{int(config.num_stacks)}i"
        f"{int(config.stack_interval)}_dispatch_"
        f"{str(config.dispatch_mode).replace('-', '_')}_normreuse"
        f"{int(config.state_normalizer_warmup_bypass)}_b"
        f"{int(config.batch_size)}_evict{config.replay_eviction_mode}_"
        f"fullrefill{int(config.resume_inference_until_replay_full)}_"
        f"detfirst{int(config.resume_deterministic_first_episode)}_"
        f"warmstart{int(config.resume_warmstart_steps)}_"
        f"stage{int(config.stage or 0)}_"
        f"attn{int(config.use_attention)}_nsim{int(config.num_sim)}"
    )
    meta["sale"] = sale
    meta["lap"] = lap
    meta["replay_sampling_mode"] = config.replay_sampling_mode
    meta["replay_eviction_mode"] = config.replay_eviction_mode
    meta["num_stacks"] = int(config.num_stacks)
    meta["stack_interval"] = int(config.stack_interval)
    meta["critic_loss_mode"] = config.critic_loss_mode
    meta["residual_action_scale"] = (
        1.0
        if action_mode == FREE_FLOW_RESIDUAL
        else float(config.action_scale)
    )
    meta["rl_cost_lambda"] = float(config.rl_cost_lambda)
    if action_mode == FREE_FLOW_RESIDUAL:
        effective_scale_start = effective_scale_end = 1.0
    elif action_mode == REGION_B_RL:
        effective_scale_start = float(config.curriculum_scale_start)
        effective_scale_end = float(config.curriculum_scale_end)
    else:
        effective_scale_start = effective_scale_end = float(
            config.action_scale
        )
    meta["action_scale"] = effective_scale_start
    meta["effective_action_scale_start"] = effective_scale_start
    meta["effective_action_scale_end"] = effective_scale_end
    meta["reward_global_alpha"] = float(reward_config.global_alpha)
    meta["reward_local_alpha"] = float(reward_config.local_alpha)
    meta["local_reward_scale"] = float(reward_config.local_reward_scale)
    meta["local_predicted_oht_weight"] = float(
        reward_config.local_predicted_oht_weight
    )
    meta["local_oht_weight"] = float(reward_config.local_oht_weight)
    meta["local_stop_weight"] = float(reward_config.local_stop_weight)
    meta["local_density_weight"] = float(reward_config.local_density_weight)
    meta["local_idle_weight"] = float(reward_config.local_idle_weight)
    meta["local_capacity_weight"] = float(reward_config.local_capacity_weight)
    meta["rail_tat_weight"] = float(reward_config.rail_tat_weight)
    meta["rail_tat_clip"] = (
        None
        if reward_config.rail_tat_clip is None
        else float(reward_config.rail_tat_clip)
    )
    meta["rail_reward_mode"] = reward_config.rail_reward_mode
    meta["rail_free_flow_neutral_ratio"] = float(
        reward_config.rail_free_flow_neutral_ratio
    )
    meta["tat_reference"] = float(reward_config.tat_reference)
    # Reward P has no TAT window. Keep the 300-second tracker explicitly
    # diagnostic-only instead of presenting it as the reward signal window.
    meta["tat_window_seconds"] = float(contract.tat_window_seconds)
    meta["recent_tat_diagnostic_window_seconds"] = float(
        reward_config.tat_window_seconds
    )
    meta["tat_weight"] = float(reward_config.tat_weight)
    meta["op_reference"] = float(reward_config.op_reference)
    meta["op_weight"] = float(reward_config.op_weight)
    meta["use_op"] = bool(reward_config.use_op)
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
    meta["tat_early_termination"] = bool(config.tat_termination_enabled)
    meta["tat_termination_policy"] = str(config.tat_termination_policy)
    meta["tat_termination_configured_start_episode"] = int(
        config.tat_termination_start_episode
    )
    meta["tat_termination_start_episode"] = int(
        config.effective_tat_termination_start_episode
    )
    meta["early_stop_tat_threshold"] = float(
        config.early_stop_tat_threshold
    )
    meta["tat_termination_grace_steps"] = int(
        config.tat_termination_grace_steps
    )
    meta["tat_above_threshold_patience"] = int(
        config.tat_above_threshold_patience
    )
    meta["tat_termination_inclusive"] = bool(
        config.tat_termination_inclusive
    )
    meta["terminal_tat_penalty"] = float(config.terminal_tat_penalty)
    meta["curriculum_end_step"] = int(config.curriculum_end_step)
    meta["curriculum_scale_start"] = float(
        config.curriculum_scale_start
    )
    meta["curriculum_scale_end"] = float(config.curriculum_scale_end)
    meta["curriculum_shape"] = str(config.curriculum_shape)
    meta["curriculum_enabled"] = bool(
        action_mode == REGION_B_RL
        and config.curriculum_scale_start != config.curriculum_scale_end
    )
    meta["configured_warmup_steps"] = int(config.warmup_steps)
    meta["warmup_steps"] = int(config.effective_warmup_steps)
    meta["effective_warmup_steps"] = int(config.effective_warmup_steps)
    meta["terminate_on_warmup_complete"] = bool(
        config.terminate_on_warmup_complete
    )
    meta["state_normalizer_load_requested"] = bool(
        config.load_state_normalizer_path
    )
    meta["state_normalizer_save_requested"] = bool(
        config.save_state_normalizer_path
    )
    meta["state_normalizer_warmup_bypass"] = bool(
        config.state_normalizer_warmup_bypass
    )
    meta["batch_size"] = int(config.batch_size)
    meta["resume_inference_until_replay_full"] = bool(
        config.resume_inference_until_replay_full
    )
    meta["resume_refill_target_env_steps"] = int(
        config.resume_refill_target_env_steps
    )
    meta["resume_deterministic_first_episode"] = bool(
        config.resume_deterministic_first_episode
    )
    meta["resume_warmstart_steps"] = int(config.resume_warmstart_steps)
    meta["stage2_policy_warm_start_on_fresh_run"] = bool(
        config.stage == STAGE_TWO
        and config.load_stage1_policy_path
        and not config.resume_checkpoint_path
    )
    meta["stage2_policy_initialization"] = (
        "stage1_encoder_actor_sale_fixed_once_targets_synchronized"
        if meta["stage2_policy_warm_start_on_fresh_run"]
        else "stage2_checkpoint_resume"
        if config.stage == STAGE_TWO and config.resume_checkpoint_path
        else "native_initialization"
    )
    meta["seed"] = int(config.seed)
    meta["replay_capacity_env_steps"] = int(
        config.replay_capacity_env_steps
    )
    meta["replay_storage"] = (
        "packed_v5_local_u8_u16_critic_tat_f32_action_q15_lap_fp16"
        if config.lap_enabled
        else "packed_v5_local_u8_u16_critic_tat_f32_action_q15_no_lap"
    )
    if action_mode == FREE_FLOW_RESIDUAL:
        meta["cost_structure"] = "residual"
        meta["rail_cost_formula"] = (
            "t_ff+0.5*d_w*c+rl_cost_lambda*t_ff*action"
        )
        meta["congestion_count_offset"] = 0.0
        meta["action_range"] = (
            f"{1.0 - float(config.rl_cost_lambda):g}-"
            f"{1.0 + float(config.rl_cost_lambda):g}"
        )
    elif action_mode == REGION_B_RL:
        meta["cost_structure"] = "b_rl"
        meta["rail_cost_formula"] = "t_ff+d_w*(c+1)*b_rl"
        meta["congestion_count_offset"] = 1.0
        meta["action_range"] = (
            "b_rl_0.0-1.0_"
            f"curriculum_{float(config.curriculum_scale_start):g}-"
            f"{float(config.curriculum_scale_end):g}"
        )
    else:
        meta["cost_structure"] = "baseline_exp_residual"
        meta["rail_cost_formula"] = (
            "(t_ff+0.5*d_w*c)*exp(applied_action)"
        )
        meta["congestion_count_offset"] = 0.0
        meta["action_range"] = f"{float(config.action_scale):g}-scaled"
    meta["exploration_noise_std"] = float(config.exploration_noise_std)
    meta["exploration_noise_final_std"] = float(
        min(config.exploration_noise_std, config.exploration_noise_final_std)
    )
    meta["exploration_noise_anneal_steps"] = int(
        config.exploration_noise_anneal_steps
    )
    meta["exploration_schedule"] = (
        "constant"
        if meta["exploration_noise_std"]
        == meta["exploration_noise_final_std"]
        else "linear_anneal"
    )
    meta["stage"] = config.stage
    meta["stage1_policy_prefix_steps"] = (
        STAGE_TWO_STAGE1_POLICY_STEPS
        if config.stage == STAGE_TWO else 0
    )
    meta["load_stage1_policy_path"] = config.load_stage1_policy_path
    meta["exploration_noise_anneal_start_step"] = int(
        config.effective_warmup_steps
    )
    meta["exploration_noise_anneal_end_step"] = int(
        config.effective_warmup_steps
        + config.exploration_noise_anneal_steps
    )
    replay_mode = "lap" if lap else "uniform"
    meta["replay"] = (
        f"{config.replay_sampling_mode}_{replay_mode}_"
        f"evict_{config.replay_eviction_mode}_"
        f"{int(config.replay_capacity_env_steps)}"
    )
    meta["description"] = (
        f"{EXP_META['description']} Directional contextual TD7 with "
        f"SALE={'on' if sale else 'off'}, LAP={'on' if lap else 'off'}, "
        f"replay_sampling={config.replay_sampling_mode}, "
        f"replay_eviction={config.replay_eviction_mode}, "
        f"stack={int(config.num_stacks)}x{int(config.stack_interval)}, "
        f"action_mode={action_mode}, dispatch_mode={config.dispatch_mode}, "
        f"rail_cost_formula={meta['rail_cost_formula']}, "
        f"critic_loss={config.critic_loss_mode}, "
        f"neighbor_aggregation={meta['neighbor_aggregation']}, "
        f"critic_target={config.critic_target_mode} over "
        f"{int(config.num_critics)} independently initialized heads"
        + (
            f" (beta={float(config.uboc_beta):.7f})"
            if config.critic_target_mode == CRITIC_TARGET_UBOC else ""
        )
        + f", policy_reads={actor_q_aggregation}, "
        f"reward {contract.version} (global/local="
        f"{reward_config.global_alpha:g}/{reward_config.local_alpha:g}), "
        f"tat_signal={contract.tat_signal_description}, "
        f"local_fixed_scale={reward_config.local_reward_scale:g}, "
        f"replay_capacity={int(config.replay_capacity_env_steps)}, "
        f"effective_action_scale={effective_scale_start:g}->"
        f"{effective_scale_end:g}, "
        f"rl_cost_lambda={float(config.rl_cost_lambda):g}, "
        "tat_termination="
        f"{config.tat_termination_policy} from episode "
        f"{int(config.effective_tat_termination_start_episode)} at "
        f"{float(config.early_stop_tat_threshold):g}, "
        f"exploration={float(config.exploration_noise_std):g}->"
        f"{meta['exploration_noise_final_std']:g} over policy schedule steps "
        f"{int(config.effective_warmup_steps)}.."
        f"{int(config.effective_warmup_steps + config.exploration_noise_anneal_steps)}, "
        "state_normalizer="
        f"{'reused' if config.state_normalizer_warmup_bypass else 'collected'}, "
        "warmup_episode_boundary="
        f"{bool(config.terminate_on_warmup_complete)}, "
        f"stage={config.stage}, num_sim={int(config.num_sim)}, "
        "stage1_policy_prefix_steps="
        f"{STAGE_TWO_STAGE1_POLICY_STEPS if config.stage == STAGE_TWO else 0}, "
        "resume_refill="
        f"{'full_capacity' if config.resume_inference_until_replay_full else 'minimum'}_"
        f"{int(config.resume_refill_target_env_steps)}, "
        "resume_deterministic_first_episode="
        f"{bool(config.resume_deterministic_first_episode)}, "
        f"resume_warmstart_steps={int(config.resume_warmstart_steps)}, "
        "control-space smoothing, "
        "single-field simulator end-time handshake, and fail-closed SimTime "
        "progress validation."
    )
    return meta


class ContextualWandbLogger:
    def __init__(self, config):
        self.enabled = bool(config.wandb_enabled)
        self.metric_keys = wandb_metric_keys_for_mode(config.mode)
        self.run = None
        self._finished = False
        if not self.enabled:
            return
        import wandb

        meta = runtime_exp_meta(config)
        expected_action_scale = (
            1.0
            if config.action_mode == FREE_FLOW_RESIDUAL
            else config.curriculum_scale_start
            if config.action_mode == REGION_B_RL
            else config.action_scale
        )
        assert meta["action_scale"] == float(expected_action_scale)
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
            for key in self.metric_keys
            if key in diagnostics
        }
        self.run.log(payload, step=int(step))

    def log_episode_final(self, *, final_tat, step):
        """Log one sparse final TAT value for a complete eval episode."""
        if self.run is None or final_tat is None:
            return
        self.run.log({"eval/final_tat": float(final_tat)}, step=int(step))

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
