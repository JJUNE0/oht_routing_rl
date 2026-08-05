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
    "reward_version": "I",
    "reward_contract_version": REWARD_VERSION,
    "reward_global_alpha": 0.5,
    "reward_local_alpha": 0.5,
    "tat_reference": 174.4236,
    "tat_weight": 9.2,
    "op_reference": 0.80,
    "op_weight": 5.0,
    "backlog_weight": 0.002,
    "local_reward_scale": 3.0,
    "rail_tat_weight": 100.0,
    "rail_tat_clip": 0.5,
    "tat_raw_clip": 1.0,
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
    "diagnostic_schema_version": "contextual_reward_diagnostic_v12",
    "centering": False,
    "replay_sampling_version": REPLAY_SAMPLING_VERSION,
    "note": "fixedscale_localdiv3_idleoff_rail100clip05_smooth025",
    "description": (
        "Reward I removes running global/local reward normalization, directly "
        "uses fixed-scale raw global reward, divides local reward by 3, removes "
        "the local idle-OHT term, scales the existing OHTTat-based rail-TAT "
        "penalty by 100 with per-rail clipping to [-0.5, 0.5], and increases "
        "REGION_B_RL temporal smoothing weight to 0.25. DistancePerVelocity is "
        "collected for diagnostics only and does not affect reward in this version."
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
) + tuple(
    f"reward/local/{term}_raw_{stat}"
    for term in ("oht", "predicted", "stop", "idle", "capacity")
    for stat in ("mean", "std", "abs_mean")
)

WANDB_METRIC_KEYS += (
    "reward/rail_tat/cycle_count",
    "reward/rail_tat/reward_applied_cycle_count",
    "reward/rail_tat/skipped_cycle_count",
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
    "reward/rail_tat/clip_assignment_ratio",
    "reward/rail_tat/clip_removed_ratio",
)


def runtime_exp_meta(config) -> dict:
    meta = dict(EXP_META)
    reward_config = ContextualRewardConfig(
        action_mode=config.action_mode,
        smooth_b_rl_weight=config.smooth_b_rl_weight,
        smooth_exp_residual_weight=config.smooth_exp_residual_weight,
        tat_confidence_n0=config.tat_confidence_n0,
        tat_confidence_ramp=config.tat_confidence_ramp,
        local_reward_scale=config.local_reward_scale,
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
    meta["rail_tat_weight"] = float(reward_config.rail_tat_weight)
    meta["rail_tat_clip"] = float(reward_config.rail_tat_clip)
    meta["tat_raw_clip"] = float(reward_config.tat_raw_clip)
    meta["tat_confidence_n0"] = float(reward_config.tat_confidence_n0)
    meta["tat_confidence_ramp"] = bool(reward_config.tat_confidence_ramp)
    meta["tat_reference"] = float(reward_config.tat_reference)
    meta["tat_weight"] = float(reward_config.tat_weight)
    meta["op_reference"] = float(reward_config.op_reference)
    meta["op_weight"] = float(reward_config.op_weight)
    meta["backlog_weight"] = float(reward_config.backlog_weight)
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
        f"reward I ({REWARD_VERSION}, global/local="
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
