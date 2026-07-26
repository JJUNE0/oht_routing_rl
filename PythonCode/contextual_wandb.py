"""Training-only W&B schema and lightweight logger."""

from __future__ import annotations

from datetime import datetime

from cocel_rl.algorithms.contextual_td7 import (
    ALGORITHM_VERSION,
    CHECKPOINT_VERSION,
    CRITIC_INITIALIZATION,
    LAP_VERSION,
    SALE_VERSION,
    contextual_algorithm_variant,
)
from contextual_action import (
    EXPLORATION_SCHEDULE_VERSION,
    REGION_B_RL,
    action_version,
)
from contextual_reward import ContextualRewardConfig, REWARD_VERSION

EXP_META = {
    "cost_structure": "b_rl",
    "action_range": "b_rl_0.0-1.0",
    "topology": "directed_10in_10out_controlled_centers_v2",
    "observation": "contextual_obs_controlled_v1",
    "reward_version": "D",
    "reward_contract_version": REWARD_VERSION,
    "reward_global_alpha": 0.5,
    "reward_local_alpha": 0.5,
    "action_scale": 0.05,
    "exploration_noise_std": 0.10,
    "exploration_noise_final_std": 0.02,
    "exploration_noise_anneal_steps": 100_000,
    "exploration_schedule_version": EXPLORATION_SCHEDULE_VERSION,
    "resume_refill_version": "global_step_v2",
    "checkpoint_rng_version": "exploration_rng_v2",
    "send_cost_logging_version": "post_send_v2",
    "protocol_version": "single_end_time_v2",
    "diagnostic_schema_version": "contextual_action_temporal_diag_v4",
    "centering": False,
    "note": "noiseanneal",
    "description": (
        "Separates deterministic-policy and exploratory temporal deltas while "
        "linearly annealing exploration noise from 0.10 to 0.02 over 100,000 "
        "post-warmup environment steps."
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
    "env/termination_reason",
    "termination/done", "termination/by_queue", "termination/by_tat",
    "protocol/stale_sim_time_ticks",
    "action/policy_mean", "action/policy_std", "action/policy_min",
    "action/policy_max", "action/policy_saturation_ratio",
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
    "reward/global_raw", "reward/global_normalized",
    "reward/global_component", "reward/local_raw_mean",
    "reward/local_raw_std", "reward/local_normalized_mean",
    "reward/local_normalized_std", "reward/local_component_mean",
    "reward/local_component_std", "reward/marginal_tat_ema",
    "reward/tat_signal_available",
    "reward/rail_tat_penalty_mean", "reward/smooth_penalty_mean",
    "reward/total_mean", "reward/total_std", "reward/finite_ratio",
    "reward/smooth_control_delta_mean", "reward/smooth_control_delta_max",
    "reward/step_reward", "reward/global_norm", "reward/local_norm",
    "reward/rail_tat", "reward/rail_tat_event_count",
    "reward/rail_tat_sum", "reward/rail_tat_mean", "reward/rail_tat_max",
    "reward/alpha", "reward/marg_tat", "reward/backlog",
    "reward/smooth_mean",
    "curriculum/action_scale",
    "global/tat", "global/op_rate", "global/queued",
    "global/queued_jobs", "global/waiting", "global/transfer",
    "global/completed",
    "job/mean_wait_priority", "job/mean_reassign", "job/queued",
    "oht/idle_count", "oht/move_to_load", "oht/move_to_unload",
    "oht/loading", "oht/unloading",
    "replay/size_env_steps", "replay/size_logical_transitions",
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


def runtime_exp_meta(config) -> dict:
    meta = dict(EXP_META)
    reward_config = ContextualRewardConfig()
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
    meta["sale"] = sale
    meta["lap"] = lap
    meta["sale_version"] = SALE_VERSION
    meta["lap_version"] = LAP_VERSION
    meta["critic_loss_mode"] = config.critic_loss_mode
    meta["action_scale"] = float(config.action_scale)
    meta["reward_global_alpha"] = float(reward_config.global_alpha)
    meta["reward_local_alpha"] = float(reward_config.local_alpha)
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
    meta["exploration_noise_anneal_start_step"] = int(config.warmup_steps)
    meta["exploration_noise_anneal_end_step"] = int(
        config.warmup_steps + config.exploration_noise_anneal_steps
    )
    meta["exploration_schedule_version"] = EXPLORATION_SCHEDULE_VERSION
    replay_mode = "lap" if lap else "uniform"
    meta["replay"] = (
        f"snapshot_{replay_mode}_{int(config.replay_capacity_env_steps)}"
    )
    meta["note"] = f"replay{int(config.replay_capacity_env_steps)}"
    meta["description"] = (
        "Directional contextual TD7 with "
        f"SALE={'on' if sale else 'off'}, LAP={'on' if lap else 'off'}, "
        f"action_mode={action_mode}, critic_loss={config.critic_loss_mode}, "
        "independently initialized Q1/Q2 heads, "
        f"reward D ({REWARD_VERSION}, global/local="
        f"{reward_config.global_alpha:g}/{reward_config.local_alpha:g}), "
        f"replay_capacity={int(config.replay_capacity_env_steps)}, "
        f"curriculum_end_step={int(config.curriculum_end_step)}, "
        f"exploration={float(config.exploration_noise_std):g}->"
        f"{meta['exploration_noise_final_std']:g} over global steps "
        f"{int(config.warmup_steps)}.."
        f"{int(config.warmup_steps + config.exploration_noise_anneal_steps)}, "
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
