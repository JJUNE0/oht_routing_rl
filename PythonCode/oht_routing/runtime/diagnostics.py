"""Bounded reward diagnostics and runtime logging for contextual TD7."""

from __future__ import annotations

import numpy as np


class ContextualRuntimeDiagnosticsMixin:
    def _update_phase2_reward_diagnostics(self, batch) -> None:
        """Maintain bounded, run-local diagnostics for provisional scaling."""
        global_scale = abs(float(batch.global_component))
        local_scale = float(np.mean(np.abs(batch.local_component)))
        rail_active = np.abs(batch.rail_reward_postclip[
            ~np.isclose(batch.rail_reward_postclip, 0.0)
        ])
        self._phase2_global_scales.append(global_scale)
        self._phase2_local_scales.append(local_scale)
        self._phase2_rail_active_scales.extend(float(x) for x in rail_active)
        self._phase2_backlogs.append(float(batch.backlog))

        backlog_p50 = float(np.median(self._phase2_backlogs))
        total_mean = float(batch.total.mean())
        if batch.total_tat_level < self.reward_builder.config.tat_reference and (
            batch.backlog <= backlog_p50
        ):
            self._phase2_good_rewards.append(total_mean)
        if batch.total_tat_level > self.reward_builder.config.tat_reference and (
            batch.backlog >= backlog_p50
        ):
            self._phase2_bad_rewards.append(total_mean)

        def median(values):
            return float(np.median(values)) if values else 0.0

        sg = median(self._phase2_global_scales)
        sl = median(self._phase2_local_scales)
        sr = median(self._phase2_rail_active_scales)
        scale_array = np.asarray([sg, sl, sr], dtype=np.float64)
        positive = scale_array[scale_array > 0.0]
        scale_mean = float(positive.mean()) if positive.size else 0.0
        balance_error = (
            float(np.mean(np.abs(positive - scale_mean)) / scale_mean)
            if scale_mean > 0.0 else 0.0
        )
        good_count = len(self._phase2_good_rewards)
        bad_count = len(self._phase2_bad_rewards)
        good_mean = (
            float(np.mean(self._phase2_good_rewards)) if good_count else 0.0
        )
        bad_mean = (
            float(np.mean(self._phase2_bad_rewards)) if bad_count else 0.0
        )
        smooth_scale = float(np.mean(np.abs(batch.smooth_penalty)))
        main_scale_mean = float(scale_array.mean())
        self.last_diagnostics.update({
            "reward/scale/global_representative": sg,
            "reward/scale/local_representative": sl,
            "reward/scale/rail_active_representative": sr,
            "reward/scale/global_to_local": sg / sl if sl > 0.0 else 0.0,
            "reward/scale/global_to_rail": sg / sr if sr > 0.0 else 0.0,
            "reward/scale/local_to_rail": sl / sr if sr > 0.0 else 0.0,
            "reward/scale/main_balance_error": balance_error,
            "reward/scale/smooth_excess_warning": float(
                main_scale_mean > 0.0
                and smooth_scale > 0.1 * main_scale_mean
            ),
            "reward/sign/good_state_sample_count": float(good_count),
            "reward/sign/good_state_total_mean": good_mean,
            "reward/sign/bad_state_sample_count": float(bad_count),
            "reward/sign/bad_state_total_mean": bad_mean,
            "reward/sign/good_minus_bad": good_mean - bad_mean,
            "reward/sign/good_state_sufficient": float(
                good_count >= self.config.minimum_sign_sample_count
            ),
            "reward/sign/bad_state_sufficient": float(
                bad_count >= self.config.minimum_sign_sample_count
            ),
        })
        q1 = self.last_diagnostics.get("critic/q1_mean")
        q2 = self.last_diagnostics.get("critic/q2_mean")
        if q1 is not None and q2 is not None:
            q_mean = 0.5 * (float(q1) + float(q2))
            if np.isfinite(q_mean):
                self._phase2_q_means.append(q_mean)
        for lag in (100, 1_000):
            self.last_diagnostics[f"critic/q_mean_delta_{lag}"] = (
                self._phase2_q_means[-1] - self._phase2_q_means[-lag - 1]
                if len(self._phase2_q_means) > lag else 0.0
            )

    def _write_reward_step_diagnostic(self, pclient, completed) -> None:
        writer = self.reward_diagnostic_writer
        global_step = self.total_steps - 1
        if (
            writer is None
            or not writer.writes_json
            or writer.window_name(global_step) is None
        ):
            return
        batch = completed.reward

        def stats(record, prefix, values, *, abs_mean=False):
            array = np.asarray(values, dtype=np.float64)
            record[prefix + "_mean"] = float(array.mean())
            record[prefix + "_std"] = float(array.std())
            record[prefix + "_min"] = float(array.min())
            record[prefix + "_max"] = float(array.max())
            if abs_mean:
                record[prefix + "_abs_mean"] = float(np.abs(array).mean())

        rail_raw = batch.rail_reward_raw
        rail_weighted = batch.rail_reward_weighted_preclip
        rail_postclip = batch.rail_reward_postclip
        clip_mask = ~np.isclose(rail_weighted, rail_postclip)
        nonzero = ~np.isclose(rail_postclip, 0.0)
        top_count = min(10, len(rail_postclip))
        top_rows = np.argsort(np.abs(rail_postclip))[-top_count:][::-1]
        policy = np.asarray(completed.action, dtype=np.float64).reshape(-1)
        applied = np.asarray(
            completed.applied_action, dtype=np.float64
        ).reshape(-1)
        b_rl = 0.5 + 0.5 * applied
        global_vector = np.full_like(
            batch.local_component, batch.global_component
        )
        components = (
            global_vector,
            batch.local_component,
            batch.rail_reward_postclip,
            -batch.smooth_penalty,
        )
        abs_means = np.asarray([
            np.mean(np.abs(value)) for value in components
        ], dtype=np.float64)
        shares = abs_means / (abs_means.sum() + np.finfo(np.float64).eps)
        reconstructed = sum(components)
        reward_error = np.abs(batch.total - reconstructed)
        diagnostics = self.last_diagnostics
        learner_available = bool(
            diagnostics.get("update/learner_count", 0.0) > 0
            and "critic/q1_mean" in diagnostics
        )

        record = {
            "reward_version": self.reward_builder.reward_version,
            "global_step": int(global_step),
            "episode_id": int(completed.episode_id),
            "episode_step": int(completed.env_step),
            "sim_time": float(getattr(pclient, "SimTime", 0.0)),
            "total_tat": batch.total_tat_level,
            "tat_reference": self.reward_builder.config.tat_reference,
            "tat_weight": self.reward_builder.config.tat_weight,
            "tat_signal_available": bool(batch.tat_signal_available),
            "tat_raw": batch.tat_raw,
            "op_rate": batch.op_rate,
            "op_reference": batch.op_reference,
            "op_error": batch.op_error,
            "op_raw": batch.op_raw,
            "waiting": batch.waiting,
            "queued": batch.queued,
            "backlog": batch.backlog,
            "backlog_weight": self.reward_builder.config.backlog_weight,
            "backlog_raw": batch.backlog_raw,
            "backlog_growth_signal": batch.backlog_growth_signal,
            "backlog_growth_raw": batch.backlog_growth_raw,
            "idle_oht_count": batch.idle_oht_count,
            "idle_reserve_signal": batch.idle_reserve_signal,
            "idle_reserve_raw": batch.idle_reserve_raw,
            "global_raw": batch.global_raw,
            "global_alpha": self.reward_builder.config.global_alpha,
            "global_component": batch.global_component,
            "global_decomposition_error": abs(
                batch.global_raw - batch.tat_raw
                - batch.op_raw
                - batch.backlog_raw
                - batch.backlog_growth_raw
                - batch.idle_reserve_raw
            ),
            "tat_raw_abs": diagnostics.get(
                "reward/budget/tat_raw_abs", 0.0
            ),
            "backlog_raw_abs": diagnostics.get(
                "reward/budget/backlog_raw_abs", 0.0
            ),
            "tat_contribution_abs": diagnostics.get(
                "reward/contribution/tat_abs", 0.0
            ),
            "backlog_contribution_abs": diagnostics.get(
                "reward/contribution/backlog_abs", 0.0
            ),
            "global_component_abs": diagnostics.get(
                "reward/budget/global_abs", 0.0
            ),
            "local_component_abs_mean": diagnostics.get(
                "reward/budget/local_abs", 0.0
            ),
            "rail_component_abs_mean": diagnostics.get(
                "reward/budget/rail_abs", 0.0
            ),
            "smooth_component_abs_mean": diagnostics.get(
                "reward/budget/smooth_abs", 0.0
            ),
            "terminal_penalty": batch.terminal_penalty,
            "reward_budget_shares": {
                name: diagnostics.get(f"reward/budget/{name}_share", 0.0)
                for name in ("tat", "backlog", "local", "rail", "smooth")
            },
            "leading_indicator_snapshot": {
                key: value
                for key, value in diagnostics.items()
                if key.startswith("lead/") or key.startswith("leadlag/")
            },
            "idle_oht_observation_mean": float(
                batch.idle_oht_observation.mean()
            ),
            "idle_oht_observation_max": float(
                batch.idle_oht_observation.max()
            ),
            "local_idle_reward_mean": float(batch.local_idle_raw.mean()),
            "local_idle_reward_max": float(batch.local_idle_raw.max()),
            "local_reward_scale": self.reward_builder.config.local_reward_scale,
            "local_predicted_oht_weight": (
                self.reward_builder.config.local_predicted_oht_weight
            ),
            "local_alpha": self.reward_builder.config.local_alpha,
            "rail_tat_cycle_count": int(
                self.reward_builder._last_rail_tat_cycle_count
            ),
            "rail_tat_controlled_assignment_count": int(
                self.reward_builder._last_rail_tat_controlled_assignment_count
            ),
            "rail_tat_uncontrolled_assignment_count": int(
                self.reward_builder._last_rail_tat_uncontrolled_assignment_count
            ),
            "rail_tat_weight": self.reward_builder.config.rail_tat_weight,
            "rail_tat_clip": self.reward_builder.config.rail_tat_clip,
            "rail_reward_mode": self.reward_builder.config.rail_reward_mode,
            "rail_free_flow_neutral_ratio": (
                self.reward_builder.config.rail_free_flow_neutral_ratio
            ),
            "rail_tat_clip_fraction": float(clip_mask.mean()),
            "rail_tat_clip_removed_abs_mean": float(
                np.abs(rail_weighted - rail_postclip).mean()
            ),
            "rail_tat_clip_removed_abs_sum": float(
                np.abs(rail_weighted - rail_postclip).sum()
            ),
            "rail_tat_positive_ratio": float((rail_postclip > 0).mean()),
            "rail_tat_negative_ratio": float((rail_postclip < 0).mean()),
            "rail_tat_nonzero_ratio": float(nonzero.mean()),
            "rail_tat_raw_nonzero_ratio": float(
                (~np.isclose(rail_raw, 0.0)).mean()
            ),
            "rail_tat_top_abs": [
                {
                    "rail_id": int(batch.controlled_rail_ids[row]),
                    "raw": float(rail_raw[row]),
                    "weighted_preclip": float(rail_weighted[row]),
                    "postclip": float(rail_postclip[row]),
                    "clip_applied": bool(clip_mask[row]),
                }
                for row in top_rows if not np.isclose(rail_postclip[row], 0.0)
            ],
            "smooth_weight": batch.smooth_weight_effective,
            "control_delta_nonzero_ratio": float(
                (~np.isclose(batch.smooth_control_delta, 0.0)).mean()
            ),
            "action_clipped_fraction": diagnostics.get(
                "action/clipped_fraction", 0.0
            ),
            "policy_saturation_ratio": diagnostics.get(
                "action/policy_saturation_ratio", 0.0
            ),
            "curriculum_action_scale": diagnostics.get(
                "curriculum/action_scale", 0.0
            ),
            "exploration_noise_std": diagnostics.get(
                "action/exploration_noise_std", 0.0
            ),
            "env_tat": diagnostics.get("env/tat", batch.total_tat_level),
            "operation_rate": diagnostics.get("env/operation_rate", batch.op_rate),
            "completed_delta": batch.completed_delta,
            "transfer_count": diagnostics.get("env/transferring", 0.0),
            "oht_idle": diagnostics.get("oht/idle_count", 0.0),
            "oht_move_to_load": diagnostics.get("oht/move_to_load", 0.0),
            "oht_loading": diagnostics.get("oht/loading", 0.0),
            "oht_move_to_unload": diagnostics.get("oht/move_to_unload", 0.0),
            "oht_unloading": diagnostics.get("oht/unloading", 0.0),
            "mean_reassign": diagnostics.get("job/mean_reassign", 0.0),
            "learner_available": learner_available,
            "learner_updates": diagnostics.get("learner/updates", 0.0),
            "reward_total_abs_mean": float(np.abs(batch.total).mean()),
            "reward_finite_ratio": float(np.isfinite(batch.total).mean()),
            "global_component_abs_mean": float(abs_means[0]),
            "local_component_abs_mean": float(abs_means[1]),
            "rail_tat_component_abs_mean": float(abs_means[2]),
            "smooth_component_abs_mean": float(abs_means[3]),
            "global_abs_share": float(shares[0]),
            "local_abs_share": float(shares[1]),
            "rail_tat_abs_share": float(shares[2]),
            "smooth_abs_share": float(shares[3]),
            "reward_decomposition_error_mean": float(reward_error.mean()),
            "reward_decomposition_error_max": float(reward_error.max()),
            "phase2_global_representative": diagnostics.get(
                "reward/scale/global_representative", 0.0
            ),
            "phase2_local_representative": diagnostics.get(
                "reward/scale/local_representative", 0.0
            ),
            "phase2_rail_active_representative": diagnostics.get(
                "reward/scale/rail_active_representative", 0.0
            ),
            "phase2_main_balance_error": diagnostics.get(
                "reward/scale/main_balance_error", 0.0
            ),
            "good_state_sample_count": diagnostics.get(
                "reward/sign/good_state_sample_count", 0.0
            ),
            "good_state_total_mean": diagnostics.get(
                "reward/sign/good_state_total_mean", 0.0
            ),
            "bad_state_sample_count": diagnostics.get(
                "reward/sign/bad_state_sample_count", 0.0
            ),
            "bad_state_total_mean": diagnostics.get(
                "reward/sign/bad_state_total_mean", 0.0
            ),
            "good_minus_bad": diagnostics.get(
                "reward/sign/good_minus_bad", 0.0
            ),
            "q_mean_delta_100": diagnostics.get(
                "critic/q_mean_delta_100", 0.0
            ),
            "q_mean_delta_1000": diagnostics.get(
                "critic/q_mean_delta_1000", 0.0
            ),
        }
        for prefix, values in (
            ("local_raw", batch.local_raw),
            ("local_oht_raw", batch.local_oht_raw),
            ("local_predicted_raw", batch.local_predicted_raw),
            ("local_stop_raw", batch.local_stop_raw),
            ("local_capacity_raw", batch.local_capacity_raw),
            ("local_normalized", batch.local_normalized),
            ("local_component", batch.local_component),
            ("rail_tat_raw", rail_raw),
            ("rail_tat_weighted_preclip", rail_weighted),
            ("rail_tat_postclip", rail_postclip),
            ("control_delta", batch.smooth_control_delta),
            ("smooth_penalty", batch.smooth_penalty),
            ("policy_action", policy),
            ("applied_action", applied),
            ("b_rl", b_rl),
            ("b_rl_temporal_delta", batch.smooth_control_delta),
            ("reward_total", batch.total),
        ):
            stats(record, prefix, values, abs_mean=prefix in {
                "rail_tat_raw", "rail_tat_weighted_preclip",
                "rail_tat_postclip", "smooth_penalty", "reward_total",
            })
        learner_fields = {
            "q1_mean": "critic/q1_mean",
            "q2_mean": "critic/q2_mean",
            "target_q_mean": "critic/target_q_mean",
            "q_min": "critic/q_min",
            "q_max": "critic/q_max",
            "q1_q2_abs_diff_mean": "critic/q_abs_diff_mean",
            "critic_loss": "learner/critic_loss",
            "q1_loss": "critic/q1_loss",
            "q2_loss": "critic/q2_loss",
            "td_error_mean": "critic/td_error_mean",
            "td_error_max": "critic/td_error_max",
            "critic_grad_norm": "grad/critic_norm",
            "encoder_grad_norm": "grad/encoder_norm",
            "actor_loss": "learner/actor_loss_last",
            "actor_grad_norm": "learner/actor_grad_norm_last",
        }
        for output, source in learner_fields.items():
            record[output] = diagnostics.get(source) if learner_available else None
        writer.append_step(global_step, record)

    def record_send_cost_ms(self, elapsed_ms: float):
        send_ms = float(elapsed_ms)
        self.last_diagnostics["runtime/send_cost_ms"] = send_ms
        self.last_diagnostics["runtime/total_ms"] = (
            float(self.last_diagnostics.get("runtime/total_algorithm_ms", 0.0))
            + send_ms
        )

    def log_wandb_tick(self):
        if (
            self.config.mode in {"training", "actor_inference"}
            and not self.training_failed
            and self.total_steps % self.config.wandb_log_interval == 0
        ):
            self.wandb_logger.log(self.last_diagnostics, self.total_steps)
