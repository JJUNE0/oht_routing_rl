"""Resolution and validation helpers for contextual runtime configuration."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from oht_dispatching.config import DISPATCH_MODES
from oht_routing.algorithms.rl.contextual_td7 import (
    ACTOR_Q_AGGREGATIONS,
    CRITIC_TARGET_CDQ,
    CRITIC_TARGET_MODES,
    REPLAY_EVICTION_MODES,
    REPLAY_EVICTION_RANDOM,
    REPLAY_SAMPLING_MODES,
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
)
from oht_routing.mdp.action import ACTION_MODES
from oht_routing.mdp.reward.builder import ContextualRewardConfig
from oht_routing.mdp.reward.config import (
    canonical_reward_version,
    reward_contract,
)
from oht_routing.mdp.termination import (
    TAT_TERMINATION_REWARD_PROFILE,
    tat_termination_profile,
)
from oht_routing.runtime.stages import (
    STAGE_TWO,
    VALID_STAGES,
    sim_end_time_for_stage,
)
from oht_routing.utils.reward_diagnostic import parse_diagnostic_windows

if TYPE_CHECKING:
    from oht_routing.runtime.config import ContextualRuntimeConfig


def _validate_simulator_endpoints(config: ContextualRuntimeConfig) -> None:
    if isinstance(config.num_sim, bool) or not isinstance(
        config.num_sim, Integral
    ) or int(config.num_sim) <= 0:
        raise ValueError("num_sim must be a positive integer")
    object.__setattr__(config, "num_sim", int(config.num_sim))

    ports = config.sim_ports
    if ports is None:
        if config.num_sim > 1:
            raise ValueError(
                "num_sim > 1 requires explicit sim_ports"
            )
        return
    if isinstance(ports, (str, bytes)) or not isinstance(ports, Sequence):
        raise ValueError("sim_ports must be a sequence of integer ports")

    normalized = []
    for port in ports:
        if (
            isinstance(port, bool)
            or not isinstance(port, Integral)
            or int(port) < 1
            or int(port) > 65_535
        ):
            raise ValueError(
                "sim_ports values must be integers in [1, 65535]"
            )
        normalized.append(int(port))
    normalized = tuple(normalized)
    if len(normalized) != config.num_sim:
        raise ValueError(
            "sim_ports count must equal num_sim: "
            f"ports={len(normalized)}, num_sim={config.num_sim}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("sim_ports must contain unique ports")
    object.__setattr__(config, "sim_ports", normalized)


def _resolve_reward_and_termination(config: ContextualRuntimeConfig) -> None:
    canonical_version = canonical_reward_version(config.reward_version)
    object.__setattr__(config, "reward_version", canonical_version)
    incompatible = []
    contract = reward_contract(canonical_version)
    termination_profile = tat_termination_profile(
        config.tat_termination_policy, contract
    )
    termination_incompatible = []
    for name, expected in termination_profile.items():
        actual = getattr(config, name)
        if actual is None:
            object.__setattr__(config, name, expected)
        elif actual != expected:
            target = (
                incompatible
                if config.tat_termination_policy
                == TAT_TERMINATION_REWARD_PROFILE
                else termination_incompatible
            )
            target.append((name, actual, expected))
    if config.terminal_tat_penalty is None:
        object.__setattr__(
            config, "terminal_tat_penalty", contract.terminal_tat_penalty
        )
    elif config.terminal_tat_penalty != contract.terminal_tat_penalty:
        incompatible.append((
            "terminal_tat_penalty",
            config.terminal_tat_penalty,
            contract.terminal_tat_penalty,
        ))
    if termination_incompatible:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, actual, expected in termination_incompatible
        )
        raise ValueError(
            f"TAT termination policy {config.tat_termination_policy} is "
            f"locked; incompatible overrides: {details}"
        )
    if incompatible:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, actual, expected in incompatible
        )
        raise ValueError(
            f"Reward {canonical_version} is a locked full "
            f"reward profile; incompatible overrides: {details}"
        )


def _validate_runtime_modes(config: ContextualRuntimeConfig) -> None:
    if config.stage is not None:
        if isinstance(config.stage, bool) or not isinstance(
            config.stage, Integral
        ):
            raise ValueError("stage must be an integer")
        if int(config.stage) not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}")
        object.__setattr__(
            config, "sim_end_time", sim_end_time_for_stage(config.stage)
        )
    if config.mode not in {"baseline_only", "actor_inference", "training"}:
        raise ValueError(
            "mode must be baseline_only, actor_inference, or training"
        )
    if config.mode == "training" and not config.action_enabled:
        raise ValueError("training mode requires explicit action_enabled")
    if config.save_data_enabled and config.mode != "actor_inference":
        raise ValueError("save_data is only supported in actor_inference mode")
    if not isinstance(config.use_attention, bool):
        raise ValueError("use_attention must be bool")
    for name in (
        "num_stacks",
        "stack_interval",
        "console_log_interval",
        "sim_end_time",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}")
    if config.replay_sampling_mode not in REPLAY_SAMPLING_MODES:
        raise ValueError(
            f"replay_sampling_mode must be one of {REPLAY_SAMPLING_MODES}"
        )
    if config.replay_eviction_mode not in REPLAY_EVICTION_MODES:
        raise ValueError(
            f"replay_eviction_mode must be one of {REPLAY_EVICTION_MODES}"
        )
    if (
        config.replay_eviction_mode == REPLAY_EVICTION_RANDOM
        and config.num_stacks != 1
    ):
        raise ValueError(
            "random replay eviction currently requires num_stacks=1"
        )
    if config.dispatch_mode not in DISPATCH_MODES:
        raise ValueError(f"dispatch_mode must be one of {DISPATCH_MODES}")
    if config.critic_target_mode not in CRITIC_TARGET_MODES:
        raise ValueError(
            f"critic_target_mode must be one of {CRITIC_TARGET_MODES}"
        )
    if not isinstance(config.num_critics, Integral) or config.num_critics < 2:
        raise ValueError("num_critics must be an integer of at least 2")
    if (
        config.critic_target_mode == CRITIC_TARGET_CDQ
        and int(config.num_critics) != 2
    ):
        raise ValueError(
            "critic_target_mode 'cdq' is the two-critic minimum; pass "
            "--num-critics 2 or use --critic-target-mode uboc"
        )
    if not np.isfinite(config.uboc_beta) or not 0.0 <= config.uboc_beta < 10.0:
        raise ValueError("uboc_beta must be finite and in [0, 10)")
    if config.actor_q_aggregation not in ACTOR_Q_AGGREGATIONS:
        raise ValueError(
            f"actor_q_aggregation must be one of {ACTOR_Q_AGGREGATIONS}"
        )
    if (
        config.replay_sampling_mode
        in {REPLAY_SAMPLING_SNAPSHOT, REPLAY_SAMPLING_RANDOM_RAIL}
        and config.lap_enabled
    ):
        raise ValueError(
            f"{config.replay_sampling_mode} replay sampling requires "
            "lap_enabled=False"
        )
    if (
        not np.isfinite(config.action_scale)
        or config.action_scale < 0.0
        or config.action_scale > 1.0
    ):
        raise ValueError("action_scale must be finite and in [0, 1]")
    if (
        not np.isfinite(config.rl_cost_lambda)
        or config.rl_cost_lambda < 0.0
        or config.rl_cost_lambda > 1.0
    ):
        raise ValueError("rl_cost_lambda must be finite and in [0, 1]")
    for name in ("resume_warmstart_steps",):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if (
        config.warmup_steps < 0
        or config.resume_warmstart_steps < 0
        or config.normalizer_freeze_steps < 0
    ):
        raise ValueError("warmup/warm-start/freeze steps must be non-negative")


def _resolve_and_validate_resume(config: ContextualRuntimeConfig) -> None:
    for name in (
        "load_state_normalizer_path",
        "save_state_normalizer_path",
        "load_stage1_policy_path",
    ):
        value = getattr(config, name)
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(f"{name} must be a non-empty path string")
    if not isinstance(config.state_normalizer_warmup_bypass, bool):
        raise ValueError("state_normalizer_warmup_bypass must be bool")
    if not isinstance(config.terminate_on_warmup_complete, bool):
        raise ValueError("terminate_on_warmup_complete must be bool")
    if not isinstance(config.resume_inference_until_replay_full, bool):
        raise ValueError("resume_inference_until_replay_full must be bool")
    if not isinstance(config.resume_deterministic_first_episode, bool):
        raise ValueError("resume_deterministic_first_episode must be bool")
    if config.load_state_normalizer_path is not None:
        object.__setattr__(config, "state_normalizer_warmup_bypass", True)
    if config.load_stage1_policy_path is not None:
        object.__setattr__(config, "state_normalizer_warmup_bypass", True)
    if config.stage == STAGE_TWO:
        # actor_inference is allowed so a Stage 2 policy can be evaluated
        # under the same per-episode contract it trained on: the frozen
        # Stage 1 prefix drives ticks 1..2,000 and the Stage 2 policy takes
        # over from 2,001. Evaluating without the prefix would put the first
        # 2,000 ticks out of distribution.
        if config.mode not in {"training", "actor_inference"}:
            raise ValueError(
                "stage 2 requires training or actor_inference mode"
            )
        if not config.action_enabled:
            raise ValueError("stage 2 requires explicit action_enabled")
        if config.load_stage1_policy_path is None:
            raise ValueError("stage 2 requires --load-stage1-policy")
        if (
            config.mode == "actor_inference"
            and config.resume_checkpoint_path is None
        ):
            raise ValueError(
                "stage 2 actor_inference requires --resume-checkpoint for "
                "the Stage 2 policy"
            )
    elif config.load_stage1_policy_path is not None:
        raise ValueError("load_stage1_policy_path requires stage 2")
    if (
        config.stage1_policy_warm_start
        and config.load_stage1_policy_path is None
    ):
        raise ValueError(
            "stage1_policy_warm_start requires --load-stage1-policy"
        )
    if (
        config.load_stage1_policy_path is not None
        and config.load_state_normalizer_path is not None
    ):
        raise ValueError(
            "standalone state-normalizer load and --load-stage1-policy are "
            "mutually exclusive; the Stage 1 checkpoint already restores "
            "the frozen state normalizers"
        )
    if config.resume_checkpoint_path is not None and (
        config.load_state_normalizer_path is not None
    ):
        raise ValueError(
            "standalone state-normalizer load and --resume-checkpoint "
            "are mutually exclusive; checkpoint resume already restores "
            "the state normalizers"
        )
    if (
        config.state_normalizer_warmup_bypass
        and config.load_state_normalizer_path is None
        and config.resume_checkpoint_path is None
        and config.load_stage1_policy_path is None
    ):
        raise ValueError(
            "state_normalizer_warmup_bypass requires a standalone "
            "normalizer load or checkpoint resume"
        )
    if (
        config.resume_inference_until_replay_full
        and config.resume_checkpoint_path is None
    ):
        raise ValueError(
            "resume_inference_until_replay_full requires checkpoint resume"
        )
    if (
        config.resume_deterministic_first_episode
        and config.resume_checkpoint_path is None
    ):
        raise ValueError(
            "resume_deterministic_first_episode requires checkpoint resume"
        )
    if (
        isinstance(config.resume_deterministic_episodes, bool)
        or not isinstance(config.resume_deterministic_episodes, int)
        or config.resume_deterministic_episodes < 0
    ):
        raise ValueError(
            "resume_deterministic_episodes must be a non-negative integer"
        )
    if (
        config.resume_deterministic_episodes > 0
        and config.resume_checkpoint_path is None
    ):
        raise ValueError(
            "resume_deterministic_episodes requires checkpoint resume"
        )
    if (
        config.resume_deterministic_episodes > 0
        and config.resume_deterministic_first_episode
    ):
        raise ValueError(
            "resume_deterministic_episodes replaces "
            "--resume-deterministic-first-episode; use only one"
        )
    if (
        config.resume_deterministic_episodes > 0
        and config.resume_warmstart_steps > 0
    ):
        raise ValueError(
            "resume_deterministic_episodes and resume_warmstart_steps are "
            "mutually exclusive"
        )
    if (
        isinstance(config.resume_stochastic_episodes, bool)
        or not isinstance(config.resume_stochastic_episodes, int)
        or config.resume_stochastic_episodes < 0
    ):
        raise ValueError(
            "resume_stochastic_episodes must be a non-negative integer"
        )
    if (
        config.resume_stochastic_episodes > 0
        and config.resume_checkpoint_path is None
    ):
        raise ValueError(
            "resume_stochastic_episodes requires checkpoint resume"
        )
    for other in (
        "resume_deterministic_episodes",
        "resume_deterministic_first_episode",
        "resume_warmstart_steps",
        "resume_inference_until_replay_full",
    ):
        if config.resume_stochastic_episodes > 0 and getattr(config, other):
            raise ValueError(
                f"resume_stochastic_episodes and {other} are mutually "
                "exclusive"
            )
    if (
        config.resume_warmstart_steps > 0
        and config.resume_checkpoint_path is None
    ):
        raise ValueError("resume_warmstart_steps requires checkpoint resume")
    if config.resume_warmstart_steps > 0 and config.mode != "training":
        raise ValueError("resume_warmstart_steps requires training mode")
    if (
        config.resume_warmstart_steps > 0
        and config.resume_deterministic_first_episode
    ):
        raise ValueError(
            "resume_warmstart_steps and "
            "resume_deterministic_first_episode are mutually exclusive"
        )
    if config.resume_warmstart_steps > 0 and config.stage == STAGE_TWO:
        raise ValueError(
            "resume_warmstart_steps and stage 2 are mutually exclusive"
        )


def _validate_distributed_runtime(config: ContextualRuntimeConfig) -> None:
    """Keep the first multi-simulator contract narrow and restart-safe."""

    if config.num_sim == 1:
        return
    if config.mode != "training" or config.stage != STAGE_TWO:
        raise ValueError(
            "num_sim > 1 currently requires Stage 2 training"
        )
    if config.load_stage1_policy_path is None:
        raise ValueError(
            "distributed Stage 2 requires --load-stage1-policy"
        )
    if config.resume_checkpoint_path is not None:
        raise ValueError(
            "distributed full-state checkpoint resume is not supported; "
            "start a fresh distributed Stage 2 learner from "
            "--load-stage1-policy"
        )
    if config.save_state_normalizer_path is not None:
        raise ValueError(
            "distributed Stage 2 uses the frozen normalizers embedded in "
            "the Stage 1 checkpoint"
        )


def _validate_training_controls(config: ContextualRuntimeConfig) -> None:
    if config.mode == "training" and config.action_scale <= 0:
        raise ValueError("training mode requires positive action_scale")
    if (
        config.curriculum_end_step <= config.effective_warmup_steps
        or not 0 < config.curriculum_scale_start <= 1
        or not 0 < config.curriculum_scale_end <= 1
        or config.curriculum_scale_start > config.curriculum_scale_end
        or config.curriculum_shape not in {"geometric", "linear"}
    ):
        raise ValueError("invalid action curriculum configuration")
    parse_diagnostic_windows(config.reward_diagnostic_windows)
    if config.minimum_sign_sample_count <= 0:
        raise ValueError("minimum_sign_sample_count must be positive")
    if (
        not np.isfinite(config.exploration_noise_std)
        or config.exploration_noise_std < 0
        or not np.isfinite(config.exploration_noise_final_std)
        or config.exploration_noise_final_std < 0
    ):
        raise ValueError(
            "exploration noise stds must be finite and non-negative"
        )
    positive = (
        config.replay_capacity_env_steps,
        config.batch_size,
        config.minimum_replay_env_steps,
        config.minimum_action_enabled_env_steps,
        config.updates_per_env_step,
        config.learn_every_env_steps,
        config.exploration_noise_anneal_steps,
        config.latest_checkpoint_interval,
        config.periodic_checkpoint_interval,
        config.wandb_log_interval,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("training counts/intervals must be positive")
    if (
        config.early_stop_queued_threshold <= 0
        or not np.isfinite(config.early_stop_tat_threshold)
        or config.early_stop_tat_threshold <= 0
        or config.max_stale_sim_time_ticks <= 0
    ):
        raise ValueError("early-stop/watchdog settings are invalid")
    if (
        isinstance(config.tat_termination_grace_steps, bool)
        or not isinstance(config.tat_termination_grace_steps, Integral)
        or config.tat_termination_grace_steps < 0
        or isinstance(config.tat_termination_start_episode, bool)
        or not isinstance(config.tat_termination_start_episode, Integral)
        or config.tat_termination_start_episode <= 0
        or isinstance(config.tat_above_threshold_patience, bool)
        or not isinstance(config.tat_above_threshold_patience, Integral)
        or config.tat_above_threshold_patience <= 0
    ):
        raise ValueError("TAT termination counts are invalid")
    if (
        not np.isfinite(config.terminal_tat_penalty)
        or config.terminal_tat_penalty > 0.0
    ):
        raise ValueError(
            "terminal_tat_penalty must be finite and non-positive"
        )
    if config.rail_tat_diagnostic_max_step < 0:
        raise ValueError(
            "rail_tat_diagnostic_max_step must be non-negative"
        )


def validate_runtime_config(config: ContextualRuntimeConfig) -> None:
    """Resolve defaults and reject incompatible runtime settings."""
    _resolve_reward_and_termination(config)
    _validate_simulator_endpoints(config)
    _validate_runtime_modes(config)
    _resolve_and_validate_resume(config)
    _validate_distributed_runtime(config)
    _validate_training_controls(config)
    make_reward_config(config)


def make_reward_config(
    config: ContextualRuntimeConfig,
) -> ContextualRewardConfig:
    """Build the reward configuration selected by the runtime settings."""
    return ContextualRewardConfig.for_version(
        config.reward_version,
        action_mode=config.action_mode,
    )
