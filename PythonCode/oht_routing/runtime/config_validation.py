"""Resolution and validation helpers for contextual runtime configuration."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from oht_dispatching.config import DISPATCH_MODES
from oht_routing.algorithms.rl.contextual_td7 import (
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
from oht_routing.utils.reward_diagnostic import parse_diagnostic_windows

if TYPE_CHECKING:
    from oht_routing.runtime.config import ContextualRuntimeConfig


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
    if config.mode not in {"baseline_only", "actor_inference", "training"}:
        raise ValueError(
            "mode must be baseline_only, actor_inference, or training"
        )
    if config.mode == "training" and not config.action_enabled:
        raise ValueError("training mode requires explicit action_enabled")
    for name in ("num_stacks", "stack_interval"):
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
    if config.dispatch_mode not in DISPATCH_MODES:
        raise ValueError(f"dispatch_mode must be one of {DISPATCH_MODES}")
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
        isinstance(config.episode_burnin_steps, bool)
        or not isinstance(config.episode_burnin_steps, Integral)
    ):
        raise ValueError("episode_burnin_steps must be an integer")
    if (
        config.warmup_steps < 0
        or config.episode_burnin_steps < 0
        or config.normalizer_freeze_steps < 0
    ):
        raise ValueError("warmup/burn-in/freeze steps must be non-negative")


def _resolve_and_validate_resume(config: ContextualRuntimeConfig) -> None:
    for name in (
        "load_state_normalizer_path",
        "save_state_normalizer_path",
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
    _validate_runtime_modes(config)
    _resolve_and_validate_resume(config)
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
