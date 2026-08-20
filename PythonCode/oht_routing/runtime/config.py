"""Validated runtime configuration for contextual TD7 v2."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    REPLAY_SAMPLING_MODES,
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    read_contextual_runtime_config,
)
from oht_routing.mdp.action import ACTION_MODES, REGION_B_RL
from oht_routing.routing.dispatch import (
    DISPATCH_FIRST_MATCH,
    DISPATCH_MODES,
)
from oht_routing.mdp.reward.builder import ContextualRewardConfig
from oht_routing.utils.reward_diagnostic import parse_diagnostic_windows
from oht_routing.mdp.reward.config import (
    REWARD_VERSION,
    canonical_reward_version,
    reward_contract,
)
from oht_routing.mdp.termination import (
    TAT_TERMINATION_REWARD_PROFILE,
    tat_termination_profile,
)


RESUME_RUNTIME_CONFIG_VERSION = "contextual_resume_full_runtime_config_v1"
RESUME_LAUNCH_CONTROL_FIELDS = {
    "mode",
    "action_enabled",
    "device",
    "topology_cache_path",
    "topology_audit_path",
    "load_state_normalizer_path",
    "save_state_normalizer_path",
    "checkpoint_root",
    "resume_checkpoint_path",
    "rail_tat_diagnostic_path",
    "rail_tat_diagnostic_max_step",
    "reward_diagnostic_dir",
    "reward_diagnostic_windows",
    "wandb_enabled",
    "dispatch_mode",
    "batch_size",
    "resume_inference_until_replay_full",
    "resume_deterministic_first_episode",
}
_ARG_ALIASES = {
    "load_state_normalizer_path": "load_state_normalizer",
    "save_state_normalizer_path": "save_state_normalizer",
    "wandb_enabled": "wandb",
    "resume_checkpoint_path": "resume_checkpoint",
    "rail_tat_diagnostic_path": "rail_tat_diagnostic",
}
_PATH_CONFIG_FIELDS = {
    "load_state_normalizer_path",
    "save_state_normalizer_path",
}


@dataclass(frozen=True)
class ContextualRuntimeConfig:
    mode: str = "baseline_only"
    action_enabled: bool = False
    reward_version: str = REWARD_VERSION
    action_mode: str = REGION_B_RL
    action_scale: float = 0.05
    num_stacks: int = 1
    stack_interval: int = 1
    curriculum_end_step: int = 20_000
    curriculum_scale_start: float = 0.05
    curriculum_scale_end: float = 1.0
    curriculum_shape: str = "geometric"
    warmup_steps: int = 10_000
    terminate_on_warmup_complete: bool = True
    episode_burnin_steps: int = 0
    normalizer_freeze_steps: int = 10_000
    load_state_normalizer_path: str | None = None
    save_state_normalizer_path: str | None = None
    state_normalizer_warmup_bypass: bool = False
    reward_diagnostic_dir: str | None = None
    reward_diagnostic_windows: str = "0:1000,10000:11000,20000:21000"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    topology_cache_path: str | None = None
    topology_audit_path: str | None = None
    exploration_noise_std: float = 0.10
    exploration_noise_final_std: float = 0.02
    exploration_noise_anneal_steps: int = 100_000
    exploration_noise_clip: float = 0.20
    replay_capacity_env_steps: int = 10_000
    replay_sampling_mode: str = REPLAY_SAMPLING_RAIL
    batch_size: int = 1_024
    minimum_replay_env_steps: int = 100
    minimum_action_enabled_env_steps: int = 100
    updates_per_env_step: int = 1
    learn_every_env_steps: int = 1
    latest_checkpoint_interval: int = 1_000
    periodic_checkpoint_interval: int = 5_000
    checkpoint_root: str | None = None
    resume_checkpoint_path: str | None = None
    resume_inference_until_replay_full: bool = False
    resume_deterministic_first_episode: bool = False
    rail_tat_diagnostic_path: str | None = None
    rail_tat_diagnostic_max_step: int = 1_000
    wandb_enabled: bool = False
    wandb_log_interval: int = 10
    sale_enabled: bool = True
    lap_enabled: bool = True
    critic_loss_mode: str = "auto"
    early_stop_queued_threshold: float = 500.0
    tat_termination_policy: str = TAT_TERMINATION_REWARD_PROFILE
    tat_termination_start_episode: int | None = None
    tat_termination_enabled: bool | None = None
    tat_termination_grace_steps: int | None = None
    early_stop_tat_threshold: float | None = None
    tat_above_threshold_patience: int | None = None
    tat_termination_inclusive: bool | None = None
    terminal_tat_penalty: float | None = None
    max_stale_sim_time_ticks: int = 5
    dispatch_mode: str = DISPATCH_FIRST_MATCH
    minimum_sign_sample_count: int = 100

    @property
    def effective_warmup_steps(self) -> int:
        return (
            0
            if self.state_normalizer_warmup_bypass
            or self.load_state_normalizer_path is not None
            else int(self.warmup_steps)
        )

    @property
    def resume_refill_target_env_steps(self) -> int:
        return int(
            self.replay_capacity_env_steps
            if self.resume_inference_until_replay_full
            else self.minimum_replay_env_steps
        )

    @property
    def effective_tat_termination_start_episode(self) -> int:
        return (
            1
            if self.state_normalizer_warmup_bypass
            else int(self.tat_termination_start_episode)
        )

    def __post_init__(self):
        canonical_version = (
            canonical_reward_version(self.reward_version)
        )
        object.__setattr__(
            self, "reward_version", canonical_version
        )
        incompatible = []
        contract = reward_contract(canonical_version)
        termination_profile = tat_termination_profile(
            self.tat_termination_policy, contract
        )
        termination_incompatible = []
        for name, expected in termination_profile.items():
            actual = getattr(self, name)
            if actual is None:
                object.__setattr__(self, name, expected)
            elif actual != expected:
                target = (
                    incompatible
                    if self.tat_termination_policy
                    == TAT_TERMINATION_REWARD_PROFILE
                    else termination_incompatible
                )
                target.append((name, actual, expected))
        if self.terminal_tat_penalty is None:
            object.__setattr__(
                self, "terminal_tat_penalty", contract.terminal_tat_penalty
            )
        elif self.terminal_tat_penalty != contract.terminal_tat_penalty:
            incompatible.append((
                "terminal_tat_penalty",
                self.terminal_tat_penalty,
                contract.terminal_tat_penalty,
            ))
        if termination_incompatible:
            details = ", ".join(
                f"{name}={actual!r} (expected {expected!r})"
                for name, actual, expected in termination_incompatible
            )
            raise ValueError(
                f"TAT termination policy {self.tat_termination_policy} is "
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
        if self.mode not in {"baseline_only", "actor_inference", "training"}:
            raise ValueError(
                "mode must be baseline_only, actor_inference, or training"
            )
        if self.mode == "training" and not self.action_enabled:
            raise ValueError("training mode requires explicit action_enabled")
        for name in ("num_stacks", "stack_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        if self.replay_sampling_mode not in REPLAY_SAMPLING_MODES:
            raise ValueError(
                f"replay_sampling_mode must be one of {REPLAY_SAMPLING_MODES}"
            )
        if self.dispatch_mode not in DISPATCH_MODES:
            raise ValueError(
                f"dispatch_mode must be one of {DISPATCH_MODES}"
            )
        if (
            self.replay_sampling_mode
            in {REPLAY_SAMPLING_SNAPSHOT, REPLAY_SAMPLING_RANDOM_RAIL}
            and self.lap_enabled
        ):
            raise ValueError(
                f"{self.replay_sampling_mode} replay sampling requires "
                "lap_enabled=False"
            )
        if (
            not np.isfinite(self.action_scale)
            or self.action_scale < 0.0
            or self.action_scale > 1.0
        ):
            raise ValueError("action_scale must be finite and in [0, 1]")
        if (
            isinstance(self.episode_burnin_steps, bool)
            or not isinstance(self.episode_burnin_steps, Integral)
        ):
            raise ValueError("episode_burnin_steps must be an integer")
        if (
            self.warmup_steps < 0
            or self.episode_burnin_steps < 0
            or self.normalizer_freeze_steps < 0
        ):
            raise ValueError("warmup/burn-in/freeze steps must be non-negative")
        for name in (
            "load_state_normalizer_path",
            "save_state_normalizer_path",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty path string")
        if not isinstance(self.state_normalizer_warmup_bypass, bool):
            raise ValueError("state_normalizer_warmup_bypass must be bool")
        if not isinstance(self.terminate_on_warmup_complete, bool):
            raise ValueError("terminate_on_warmup_complete must be bool")
        if not isinstance(self.resume_inference_until_replay_full, bool):
            raise ValueError(
                "resume_inference_until_replay_full must be bool"
            )
        if not isinstance(self.resume_deterministic_first_episode, bool):
            raise ValueError(
                "resume_deterministic_first_episode must be bool"
            )
        if self.load_state_normalizer_path is not None:
            object.__setattr__(
                self, "state_normalizer_warmup_bypass", True
            )
        if self.resume_checkpoint_path is not None and (
            self.load_state_normalizer_path is not None
        ):
            raise ValueError(
                "standalone state-normalizer load and --resume-checkpoint "
                "are mutually exclusive; checkpoint resume already restores "
                "the state normalizers"
            )
        if (
            self.state_normalizer_warmup_bypass
            and self.load_state_normalizer_path is None
            and self.resume_checkpoint_path is None
        ):
            raise ValueError(
                "state_normalizer_warmup_bypass requires a standalone "
                "normalizer load or checkpoint resume"
            )
        if (
            self.resume_inference_until_replay_full
            and self.resume_checkpoint_path is None
        ):
            raise ValueError(
                "resume_inference_until_replay_full requires checkpoint "
                "resume"
            )
        if (
            self.resume_deterministic_first_episode
            and self.resume_checkpoint_path is None
        ):
            raise ValueError(
                "resume_deterministic_first_episode requires checkpoint "
                "resume"
            )
        if self.mode == "training" and self.action_scale <= 0:
            raise ValueError("training mode requires positive action_scale")
        if (
            self.curriculum_end_step <= self.effective_warmup_steps
            or not 0 < self.curriculum_scale_start <= 1
            or not 0 < self.curriculum_scale_end <= 1
            or self.curriculum_scale_start > self.curriculum_scale_end
            or self.curriculum_shape not in {"geometric", "linear"}
        ):
            raise ValueError("invalid action curriculum configuration")
        parse_diagnostic_windows(self.reward_diagnostic_windows)
        if self.minimum_sign_sample_count <= 0:
            raise ValueError("minimum_sign_sample_count must be positive")
        if (
            not np.isfinite(self.exploration_noise_std)
            or self.exploration_noise_std < 0
            or not np.isfinite(self.exploration_noise_final_std)
            or self.exploration_noise_final_std < 0
        ):
            raise ValueError("exploration noise stds must be finite and non-negative")
        positive = (
            self.replay_capacity_env_steps, self.batch_size,
            self.minimum_replay_env_steps,
            self.minimum_action_enabled_env_steps,
            self.updates_per_env_step, self.learn_every_env_steps,
            self.exploration_noise_anneal_steps,
            self.latest_checkpoint_interval,
            self.periodic_checkpoint_interval, self.wandb_log_interval,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("training counts/intervals must be positive")
        if (
            self.early_stop_queued_threshold <= 0
            or not np.isfinite(self.early_stop_tat_threshold)
            or self.early_stop_tat_threshold <= 0
            or self.max_stale_sim_time_ticks <= 0
        ):
            raise ValueError("early-stop/watchdog settings are invalid")
        if (
            isinstance(self.tat_termination_grace_steps, bool)
            or not isinstance(self.tat_termination_grace_steps, Integral)
            or self.tat_termination_grace_steps < 0
            or isinstance(self.tat_termination_start_episode, bool)
            or not isinstance(self.tat_termination_start_episode, Integral)
            or self.tat_termination_start_episode <= 0
            or isinstance(self.tat_above_threshold_patience, bool)
            or not isinstance(self.tat_above_threshold_patience, Integral)
            or self.tat_above_threshold_patience <= 0
        ):
            raise ValueError("TAT termination counts are invalid")
        if (
            not np.isfinite(self.terminal_tat_penalty)
            or self.terminal_tat_penalty > 0.0
        ):
            raise ValueError(
                "terminal_tat_penalty must be finite and non-positive"
            )
        if self.rail_tat_diagnostic_max_step < 0:
            raise ValueError(
                "rail_tat_diagnostic_max_step must be non-negative"
            )
        self.make_reward_config()

    def make_reward_config(self) -> ContextualRewardConfig:
        """Resolve and validate the complete named reward contract."""
        return ContextualRewardConfig.for_version(
            self.reward_version,
            action_mode=self.action_mode,
        )


def seed_everything(seed):
    """Seed process-wide RNGs and request deterministic Torch algorithms."""
    seed = int(seed)
    if seed < 0:
        raise ValueError("--seed must be non-negative")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def restore_checkpoint_runtime_config(config_kwargs, checkpoint_path):
    """Apply saved Reward N settings while preserving launch controls."""
    saved, complete = read_contextual_runtime_config(checkpoint_path)
    valid_fields = set(ContextualRuntimeConfig.__dataclass_fields__)
    restored = []
    for key, value in saved.items():
        if key not in valid_fields or key in RESUME_LAUNCH_CONTROL_FIELDS:
            continue
        config_kwargs[key] = value
        restored.append(key)
    if "tat_termination_policy" not in saved:
        config_kwargs["tat_termination_policy"] = (
            TAT_TERMINATION_REWARD_PROFILE
        )
        config_kwargs["tat_termination_start_episode"] = 1
    config_kwargs["state_normalizer_warmup_bypass"] = True
    source = "full" if complete else "legacy-partial"
    print(
        "[checkpoint-config] "
        f"version={RESUME_RUNTIME_CONFIG_VERSION}, source={source}, "
        f"restored={','.join(sorted(restored)) or 'none'}"
    )
    if not complete:
        print(
            "[checkpoint-config] v3 checkpoint does not contain every runtime "
            "setting; unavailable fields retain the current CLI/default value."
        )
    return config_kwargs


def runtime_config_from_args(args) -> ContextualRuntimeConfig:
    """Build the validated runtime config from CLI and optional resume state."""
    fields = ContextualRuntimeConfig.__dataclass_fields__
    config_kwargs = {
        name: getattr(args, name)
        for name in fields
        if name != "device" and hasattr(args, name)
    }
    for config_name, argument_name in _ARG_ALIASES.items():
        value = getattr(args, argument_name)
        if config_name in _PATH_CONFIG_FIELDS and value is not None:
            value = str(value)
        config_kwargs[config_name] = value

    if args.device:
        config_kwargs["device"] = args.device
    if args.resume_checkpoint:
        config_kwargs = restore_checkpoint_runtime_config(
            config_kwargs, args.resume_checkpoint
        )
    return ContextualRuntimeConfig(**config_kwargs)


__all__ = (
    "ContextualRuntimeConfig",
    "RESUME_RUNTIME_CONFIG_VERSION",
    "restore_checkpoint_runtime_config",
    "runtime_config_from_args",
    "seed_everything",
)
