"""Validated runtime configuration for the current contextual TD7."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from oht_dispatching.config import DISPATCH_FIRST_MATCH
from oht_routing.algorithms.rl.contextual_td7 import (
    REPLAY_SAMPLING_RAIL,
    read_contextual_runtime_config,
)
from oht_routing.mdp.action import REGION_B_RL
from oht_routing.mdp.reward.config import REWARD_VERSION
from oht_routing.mdp.termination import TAT_TERMINATION_REWARD_PROFILE
from oht_routing.runtime.console import (
    print_entries,
    print_header,
    print_wrapped_items,
)
from oht_routing.runtime.config_validation import validate_runtime_config
from oht_routing.version import CONTEXTUAL_VERSION


RESUME_LAUNCH_CONTROL_FIELDS = {
    "mode",
    "action_enabled",
    "reward_version",
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
    "console_log_interval",
    "stage",
    "sim_end_time",
    "save_data_enabled",
    "dispatch_mode",
    "batch_size",
    "replay_capacity_env_steps",
    "periodic_checkpoint_interval",
    "lap_enabled",
    "use_attention",
    "resume_inference_until_replay_full",
    "resume_deterministic_first_episode",
    "resume_warmstart_steps",
    "load_stage1_policy_path",
}
_PATH_CONFIG_FIELDS = {
    "load_state_normalizer_path",
    "save_state_normalizer_path",
    "load_stage1_policy_path",
}


@dataclass(frozen=True)
class ContextualRuntimeConfig:
    """Canonical runtime defaults; CLI values are explicit overrides only."""

    mode: str = "baseline_only"
    stage: int | None = None
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
    replay_capacity_env_steps: int = 100_000
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
    load_stage1_policy_path: str | None = None
    resume_inference_until_replay_full: bool = False
    resume_deterministic_first_episode: bool = False
    resume_warmstart_steps: int = 0
    rail_tat_diagnostic_path: str | None = None
    rail_tat_diagnostic_max_step: int = 1_000
    # None means the canonical mode-aware default: enabled for training and
    # actor inference, disabled for baseline-only execution.  CLI flags are
    # explicit overrides, never a second source of defaults.
    wandb_enabled: bool | None = None
    wandb_log_interval: int = 10
    console_log_interval: int = 100
    sim_end_time: int = 45_000
    save_data_enabled: bool = False
    sale_enabled: bool = True
    lap_enabled: bool = True
    use_attention: bool = False
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
            or self.load_stage1_policy_path is not None
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
        if self.wandb_enabled is None:
            object.__setattr__(
                self,
                "wandb_enabled",
                self.mode in {"training", "actor_inference"},
            )
        validate_runtime_config(self)


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
    """Apply compatible saved settings while preserving launch controls."""
    saved, _ = read_contextual_runtime_config(
        checkpoint_path,
        expected_reward_version=config_kwargs["reward_version"],
    )
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
    print_header("checkpoint-config")
    print_entries(
        (
            ("version", CONTEXTUAL_VERSION),
            ("source", "full"),
            ("restored", f"{len(restored)} fields"),
        ),
        indent=2,
    )
    if restored:
        print_wrapped_items(sorted(restored))
    else:
        print("    (none)")
    return config_kwargs


def runtime_config_from_args(args) -> ContextualRuntimeConfig:
    """Apply only explicit CLI overrides to the canonical config defaults."""
    fields = ContextualRuntimeConfig.__dataclass_fields__
    config_kwargs = {}
    for name in fields:
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if name in _PATH_CONFIG_FIELDS and value is not None:
            value = str(value)
        config_kwargs[name] = value

    config = ContextualRuntimeConfig(**config_kwargs)
    if config.resume_checkpoint_path:
        config_kwargs = restore_checkpoint_runtime_config(
            asdict(config), config.resume_checkpoint_path
        )
        config = ContextualRuntimeConfig(**config_kwargs)
    return config


__all__ = (
    "ContextualRuntimeConfig",
    "restore_checkpoint_runtime_config",
    "runtime_config_from_args",
    "seed_everything",
)
