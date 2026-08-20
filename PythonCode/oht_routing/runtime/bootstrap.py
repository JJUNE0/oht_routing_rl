"""Runtime configuration assembly and deterministic process setup."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from oht_routing.runtime.config import ContextualRuntimeConfig
from oht_routing.algorithms.rl.contextual_td7 import read_contextual_runtime_config
from oht_routing.mdp.termination import TAT_TERMINATION_REWARD_PROFILE


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


def seed_everything(seed):
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
    config_kwargs = {
        "mode": args.mode,
        "action_enabled": args.action_enabled,
        "reward_version": args.reward_version,
        "action_mode": args.action_mode,
        "dispatch_mode": args.dispatch_mode,
        "action_scale": args.action_scale,
        "num_stacks": args.num_stacks,
        "stack_interval": args.stack_interval,
        "curriculum_end_step": args.curriculum_end_step,
        "curriculum_scale_start": args.curriculum_scale_start,
        "curriculum_scale_end": args.curriculum_scale_end,
        "curriculum_shape": args.curriculum_shape,
        "warmup_steps": args.warmup_steps,
        "terminate_on_warmup_complete": args.terminate_on_warmup_complete,
        "episode_burnin_steps": args.episode_burnin_steps,
        "normalizer_freeze_steps": args.normalizer_freeze_steps,
        "load_state_normalizer_path": (
            None
            if args.load_state_normalizer is None
            else str(args.load_state_normalizer)
        ),
        "save_state_normalizer_path": (
            None
            if args.save_state_normalizer is None
            else str(args.save_state_normalizer)
        ),
        "reward_diagnostic_dir": args.reward_diagnostic_dir,
        "reward_diagnostic_windows": args.reward_diagnostic_windows,
        "seed": args.seed,
        "exploration_noise_std": args.exploration_noise_std,
        "exploration_noise_final_std": args.exploration_noise_final_std,
        "exploration_noise_anneal_steps": args.exploration_noise_anneal_steps,
        "exploration_noise_clip": args.exploration_noise_clip,
        "replay_capacity_env_steps": args.replay_capacity_env_steps,
        "replay_sampling_mode": args.replay_sampling_mode,
        "batch_size": args.batch_size,
        "minimum_replay_env_steps": args.minimum_replay_env_steps,
        "minimum_action_enabled_env_steps": (
            args.minimum_action_enabled_env_steps
        ),
        "updates_per_env_step": args.updates_per_env_step,
        "learn_every_env_steps": args.learn_every_env_steps,
        "wandb_enabled": args.wandb,
        "sale_enabled": args.sale,
        "lap_enabled": args.lap,
        "critic_loss_mode": args.critic_loss_mode,
        "wandb_log_interval": args.wandb_log_interval,
        "early_stop_queued_threshold": args.early_stop_queued_threshold,
        "max_stale_sim_time_ticks": args.max_stale_sim_time_ticks,
        "checkpoint_root": args.checkpoint_root,
        "resume_checkpoint_path": args.resume_checkpoint,
        "resume_inference_until_replay_full": (
            args.resume_inference_until_replay_full
        ),
        "resume_deterministic_first_episode": (
            args.resume_deterministic_first_episode
        ),
        "rail_tat_diagnostic_path": args.rail_tat_diagnostic,
        "rail_tat_diagnostic_max_step": args.rail_tat_diagnostic_max_step,
    }
    if args.device:
        config_kwargs["device"] = args.device
    if args.resume_checkpoint:
        config_kwargs = restore_checkpoint_runtime_config(
            config_kwargs, args.resume_checkpoint
        )
    return ContextualRuntimeConfig(**config_kwargs)


__all__ = (
    "RESUME_RUNTIME_CONFIG_VERSION",
    "restore_checkpoint_runtime_config",
    "runtime_config_from_args",
    "seed_everything",
)
