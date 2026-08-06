"""Strict contextual SAC checkpoints without replay payloads."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from contextual_action import action_version
from contextual_observation import OBSERVATION_VERSION
from contextual_reward import REWARD_VERSION

from cocel_rl.algorithms.contextual_td7.replay_buffer import REPLAY_VERSION

from .config import ALGORITHM_VERSION, LEARNER_VERSION


CHECKPOINT_VERSION = "contextual_sac_checkpoint_v1"


class ContextualSACCheckpointError(RuntimeError):
    pass


def read_contextual_sac_runtime_config(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ContextualSACCheckpointError(
            f"incompatible SAC checkpoint: {payload.get('checkpoint_version')!r}"
        )
    saved = payload.get("runtime_config")
    return (dict(saved), True) if isinstance(saved, dict) else ({}, False)


def save_contextual_sac_checkpoint(
    path,
    learner,
    *,
    observation_builder,
    reward_builder,
    runtime_metadata=None,
    runtime_config=None,
    exploration_rng=None,
):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "learner_version": LEARNER_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "reward_version": REWARD_VERSION,
        "action_version": action_version(learner.config.action_mode),
        "replay_version": REPLAY_VERSION,
        "topology_hash": learner.replay.topology.topology_hash,
        "mapping_hash": learner.replay.topology.mapping_hash,
        "network_config": asdict(learner.network_config),
        "learner_config": asdict(learner.config),
        "applied_action_scale": learner.applied_action_scale,
        "online_encoder": learner.encoder.state_dict(),
        "online_actor": learner.actor.state_dict(),
        "online_critic": learner.critic.state_dict(),
        "target_encoder": learner.target_encoder.state_dict(),
        "target_critic": learner.target_critic.state_dict(),
        "encoder_optimizer": learner.encoder_optimizer.state_dict(),
        "actor_optimizer": learner.actor_optimizer.state_dict(),
        "critic_optimizer": learner.critic_optimizer.state_dict(),
        "log_alpha": learner.log_alpha.detach().cpu(),
        "alpha_optimizer": learner.alpha_optimizer.state_dict(),
        "learner_update_count": learner.learner_update_count,
        "actor_update_count": learner.actor_update_count,
        "target_update_count": learner.target_update_count,
        "last_actor_loss": learner.last_actor_loss,
        "last_actor_grad_norm": learner.last_actor_grad_norm,
        "last_actor_update_step": learner.last_actor_update_step,
        "observation_local_normalizer": observation_builder.local_normalizer.state_dict(),
        "observation_global_normalizer": observation_builder.global_normalizer.state_dict(),
        "reward_local_normalizer": reward_builder.local_normalizer.state_dict(),
        "reward_global_normalizer": reward_builder.global_normalizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "replay_rng_state": learner.replay.rng.bit_generator.state,
        "exploration_rng_state": (
            exploration_rng.bit_generator.state
            if exploration_rng is not None else None
        ),
        "replay_saved": False,
        "runtime_metadata": dict(runtime_metadata or {}),
        "runtime_config": dict(runtime_config or {}),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_contextual_sac_checkpoint(
    path,
    learner,
    *,
    observation_builder,
    reward_builder,
    expected_runtime_metadata=None,
    exploration_rng=None,
    exploration_seed=0,
):
    payload = torch.load(Path(path), map_location=learner.device, weights_only=False)
    runtime_metadata = dict(payload.get("runtime_metadata", {}))
    if (
        runtime_metadata.get("checkpoint_kind") == "crash"
        or runtime_metadata.get("training_failed")
    ):
        raise ContextualSACCheckpointError(
            "crash/failure checkpoint is diagnostic-only and cannot be resumed"
        )
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "learner_version": LEARNER_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "reward_version": REWARD_VERSION,
        "action_version": action_version(learner.config.action_mode),
        "replay_version": REPLAY_VERSION,
        "topology_hash": learner.replay.topology.topology_hash,
        "mapping_hash": learner.replay.topology.mapping_hash,
        "network_config": asdict(learner.network_config),
        "learner_config": asdict(learner.config),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContextualSACCheckpointError(
                f"checkpoint {key} mismatch: saved={payload.get(key)!r}, runtime={value!r}"
            )
    for key, value in dict(expected_runtime_metadata or {}).items():
        if runtime_metadata.get(key) != value:
            raise ContextualSACCheckpointError(
                f"runtime metadata {key} mismatch: saved={runtime_metadata.get(key)!r}, runtime={value!r}"
            )

    learner.encoder.load_state_dict(payload["online_encoder"])
    learner.actor.load_state_dict(payload["online_actor"])
    learner.critic.load_state_dict(payload["online_critic"])
    learner.target_encoder.load_state_dict(payload["target_encoder"])
    learner.target_critic.load_state_dict(payload["target_critic"])
    learner.encoder_optimizer.load_state_dict(payload["encoder_optimizer"])
    learner.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    learner.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    learner.log_alpha.data.copy_(payload["log_alpha"].to(learner.device))
    learner.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
    learner.learner_update_count = int(payload["learner_update_count"])
    learner.actor_update_count = int(payload["actor_update_count"])
    learner.target_update_count = int(payload["target_update_count"])
    learner.last_actor_loss = float(payload["last_actor_loss"])
    learner.last_actor_grad_norm = float(payload["last_actor_grad_norm"])
    learner.last_actor_update_step = int(payload["last_actor_update_step"])
    learner.set_applied_action_scale(float(payload["applied_action_scale"]))
    observation_builder.local_normalizer.load_state_dict(
        payload["observation_local_normalizer"]
    )
    observation_builder.global_normalizer.load_state_dict(
        payload["observation_global_normalizer"]
    )
    reward_builder.local_normalizer.load_state_dict(
        payload["reward_local_normalizer"]
    )
    reward_builder.global_normalizer.load_state_dict(
        payload["reward_global_normalizer"]
    )
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    if torch.cuda.is_available() and payload["cuda_random_state"] is not None:
        torch.cuda.set_rng_state_all([
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in payload["cuda_random_state"]
        ])
    learner.replay.rng.bit_generator.state = payload["replay_rng_state"]
    if exploration_rng is not None:
        state = payload.get("exploration_rng_state")
        if state is not None:
            exploration_rng.bit_generator.state = state
        else:
            fallback = np.random.default_rng(np.random.SeedSequence([
                int(exploration_seed),
                int(runtime_metadata.get("runtime_env_step", 0)),
            ]))
            exploration_rng.bit_generator.state = fallback.bit_generator.state
    return runtime_metadata
