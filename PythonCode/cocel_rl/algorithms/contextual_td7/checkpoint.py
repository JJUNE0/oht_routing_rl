"""Strict contextual learner checkpoint without replay payload."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from contextual_action import action_version
from contextual_observation import OBSERVATION_VERSION
from contextual_reward import REWARD_VERSION

from .learner import LEARNER_VERSION
from .replay_buffer import REPLAY_VERSION
from .replay_buffer import LAP_VERSION
from .sale import SALE_VERSION


CHECKPOINT_VERSION = "contextual_td7_checkpoint_v3_independent_twin_critic"
CRITIC_INITIALIZATION = "independent"


class ContextualCheckpointError(RuntimeError):
    pass


def _normalizer_state(normalizer):
    return normalizer.state_dict()


def _load_normalizer(normalizer, state):
    normalizer.load_state_dict(state)


def _twin_state_max_abs_diff(state_dict) -> float:
    q1 = {
        key[3:]: value
        for key, value in state_dict.items()
        if key.startswith("q1.")
    }
    q2 = {
        key[3:]: value
        for key, value in state_dict.items()
        if key.startswith("q2.")
    }
    if not q1 or q1.keys() != q2.keys():
        raise ContextualCheckpointError(
            "checkpoint twin-critic parameter schema is invalid"
        )
    return max(
        float((q1[key].detach() - q2[key].detach()).abs().max().cpu())
        for key in q1
    )


def save_contextual_checkpoint(
    path,
    learner,
    *,
    observation_builder,
    reward_builder,
    runtime_metadata=None,
    exploration_rng=None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if learner.twin_parameter_diagnostics()[
        "critic/parameter_max_abs_diff"
    ] <= 0.0:
        raise ContextualCheckpointError(
            "refusing to save a symmetric online twin critic"
        )
    if _twin_state_max_abs_diff(
        learner.target_critic.state_dict()
    ) <= 0.0:
        raise ContextualCheckpointError(
            "refusing to save a symmetric target twin critic"
        )
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "critic_initialization": CRITIC_INITIALIZATION,
        "learner_version": LEARNER_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "reward_version": REWARD_VERSION,
        "action_version": action_version(learner.config.action_mode),
        "replay_version": REPLAY_VERSION,
        "topology_hash": learner.replay.topology.topology_hash,
        "mapping_hash": learner.replay.topology.mapping_hash,
        "network_config": asdict(learner.network_config),
        "learner_config": asdict(learner.config),
        "action_scale": learner.config.action_scale,
        "applied_action_scale": learner.applied_action_scale,
        "algorithm_variant": learner.config.algorithm_variant,
        "sale_enabled": learner.config.sale_enabled,
        "lap_enabled": learner.config.lap_enabled,
        "sale_version": SALE_VERSION,
        "lap_version": LAP_VERSION,
        "lap_metadata": {
            "enabled": learner.config.lap_enabled,
            "alpha": learner.config.lap_alpha,
            "min_priority": learner.config.lap_min_priority,
            "priority_array_saved": False,
            "resume_requires_replay_refill": True,
        },
        "online_encoder": learner.encoder.state_dict(),
        "online_actor": learner.actor.state_dict(),
        "online_critic": learner.critic.state_dict(),
        "target_encoder": learner.target_encoder.state_dict(),
        "target_actor": learner.target_actor.state_dict(),
        "target_critic": learner.target_critic.state_dict(),
        "encoder_optimizer": learner.encoder_optimizer.state_dict(),
        "actor_optimizer": learner.actor_optimizer.state_dict(),
        "critic_optimizer": learner.critic_optimizer.state_dict(),
        "learner_update_count": learner.learner_update_count,
        "actor_update_count": learner.actor_update_count,
        "last_actor_loss": learner.last_actor_loss,
        "last_actor_grad_norm": learner.last_actor_grad_norm,
        "last_actor_update_step": learner.last_actor_update_step,
        "target_update_count": learner.target_update_count,
        "observation_local_normalizer": _normalizer_state(
            observation_builder.local_normalizer
        ),
        "observation_global_normalizer": _normalizer_state(
            observation_builder.global_normalizer
        ),
        "reward_local_normalizer": _normalizer_state(
            reward_builder.local_normalizer
        ),
        "reward_global_normalizer": _normalizer_state(
            reward_builder.global_normalizer
        ),
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
    }
    if learner.config.sale_enabled:
        payload.update({
            "sale_online": learner.sale_online.state_dict(),
            "sale_fixed": learner.sale_fixed.state_dict(),
            "sale_target_fixed": learner.sale_target_fixed.state_dict(),
            "sale_optimizer": learner.sale_optimizer.state_dict(),
            "sale_update_count": learner.sale_update_count,
            "sale_fixed_generation": learner.sale_fixed_generation,
        })
    payload["current_target_q_min"] = learner.current_target_q_min
    payload["current_target_q_max"] = learner.current_target_q_max
    payload["fixed_target_q_min"] = learner.fixed_target_q_min
    payload["fixed_target_q_max"] = learner.fixed_target_q_max
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_contextual_checkpoint(
    path,
    learner,
    *,
    observation_builder,
    reward_builder,
    expected_runtime_metadata=None,
    exploration_rng=None,
    exploration_seed=0,
) -> None:
    payload = torch.load(
        Path(path), map_location=learner.device, weights_only=False
    )
    runtime_metadata = dict(payload.get("runtime_metadata", {}))
    if (
        runtime_metadata.get("checkpoint_kind") == "crash"
        or runtime_metadata.get("training_failed") is True
    ):
        raise ContextualCheckpointError(
            "Crash checkpoint resume refused. "
            "Crash checkpoints are diagnostic artifacts only."
        )
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ContextualCheckpointError(
            "Legacy or incompatible contextual checkpoint resume refused: "
            f"saved={payload.get('checkpoint_version')!r}, "
            f"runtime={CHECKPOINT_VERSION!r}. A fresh independent twin-critic "
            "run is required."
        )
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "critic_initialization": CRITIC_INITIALIZATION,
        "learner_version": LEARNER_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "reward_version": REWARD_VERSION,
        "action_version": action_version(learner.config.action_mode),
        "replay_version": REPLAY_VERSION,
        "topology_hash": learner.replay.topology.topology_hash,
        "mapping_hash": learner.replay.topology.mapping_hash,
        "network_config": asdict(learner.network_config),
        "learner_config": asdict(learner.config),
        "action_scale": learner.config.action_scale,
        "algorithm_variant": learner.config.algorithm_variant,
        "sale_enabled": learner.config.sale_enabled,
        "lap_enabled": learner.config.lap_enabled,
        "sale_version": SALE_VERSION,
        "lap_version": LAP_VERSION,
        "lap_metadata": {
            "enabled": learner.config.lap_enabled,
            "alpha": learner.config.lap_alpha,
            "min_priority": learner.config.lap_min_priority,
            "priority_array_saved": False,
            "resume_requires_replay_refill": True,
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContextualCheckpointError(
                f"checkpoint {key} mismatch: saved={payload.get(key)!r}, "
                f"runtime={value!r}"
            )
    for state_key in ("online_critic", "target_critic"):
        if _twin_state_max_abs_diff(payload[state_key]) <= 0.0:
            raise ContextualCheckpointError(
                "symmetric twin-critic checkpoint resume refused: "
                f"{state_key} has identical Q1/Q2 parameters"
            )
    for key, value in dict(expected_runtime_metadata or {}).items():
        saved = payload.get("runtime_metadata", {}).get(key)
        if saved != value:
            raise ContextualCheckpointError(
                f"checkpoint runtime metadata {key} mismatch: "
                f"saved={saved!r}, runtime={value!r}"
            )
    learner.encoder.load_state_dict(payload["online_encoder"])
    learner.actor.load_state_dict(payload["online_actor"])
    learner.critic.load_state_dict(payload["online_critic"])
    learner.target_encoder.load_state_dict(payload["target_encoder"])
    learner.target_actor.load_state_dict(payload["target_actor"])
    learner.target_critic.load_state_dict(payload["target_critic"])
    learner._distance_diagnostics.update(
        learner.twin_parameter_diagnostics()
    )
    if learner.config.sale_enabled:
        learner.sale_online.load_state_dict(payload["sale_online"])
        learner.sale_fixed.load_state_dict(payload["sale_fixed"])
        learner.sale_target_fixed.load_state_dict(payload["sale_target_fixed"])
        learner.sale_optimizer.load_state_dict(payload["sale_optimizer"])
        learner.sale_update_count = int(payload["sale_update_count"])
        learner.sale_fixed_generation = int(payload["sale_fixed_generation"])
    learner.encoder_optimizer.load_state_dict(payload["encoder_optimizer"])
    learner.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    learner.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    learner.learner_update_count = int(payload["learner_update_count"])
    learner.actor_update_count = int(payload["actor_update_count"])
    learner.target_update_count = int(payload["target_update_count"])
    learner.last_actor_loss = float(payload["last_actor_loss"])
    learner.last_actor_grad_norm = float(payload["last_actor_grad_norm"])
    learner.last_actor_update_step = int(payload["last_actor_update_step"])
    learner.set_applied_action_scale(
        float(payload.get("applied_action_scale", learner.config.action_scale))
    )
    learner.current_target_q_min = float(payload["current_target_q_min"])
    learner.current_target_q_max = float(payload["current_target_q_max"])
    learner.fixed_target_q_min = float(payload["fixed_target_q_min"])
    learner.fixed_target_q_max = float(payload["fixed_target_q_max"])
    _load_normalizer(
        observation_builder.local_normalizer,
        payload["observation_local_normalizer"],
    )
    _load_normalizer(
        observation_builder.global_normalizer,
        payload["observation_global_normalizer"],
    )
    _load_normalizer(
        reward_builder.local_normalizer,
        payload["reward_local_normalizer"],
    )
    _load_normalizer(
        reward_builder.global_normalizer,
        payload["reward_global_normalizer"],
    )
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    if torch.cuda.is_available() and payload["cuda_random_state"] is not None:
        # The checkpoint itself is loaded with map_location=learner.device.
        # On CUDA resume that also moves saved RNG byte tensors to CUDA, while
        # PyTorch's RNG restore API requires CPU ByteTensors.
        cuda_rng_states = [
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in payload["cuda_random_state"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)
    learner.replay.rng.bit_generator.state = payload["replay_rng_state"]
    if exploration_rng is not None:
        exploration_state = payload.get("exploration_rng_state")
        if exploration_state is not None:
            exploration_rng.bit_generator.state = exploration_state
        else:
            seed_sequence = np.random.SeedSequence([
                int(exploration_seed),
                int(runtime_metadata.get("runtime_env_step", 0)),
                int(runtime_metadata.get("episode_id", 0)),
            ])
            fallback = np.random.default_rng(seed_sequence)
            exploration_rng.bit_generator.state = fallback.bit_generator.state
    return dict(payload.get("runtime_metadata", {}))
