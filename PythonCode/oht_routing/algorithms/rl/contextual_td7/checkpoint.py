"""Strict contextual learner checkpoint without replay payload."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from oht_routing.version import (
    CONTEXTUAL_VERSION,
    is_compatible_contextual_version,
)


PROMOTED_CHECKPOINTS: dict[str, str] = {}
CRITIC_INITIALIZATION = "independent"


class ContextualCheckpointError(RuntimeError):
    pass


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint_version(
    path: str | Path,
    payload: dict,
    *,
    announce_promotion: bool = False,
) -> str:
    """Return the compatible v3 version; v2 observation artifacts are rejected."""
    saved_version = payload.get("version")
    if is_compatible_contextual_version(saved_version):
        return CONTEXTUAL_VERSION

    legacy_version = payload.get("checkpoint_version")
    if PROMOTED_CHECKPOINTS:
        target = Path(path)
        fingerprint = _checkpoint_sha256(target)
        promoted_legacy_version = PROMOTED_CHECKPOINTS.get(fingerprint)
        if (
            promoted_legacy_version is not None
            and promoted_legacy_version == legacy_version
        ):
            if announce_promotion:
                print(
                    "[checkpoint-compat] promoted checkpoint artifact: "
                    f"{legacy_version} -> {CONTEXTUAL_VERSION}, "
                    f"sha256={fingerprint}",
                    flush=True,
                )
            return CONTEXTUAL_VERSION

    raise ContextualCheckpointError(
        "checkpoint version mismatch: "
        f"saved={saved_version or legacy_version!r}, "
        f"runtime={CONTEXTUAL_VERSION!r}. A checkpoint must use the current "
        "major version without being newer than the runtime. The former v2 "
        "step_400000.pt promotion is intentionally incompatible with the v3 "
        "observation and network contract."
    )


def read_contextual_runtime_config(path) -> tuple[dict, bool]:
    """Read saved runtime settings before constructing the runtime.

    Current checkpoints contain the complete ContextualRuntimeConfig.
    Older-major artifacts are rejected before their runtime configuration is
    applied.
    """
    target = Path(path)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    _resolve_checkpoint_version(target, payload)
    saved = payload.get("runtime_config")
    if isinstance(saved, dict) and saved:
        return dict(saved), True
    raise ContextualCheckpointError(
        "checkpoint runtime_config is missing; this artifact cannot be "
        "restored under the unified version contract"
    )


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
    runtime_config=None,
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
        "version": CONTEXTUAL_VERSION,
        "critic_initialization": CRITIC_INITIALIZATION,
        "reward_version": reward_builder.reward_version,
        "action_mode": learner.config.action_mode,
        "topology_hash": learner.replay.topology.topology_hash,
        "mapping_hash": learner.replay.topology.mapping_hash,
        "network_config": asdict(learner.network_config),
        "learner_config": asdict(learner.config),
        "action_scale": learner.config.action_scale,
        "applied_action_scale": learner.applied_action_scale,
        "algorithm_variant": learner.config.algorithm_variant,
        "sale_enabled": learner.config.sale_enabled,
        "lap_enabled": learner.config.lap_enabled,
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
        "reward_steps": int(reward_builder.reward_steps),
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
    exploration_rng=None,
    exploration_seed=0,
) -> None:
    target = Path(path)
    payload = torch.load(target, map_location=learner.device, weights_only=False)
    runtime_metadata = dict(payload.get("runtime_metadata", {}))
    if (
        runtime_metadata.get("checkpoint_kind") == "crash"
        or runtime_metadata.get("training_failed") is True
    ):
        raise ContextualCheckpointError(
            "Crash checkpoint resume refused. "
            "Crash checkpoints are diagnostic artifacts only."
        )
    _resolve_checkpoint_version(target, payload, announce_promotion=True)
    expected = {
        "topology_hash": learner.replay.topology.topology_hash,
        "mapping_hash": learner.replay.topology.mapping_hash,
        "network_config": asdict(learner.network_config),
        "action_mode": learner.config.action_mode,
        "reward_version": reward_builder.reward_version,
        "sale_enabled": learner.config.sale_enabled,
        "lap_enabled": learner.config.lap_enabled,
    }
    for key, value in expected.items():
        saved = payload.get(key)
        if key == "action_mode" and saved is None:
            saved = dict(payload.get("learner_config", {})).get(key)
        if saved == value:
            continue
        raise ContextualCheckpointError(
            f"checkpoint {key} mismatch: saved={saved!r}, "
            f"runtime={value!r}"
        )
    for state_key in ("online_critic", "target_critic"):
        if _twin_state_max_abs_diff(payload[state_key]) <= 0.0:
            raise ContextualCheckpointError(
                "symmetric twin-critic checkpoint resume refused: "
                f"{state_key} has identical Q1/Q2 parameters"
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
    reward_builder.reward_steps = int(payload["reward_steps"])
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
