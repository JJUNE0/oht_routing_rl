"""Strict contextual learner checkpoint without replay payload."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class FrozenContextualPolicy:
    """Inference-only Stage 1 modules isolated from the Stage 2 learner."""

    encoder: torch.nn.Module
    actor: torch.nn.Module
    sale_fixed: torch.nn.Module | None
    action_mode: str
    applied_action_scale: float
    checkpoint_path: Path
    checkpoint_sha256: str
    runtime_metadata: dict


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
    """Return the compatible current-major version."""
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
        "major version without being newer than the runtime. Earlier-major "
        "observation and network contracts are intentionally incompatible."
    )


def _validate_checkpoint_reward_identity(
    payload: dict,
    *,
    expected_reward_version: str | None = None,
) -> None:
    saved_reward_version = payload.get("reward_version")
    runtime_config = payload.get("runtime_config")
    configured_reward_version = (
        runtime_config.get("reward_version")
        if isinstance(runtime_config, dict)
        else None
    )
    if (
        configured_reward_version is not None
        and configured_reward_version != saved_reward_version
    ):
        raise ContextualCheckpointError(
            "checkpoint reward_version mismatch: "
            f"payload={saved_reward_version!r}, "
            f"runtime_config={configured_reward_version!r}"
        )
    if (
        expected_reward_version is not None
        and saved_reward_version != expected_reward_version
    ):
        raise ContextualCheckpointError(
            "checkpoint reward_version mismatch: "
            f"saved={saved_reward_version!r}, "
            f"runtime={expected_reward_version!r}"
        )


def read_contextual_runtime_config(
    path,
    *,
    expected_reward_version: str | None = None,
) -> tuple[dict, bool]:
    """Read saved runtime settings before constructing the runtime.

    Current checkpoints contain the complete ContextualRuntimeConfig.
    Older-major artifacts are rejected before their runtime configuration is
    applied.
    """
    target = Path(path)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if expected_reward_version is not None:
        _validate_checkpoint_reward_identity(
            payload,
            expected_reward_version=expected_reward_version,
        )
    _resolve_checkpoint_version(target, payload)
    if expected_reward_version is None:
        _validate_checkpoint_reward_identity(payload)
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


def _freeze_policy_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    module.eval()
    for parameter in module.parameters():
        if not torch.isfinite(parameter).all():
            raise ContextualCheckpointError(
                "frozen Stage 1 policy contains NaN or Inf"
            )
        parameter.requires_grad_(False)


def _validate_frozen_normalizer_state(name: str, state: object) -> None:
    if not isinstance(state, dict):
        raise ContextualCheckpointError(
            f"checkpoint {name} normalizer state is missing"
        )
    if not bool(state.get("frozen", False)):
        raise ContextualCheckpointError(
            f"checkpoint {name} normalizer must be frozen"
        )
    if int(state.get("count", 0)) <= 0:
        raise ContextualCheckpointError(
            f"checkpoint {name} normalizer must be populated"
        )


def _initialize_fresh_learner_policy(
    learner,
    *,
    frozen_encoder: torch.nn.Module,
    frozen_actor: torch.nn.Module,
    frozen_sale: torch.nn.Module | None,
) -> None:
    """Warm-start a pristine learner's policy path without resuming training.

    Stage 2 keeps a fresh critic, optimizer state, replay, counters, Q bounds,
    and RNG streams.  Loading state into the existing module objects also
    preserves the parameter references already owned by the fresh optimizers.
    """
    counters = {
        name: int(getattr(learner, name, 0))
        for name in (
            "learner_update_count",
            "actor_update_count",
            "target_update_count",
            "sale_update_count",
            "sale_fixed_generation",
        )
    }
    replay_size = int(getattr(learner.replay, "size_env_steps", 0))
    optimizers = tuple(
        optimizer
        for optimizer in (
            getattr(learner, "encoder_optimizer", None),
            getattr(learner, "actor_optimizer", None),
            getattr(learner, "critic_optimizer", None),
            getattr(learner, "sale_optimizer", None),
        )
        if optimizer is not None
    )
    if (
        bool(getattr(learner, "_stage1_policy_warm_started", False))
        or replay_size != 0
        or any(value != 0 for value in counters.values())
        or any(bool(optimizer.state) for optimizer in optimizers)
    ):
        raise ContextualCheckpointError(
            "Stage 2 policy warm-start requires a pristine learner with an "
            "empty replay, zero update counters, and fresh optimizers"
        )

    learner.encoder.load_state_dict(frozen_encoder.state_dict(), strict=True)
    learner.actor.load_state_dict(frozen_actor.state_dict(), strict=True)
    # A new Stage 2 clock must not inherit the lagged Stage 1 target clock.
    # Start its policy targets exactly at the behaviorally active online copy.
    learner.target_encoder.load_state_dict(
        frozen_encoder.state_dict(), strict=True
    )
    learner.target_actor.load_state_dict(frozen_actor.state_dict(), strict=True)

    if learner.config.sale_enabled:
        if frozen_sale is None:
            raise ContextualCheckpointError(
                "SALE-enabled Stage 2 policy warm-start requires sale_fixed"
            )
        sale_state = frozen_sale.state_dict()
        # The saved actor consumed sale_fixed.  Use that same basis for every
        # fresh Stage 2 SALE role so the first target update cannot replace it
        # with a random online representation.
        learner.sale_online.load_state_dict(sale_state, strict=True)
        learner.sale_fixed.load_state_dict(sale_state, strict=True)
        learner.sale_target_fixed.load_state_dict(sale_state, strict=True)
    elif frozen_sale is not None:
        raise ContextualCheckpointError(
            "SALE-disabled Stage 2 policy received an unexpected SALE module"
        )
    learner._stage1_policy_warm_started = True


def load_frozen_contextual_policy(
    path,
    learner,
    *,
    observation_builder,
    expected_reward_version: str,
    initialize_fresh_learner_policy: bool = False,
) -> FrozenContextualPolicy:
    """Load the policy-side Stage 1 contract for a Stage 2 prefix.

    By default this only creates the isolated frozen prefix policy.  A fresh
    Stage 2 run may additionally initialize its online/target policy path from
    that behavior via ``initialize_fresh_learner_policy=True``.  This never
    restores the Stage 1 critic, optimizer state, replay, counters, reward
    state, Q bounds, applied-action scale, or process RNG streams.
    """
    if not isinstance(initialize_fresh_learner_policy, bool):
        raise TypeError("initialize_fresh_learner_policy must be bool")
    target = Path(path)
    checkpoint_sha256 = _checkpoint_sha256(target)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    runtime_metadata = dict(payload.get("runtime_metadata", {}))
    if (
        runtime_metadata.get("checkpoint_kind") == "crash"
        or runtime_metadata.get("training_failed") is True
    ):
        raise ContextualCheckpointError(
            "Crash checkpoint Stage 1 policy load refused. "
            "Crash checkpoints are diagnostic artifacts only."
        )
    if runtime_metadata.get("stage") != 1:
        raise ContextualCheckpointError(
            "Stage 2 prefix policy load requires a Stage 1 checkpoint: "
            f"saved_stage={runtime_metadata.get('stage')!r}"
        )
    _validate_checkpoint_reward_identity(
        payload,
        expected_reward_version=expected_reward_version,
    )
    _resolve_checkpoint_version(target, payload, announce_promotion=True)

    saved_action_mode = payload.get("action_mode")
    if saved_action_mode is None:
        saved_action_mode = dict(payload.get("learner_config", {})).get(
            "action_mode"
        )

    expected = {
        "topology_hash": learner.replay.topology.topology_hash,
        "mapping_hash": learner.replay.topology.mapping_hash,
        "network_config": asdict(learner.network_config),
        "action_mode": learner.config.action_mode,
        "sale_enabled": learner.config.sale_enabled,
    }
    for key, value in expected.items():
        saved = (
            saved_action_mode if key == "action_mode" else payload.get(key)
        )
        if saved != value:
            raise ContextualCheckpointError(
                f"Stage 1 checkpoint {key} mismatch: saved={saved!r}, "
                f"stage2={value!r}"
            )
    saved_learner_config = dict(payload.get("learner_config", {}))
    for key in ("sale_embedding_dim", "sale_feature_dim"):
        if not learner.config.sale_enabled:
            break
        saved = saved_learner_config.get(key)
        expected_value = getattr(learner.config, key)
        if saved != expected_value:
            raise ContextualCheckpointError(
                f"Stage 1 checkpoint {key} mismatch: saved={saved!r}, "
                f"stage2={expected_value!r}"
            )

    required_states = ["online_encoder", "online_actor"]
    if learner.config.sale_enabled:
        required_states.append("sale_fixed")
    missing = [key for key in required_states if key not in payload]
    if missing:
        raise ContextualCheckpointError(
            f"Stage 1 checkpoint policy state is missing: {missing}"
        )

    normalizer_states = {
        "local": payload.get("observation_local_normalizer"),
        "global": payload.get("observation_global_normalizer"),
        "critic": payload.get(
            "observation_critic_total_tat_normalizer"
        ),
    }
    for name, state in normalizer_states.items():
        _validate_frozen_normalizer_state(name, state)

    devices = (
        [learner.device]
        if torch.device(learner.device).type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        frozen_encoder = copy.deepcopy(learner.encoder).to(learner.device)
        frozen_actor = copy.deepcopy(learner.actor).to(learner.device)
        frozen_sale = (
            copy.deepcopy(learner.sale_fixed).to(learner.device)
            if learner.config.sale_enabled
            else None
        )
        frozen_encoder.load_state_dict(payload["online_encoder"], strict=True)
        frozen_actor.load_state_dict(payload["online_actor"], strict=True)
        if frozen_sale is not None:
            frozen_sale.load_state_dict(payload["sale_fixed"], strict=True)
        for module in (frozen_encoder, frozen_actor, frozen_sale):
            _freeze_policy_module(module)

    stage2_parameter_ids = {
        id(parameter)
        for module in (learner.encoder, learner.actor, learner.sale_fixed)
        if module is not None
        for parameter in module.parameters()
    }
    frozen_parameter_ids = {
        id(parameter)
        for module in (frozen_encoder, frozen_actor, frozen_sale)
        if module is not None
        for parameter in module.parameters()
    }
    if stage2_parameter_ids & frozen_parameter_ids:
        raise ContextualCheckpointError(
            "Stage 1 frozen policy shares parameters with the Stage 2 learner"
        )

    try:
        _load_normalizer(
            observation_builder.local_normalizer,
            normalizer_states["local"],
        )
        _load_normalizer(
            observation_builder.global_normalizer,
            normalizer_states["global"],
        )
        _load_normalizer(
            observation_builder.critic_normalizer,
            normalizer_states["critic"],
        )
    except Exception as error:
        raise ContextualCheckpointError(
            f"Stage 1 checkpoint normalizer load failed: {error}"
        ) from error

    applied_action_scale = float(
        payload.get("applied_action_scale", payload.get("action_scale", 0.0))
    )
    if (
        not np.isfinite(applied_action_scale)
        or applied_action_scale <= 0.0
        or applied_action_scale > 1.0
    ):
        raise ContextualCheckpointError(
            "Stage 1 checkpoint applied_action_scale must be finite and in (0, 1]"
        )
    if _checkpoint_sha256(target) != checkpoint_sha256:
        raise ContextualCheckpointError(
            "Stage 1 checkpoint changed while it was being loaded"
        )
    if initialize_fresh_learner_policy:
        _initialize_fresh_learner_policy(
            learner,
            frozen_encoder=frozen_encoder,
            frozen_actor=frozen_actor,
            frozen_sale=frozen_sale,
        )
    return FrozenContextualPolicy(
        encoder=frozen_encoder,
        actor=frozen_actor,
        sale_fixed=frozen_sale,
        action_mode=str(saved_action_mode),
        applied_action_scale=applied_action_scale,
        checkpoint_path=target.resolve(),
        checkpoint_sha256=checkpoint_sha256,
        runtime_metadata=runtime_metadata,
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
        "observation_critic_total_tat_normalizer": _normalizer_state(
            observation_builder.critic_normalizer
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
    _validate_checkpoint_reward_identity(
        payload,
        expected_reward_version=reward_builder.reward_version,
    )
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
    reject_distributed_full_resume=False,
) -> dict:
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
    if (
        reject_distributed_full_resume
        and runtime_metadata.get("distributed") is True
        and runtime_metadata.get(
            "distributed_full_resume_supported"
        ) is not True
    ):
        raise ContextualCheckpointError(
            "distributed checkpoint full-state training resume is not "
            "supported; start a fresh distributed Stage 2 run from "
            "--load-stage1-policy"
        )
    _validate_checkpoint_reward_identity(
        payload,
        expected_reward_version=reward_builder.reward_version,
    )
    _resolve_checkpoint_version(target, payload, announce_promotion=True)
    expected = {
        "topology_hash": learner.replay.topology.topology_hash,
        "mapping_hash": learner.replay.topology.mapping_hash,
        "network_config": asdict(learner.network_config),
        "action_mode": learner.config.action_mode,
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
    _load_normalizer(
        observation_builder.critic_normalizer,
        payload["observation_critic_total_tat_normalizer"],
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
