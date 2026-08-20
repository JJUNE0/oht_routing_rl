"""Strict contextual learner checkpoint without replay payload."""

from __future__ import annotations

import random
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from oht_routing.mdp.action import action_version
from oht_routing.mdp.observation import OBSERVATION_VERSION
from .learner import LEARNER_VERSION
from .replay_buffer import REPLAY_VERSION
from .replay_buffer import LAP_VERSION
from .sale import SALE_VERSION
from .stacking import STACK_VERSION


CHECKPOINT_VERSION = "contextual_td7_checkpoint_v8_reward_n_only"
RESUME_REPLAY_REFILL_VERSION = (
    "contextual_resume_replay_refill_v3_optional_full_capacity"
)
RESUME_DETERMINISTIC_EPISODE_VERSION = (
    "contextual_resume_deterministic_first_episode_v1"
)
LEGACY_CHECKPOINT_VERSIONS = {
    "contextual_td7_checkpoint_v3_independent_twin_critic",
    "contextual_td7_checkpoint_v7_locked_reward_profile",
}
CRITIC_INITIALIZATION = "independent"


class ContextualCheckpointError(RuntimeError):
    pass


_ADDITIVE_RESUME_RUNTIME_METADATA = {
    "action_scale_schedule_version",
    "state_normalizer_snapshot_version",
    "warmup_episode_transition_version",
    "terminate_on_warmup_complete",
    # Legacy v7 metadata only; v8 has no reward-normalizer state.
    "reward_normalizer_snapshot_version",
    "tat_termination_policy_version",
    "tat_termination_policy",
    "tat_termination_configured_start_episode",
    "resume_replay_refill_version",
    "resume_deterministic_episode_version",
}
_RUNTIME_CONFIG_BACKED_RESUME_METADATA = {
    "tat_termination_enabled",
    "tat_termination_grace_steps",
    "early_stop_tat_threshold",
    "tat_above_threshold_patience",
    "tat_termination_inclusive",
}


def _resume_algorithm_versions_compatible(saved, runtime) -> bool:
    """Accept only known resume-lifecycle changes to an algorithm identity."""
    if not isinstance(saved, str) or not isinstance(runtime, str):
        return False
    # Checkpoints written before the lifecycle fields were added end at the
    # stack identity. All model/reward/observation contracts are still checked
    # independently below.
    if runtime.startswith(saved + "_"):
        return True

    def normalize(value):
        value = re.sub(
            r"rewardnormreuse[01]", "rewardnormreuse*", value
        )
        value = re.sub(r"_ep\d+_", "_ep*_", value)
        value = re.sub(r"normreuse[01]", "normreuse*", value)
        value = re.sub(r"detfirst[01]", "detfirst*", value)
        return re.sub(r"fullrefill[01]", "fullrefill*", value)

    return normalize(saved) == normalize(runtime)


def _resume_runtime_metadata_transition_allowed(key, saved, runtime) -> bool:
    """Allow additive metadata and intentional normalizer-resume transitions."""
    if key == "algorithm_version":
        return _resume_algorithm_versions_compatible(saved, runtime)
    if key in {
        "resume_inference_until_replay_full",
        "resume_deterministic_first_episode",
    }:
        return isinstance(runtime, bool) and (
            saved is None or isinstance(saved, bool)
        )
    if key == "resume_refill_target_env_steps":
        return runtime > 0 and (saved is None or int(saved) > 0)
    if key in _ADDITIVE_RESUME_RUNTIME_METADATA:
        return saved is None
    if key == "state_normalizer_warmup_bypass":
        return runtime is True and saved in (None, False)
    if key == "effective_warmup_steps":
        return runtime == 0 and (saved is None or int(saved) >= 0)
    if key == "reward_normalizer_reuse":
        # Compatibility with v7 lifecycle metadata; not an active v8 feature.
        return runtime is True and saved in (None, False)
    if key == "tat_termination_start_episode":
        return runtime == 1 and (saved is None or int(saved) >= 1)
    return False


def read_contextual_runtime_config(path) -> tuple[dict, bool]:
    """Read saved runtime settings before constructing the runtime.

    Current checkpoints contain the complete ContextualRuntimeConfig.
    Version 3 checkpoints are reconstructed from the metadata that existed at
    the time; settings absent from that format must retain current defaults.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = payload.get("checkpoint_version")
    if version != CHECKPOINT_VERSION and version not in LEGACY_CHECKPOINT_VERSIONS:
        raise ContextualCheckpointError(
            "Legacy or incompatible contextual checkpoint resume refused: "
            f"saved={version!r}, runtime={CHECKPOINT_VERSION!r}."
        )
    saved = payload.get("runtime_config")
    if isinstance(saved, dict) and saved:
        return dict(saved), True

    # Best-effort compatibility for checkpoints produced by the already
    # running v3 process. These formats did not persist every runtime field.
    reconstructed = {}
    metadata = dict(payload.get("runtime_metadata", {}))
    learner_config = dict(payload.get("learner_config", {}))
    direct_metadata_fields = (
        "action_mode",
        "action_scale",
        "curriculum_end_step",
        "curriculum_scale_start",
        "curriculum_scale_end",
        "curriculum_shape",
        "exploration_noise_std",
        "exploration_noise_final_std",
        "exploration_noise_anneal_steps",
        "critic_loss_mode",
    )
    for key in direct_metadata_fields:
        if key in metadata:
            reconstructed[key] = metadata[key]
    if "exploration_noise_anneal_start_step" in metadata:
        reconstructed["warmup_steps"] = metadata[
            "exploration_noise_anneal_start_step"
        ]
    for key in (
        "batch_size",
        "minimum_replay_env_steps",
        "minimum_action_enabled_env_steps",
        "sale_enabled",
        "lap_enabled",
        "critic_loss_mode",
    ):
        if key in learner_config:
            reconstructed[key] = learner_config[key]
    return reconstructed, False


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
        "checkpoint_version": CHECKPOINT_VERSION,
        "critic_initialization": CRITIC_INITIALIZATION,
        "learner_version": LEARNER_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "reward_version": reward_builder.reward_version,
        "reward_contract_version": reward_builder.reward_contract_version,
        "reward_tat_version": reward_builder.reward_tat_version,
        "reward_normalization_version": (
            reward_builder.reward_normalization_version
        ),
        "action_version": action_version(learner.config.action_mode),
        "replay_version": REPLAY_VERSION,
        "stack_version": STACK_VERSION,
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
    expected_runtime_metadata=None,
    allow_resume_metadata_upgrade=False,
    allowed_learner_config_overrides=(),
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
    saved_checkpoint_version = payload.get("checkpoint_version")
    if (
        saved_checkpoint_version != CHECKPOINT_VERSION
        and saved_checkpoint_version not in LEGACY_CHECKPOINT_VERSIONS
    ):
        raise ContextualCheckpointError(
            "Legacy or incompatible contextual checkpoint resume refused: "
            f"saved={saved_checkpoint_version!r}, "
            f"runtime={CHECKPOINT_VERSION!r}. A fresh independent twin-critic "
            "run is required."
        )
    expected = {
        "critic_initialization": CRITIC_INITIALIZATION,
        "learner_version": LEARNER_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": action_version(learner.config.action_mode),
        "replay_version": REPLAY_VERSION,
        "stack_version": STACK_VERSION,
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
    allowed_learner_config_overrides = frozenset(
        allowed_learner_config_overrides
    )
    for key, value in expected.items():
        saved = payload.get(key)
        if saved == value:
            continue
        if key == "learner_config" and allowed_learner_config_overrides:
            saved_config = dict(saved or {})
            runtime_config = dict(value)
            differing = {
                name
                for name in set(saved_config) | set(runtime_config)
                if saved_config.get(name) != runtime_config.get(name)
            }
            if differing and differing <= allowed_learner_config_overrides:
                print(
                    "[checkpoint-compat] accepted learner config override: "
                    + ",".join(sorted(differing)),
                    flush=True,
                )
                continue
        raise ContextualCheckpointError(
            f"checkpoint {key} mismatch: saved={saved!r}, "
            f"runtime={value!r}"
        )
    expected_reward = {
        "reward_version": reward_builder.reward_version,
        "reward_contract_version": reward_builder.reward_contract_version,
        "reward_tat_version": reward_builder.reward_tat_version,
        "reward_normalization_version": (
            reward_builder.reward_normalization_version
        ),
    }
    for key, value in expected_reward.items():
        if payload.get(key) != value:
            raise ContextualCheckpointError(
                f"checkpoint {key} mismatch: "
                f"saved={payload.get(key)!r}, runtime={value!r}. A fresh "
                "run is required because reward formula/normalizer state "
                "from another reward profile is incompatible."
            )
    for state_key in ("online_critic", "target_critic"):
        if _twin_state_max_abs_diff(payload[state_key]) <= 0.0:
            raise ContextualCheckpointError(
                "symmetric twin-critic checkpoint resume refused: "
                f"{state_key} has identical Q1/Q2 parameters"
            )
    upgraded_runtime_metadata = []
    saved_runtime_config = dict(payload.get("runtime_config", {}) or {})
    for key, value in dict(expected_runtime_metadata or {}).items():
        saved = payload.get("runtime_metadata", {}).get(key)
        if saved == value:
            continue
        if (
            allow_resume_metadata_upgrade
            and saved is None
            and key in _RUNTIME_CONFIG_BACKED_RESUME_METADATA
            and saved_runtime_config.get(key) == value
        ):
            upgraded_runtime_metadata.append(key)
            continue
        if allow_resume_metadata_upgrade and (
            _resume_runtime_metadata_transition_allowed(key, saved, value)
        ):
            upgraded_runtime_metadata.append(key)
            continue
        raise ContextualCheckpointError(
            f"checkpoint runtime metadata {key} mismatch: "
            f"saved={saved!r}, runtime={value!r}"
        )
    if upgraded_runtime_metadata:
        print(
            "[checkpoint-compat] accepted resume-only metadata upgrade: "
            + ",".join(upgraded_runtime_metadata),
            flush=True,
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
