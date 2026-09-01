"""Thread-safe value contracts for the multi-simulator runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


POLICY_STAGE1 = "stage1"
POLICY_ONLINE = "online"
POLICY_KINDS = (POLICY_STAGE1, POLICY_ONLINE)


@dataclass(frozen=True)
class Stage1PolicyMetadata:
    """Policy fields collectors need without owning the frozen CUDA modules."""

    action_mode: str
    applied_action_scale: float
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class CoordinatorSnapshot:
    """Lock-protected central state safe for collector-side diagnostics."""

    schedule_step: int = 0
    replay_size_env_steps: int = 0
    action_enabled_env_steps: int = 0
    learner_updates: int = 0
    policy_version: int = 0
    gate_open: bool = False
    stopped: bool = False
    latest_diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributedBootstrap:
    """Canonical topology/policy state returned after all workers register."""

    normalizer_states: dict[str, dict]
    stage1_policy: Stage1PolicyMetadata
    snapshot: CoordinatorSnapshot


@dataclass(frozen=True)
class InferenceResult:
    """One deterministic policy response returned to a collector."""

    action: object
    diagnostics: dict[str, float]
    policy_version: int
    aggregate_batch_id: int
    schedule_step: int


def namespace_episode_id(worker_id: int, episode_id: int) -> int:
    """Pack a worker/local episode pair into a non-negative signed int64."""

    worker = int(worker_id)
    episode = int(episode_id)
    if worker < 0 or worker >= (1 << 15):
        raise ValueError("worker_id must be in [0, 32767]")
    if episode < 0 or episode >= (1 << 48):
        raise ValueError("episode_id must be in [0, 2**48)")
    return (worker << 48) | episode


__all__ = (
    "CoordinatorSnapshot",
    "DistributedBootstrap",
    "InferenceResult",
    "POLICY_KINDS",
    "POLICY_ONLINE",
    "POLICY_STAGE1",
    "Stage1PolicyMetadata",
    "namespace_episode_id",
)
