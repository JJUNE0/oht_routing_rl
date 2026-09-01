"""Collector-side adapter for the single-owner distributed TD7 runtime.

The simulator-facing algorithm remains deliberately local: observations,
reward temporal state, exploration, action application, and transition
alignment still use :class:`ClientAlgorithm`.  Only deterministic policy
inference, replay ownership, learner updates, checkpointing, and W&B ownership
are delegated to the coordinator supplied by the launcher.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Protocol

import numpy as np

from oht_routing.runtime.client import (
    ClientAlgorithm,
    ContextualTrainingFailure,
)
from oht_routing.runtime.config import ContextualRuntimeConfig

from .policy import pack_policy_input
from .types import (
    CoordinatorSnapshot,
    DistributedBootstrap,
    InferenceResult,
    POLICY_ONLINE,
    POLICY_STAGE1,
    Stage1PolicyMetadata,
    namespace_episode_id,
)


class DistributedCoordinator(Protocol):
    """Duck-typed collector contract implemented by the central owner."""

    def register_worker(
        self, worker_id, topology, observation_builder, cache_path
    ) -> DistributedBootstrap: ...

    def infer(
        self,
        worker_id,
        generation,
        request_seq,
        policy_kind,
        packed,
        attention_diagnostics=False,
    ) -> InferenceResult: ...

    def submit_transition(
        self, worker_id, generation, transition, action_scale
    ) -> CoordinatorSnapshot: ...

    def advance_stage2_step(
        self,
        worker_id,
        generation,
        episode_id,
        episode_step,
        action_scale,
    ) -> CoordinatorSnapshot: ...

    def snapshot(self) -> CoordinatorSnapshot: ...

    def set_worker_generation(self, worker_id, generation) -> None: ...

    def submit_diagnostics(
        self, worker_id, generation, diagnostics
    ) -> None: ...

    def report_worker_failure(self, worker_id, error) -> None: ...


def distributed_worker_config(
    config: ContextualRuntimeConfig, worker_id: int
) -> ContextualRuntimeConfig:
    """Return an isolated CPU-only collector configuration.

    The central coordinator is the only CUDA, checkpoint, diagnostics-file,
    and W&B owner.  A deterministic worker offset keeps exploration streams
    independent without changing the launch-level seed contract.
    """

    worker = int(worker_id)
    # Reuse the namespace validator so configuration and replay identity have
    # exactly the same worker-ID range.
    namespace_episode_id(worker, 0)
    seed = int(config.seed) + worker
    if seed < 0:
        raise ValueError("distributed worker seed must be non-negative")
    return replace(
        config,
        device="cpu",
        seed=seed,
        wandb_enabled=False,
        save_data_enabled=False,
        reward_diagnostic_dir=None,
        rail_tat_diagnostic_path=None,
        rail_tat_diagnostic_max_step=0,
    )


class RemoteLearnerProxy:
    """The tiny learner surface still referenced by ``ClientAlgorithm``."""

    def __init__(self, owner: "DistributedCollectorClient"):
        self._owner = owner
        self.applied_action_scale = float(owner.config.action_scale)

    @property
    def learner_update_count(self) -> int:
        return int(self._owner.coordinator_snapshot.learner_updates)

    def set_applied_action_scale(self, scale: float) -> None:
        value = float(scale)
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError("applied action scale must be in (0, 1]")
        # submit_transition carries this tick's canonical scale to the owner.
        # Keeping a local copy satisfies the runtime's learner-shaped contract
        # without mutating central state from four separate listener threads.
        self.applied_action_scale = value

    def update(self, _batch=None):
        raise RuntimeError("collector-local learner updates are disabled")


class RemoteReplayProxy:
    """Forward complete transitions while exposing read-only central status."""

    def __init__(self, owner: "DistributedCollectorClient"):
        self._owner = owner
        self.topology = owner.topology

    @property
    def size_env_steps(self) -> int:
        return int(self._owner.coordinator_snapshot.replay_size_env_steps)

    def push_transition(self, transition):
        global_episode = namespace_episode_id(
            self._owner.worker_id, transition.episode_id
        )
        reward = replace(transition.reward, episode_id=global_episode)
        namespaced = replace(
            transition,
            episode_id=global_episode,
            reward=reward,
        )
        snapshot = self._owner.coordinator.submit_transition(
            self._owner.worker_id,
            self._owner.generation,
            namespaced,
            self._owner._action_scale(),
        )
        self._owner._adopt_coordinator_snapshot(
            snapshot, update_schedule=False
        )
        return snapshot

    def sample(self, *_args, **_kwargs):
        raise RuntimeError("collector-local replay sampling is disabled")

    def diagnostics(self) -> dict[str, float]:
        return self._owner._distributed_diagnostics()


class DistributedCollectorClient(ClientAlgorithm):
    """One simulator collector backed by a shared central GPU learner."""

    def __init__(
        self,
        config: ContextualRuntimeConfig,
        worker_id: int,
        coordinator: DistributedCoordinator,
    ):
        self.worker_id = int(worker_id)
        namespace_episode_id(self.worker_id, 0)
        self.coordinator = coordinator
        self._generation = 0
        self._connection_generation_started = False
        self._request_seq = 0
        self._distributed_registered = False
        self._coordinator_snapshot = CoordinatorSnapshot()
        self._distributed_schedule_step = 0
        self._reported_failure = False
        super().__init__(distributed_worker_config(config, self.worker_id))

    @property
    def generation(self) -> int:
        return int(self._generation)

    @property
    def coordinator_snapshot(self) -> CoordinatorSnapshot:
        return self._coordinator_snapshot

    def _adopt_coordinator_snapshot(
        self,
        snapshot: CoordinatorSnapshot,
        *,
        update_schedule: bool,
    ) -> None:
        if not isinstance(snapshot, CoordinatorSnapshot):
            raise TypeError("coordinator must return CoordinatorSnapshot")
        if int(snapshot.schedule_step) < 0:
            raise ValueError("coordinator schedule_step must be non-negative")
        self._coordinator_snapshot = snapshot
        if update_schedule:
            self._distributed_schedule_step = int(snapshot.schedule_step)
        if snapshot.stopped:
            raise ContextualTrainingFailure("central coordinator has stopped")

    def _load_distributed_bootstrap(
        self, bootstrap: DistributedBootstrap
    ) -> None:
        if not isinstance(bootstrap, DistributedBootstrap):
            raise TypeError("register_worker must return DistributedBootstrap")
        expected = {
            "local": self.observation_builder.local_normalizer,
            "global": self.observation_builder.global_normalizer,
            "critic": self.observation_builder.critic_normalizer,
        }
        missing = sorted(set(expected).difference(bootstrap.normalizer_states))
        if missing:
            raise ContextualTrainingFailure(
                f"distributed bootstrap normalizers are missing: {missing}"
            )
        try:
            for name, normalizer in expected.items():
                normalizer.load_state_dict(bootstrap.normalizer_states[name])
        except Exception as error:
            raise ContextualTrainingFailure(
                f"distributed normalizer bootstrap failed: {error}"
            ) from error
        if not self._state_normalizers_ready_for_bypass():
            raise ContextualTrainingFailure(
                "distributed Stage 2 requires populated, frozen Stage 1 "
                "normalizers"
            )

        stage1 = bootstrap.stage1_policy
        if not isinstance(stage1, Stage1PolicyMetadata):
            raise TypeError(
                "distributed bootstrap must contain Stage1PolicyMetadata"
            )
        if stage1.action_mode != self.config.action_mode:
            raise ContextualTrainingFailure(
                "distributed Stage 1 action_mode mismatch: "
                f"saved={stage1.action_mode!r}, "
                f"runtime={self.config.action_mode!r}"
            )
        scale = float(stage1.applied_action_scale)
        if not math.isfinite(scale) or scale <= 0.0 or scale > 1.0:
            raise ContextualTrainingFailure(
                "distributed Stage 1 applied_action_scale must be in (0, 1]"
            )
        if not str(stage1.checkpoint_sha256):
            raise ContextualTrainingFailure(
                "distributed Stage 1 checkpoint SHA-256 is missing"
            )
        self.stage1_policy = stage1
        self._adopt_coordinator_snapshot(
            bootstrap.snapshot, update_schedule=True
        )

    def _initialize_learner_runtime(self):
        if self.learner is not None:
            return
        bootstrap = self.coordinator.register_worker(
            self.worker_id,
            self.topology,
            self.observation_builder,
            self.cache_path,
        )
        self._load_distributed_bootstrap(bootstrap)
        self.learner = RemoteLearnerProxy(self)
        self.replay_buffer = RemoteReplayProxy(self)
        # The base training path completes transitions with dispatch=False and
        # calls _on_completed_transition exactly once after its (closed) local
        # learner gate.  Do not install a second callback here.
        self.transition_aligner.callback = None
        self._distributed_registered = True
        self.coordinator.set_worker_generation(
            self.worker_id, self.generation
        )

    def _distributed_inference(
        self,
        observation,
        *,
        policy_kind: str,
        attention_diagnostics: bool,
    ):
        packed = pack_policy_input(observation)
        request_seq = self._request_seq
        self._request_seq += 1
        result = self.coordinator.infer(
            self.worker_id,
            self.generation,
            request_seq,
            policy_kind,
            packed,
            attention_diagnostics=attention_diagnostics,
        )
        if not isinstance(result, InferenceResult):
            raise TypeError("coordinator infer must return InferenceResult")
        action = np.asarray(result.action, dtype=np.float32).reshape(-1)
        expected_rows = len(self.topology.controlled_rail_ids)
        if action.shape != (expected_rows,):
            raise ContextualTrainingFailure(
                "distributed actor action shape mismatch: "
                f"actual={action.shape}, expected=({expected_rows},)"
            )
        if not np.isfinite(action).all():
            raise FloatingPointError(
                "distributed actor action contains NaN or Inf"
            )
        if (np.abs(action) > 1.0 + 1e-6).any():
            raise ContextualTrainingFailure(
                "distributed actor action must be in [-1, 1]"
            )
        diagnostics = dict(result.diagnostics)
        diagnostics.setdefault("runtime/tensor_conversion_ms", packed.pack_ms)
        diagnostics["runtime/policy_pack_ms"] = float(packed.pack_ms)
        diagnostics["distributed/worker_id"] = float(self.worker_id)
        diagnostics["distributed/generation"] = float(self.generation)
        diagnostics["distributed/request_seq"] = float(request_seq)
        diagnostics["distributed/policy_version"] = float(
            result.policy_version
        )
        diagnostics["distributed/aggregate_batch_id"] = float(
            result.aggregate_batch_id
        )
        diagnostics["distributed/inference_schedule_step"] = float(
            result.schedule_step
        )
        return np.ascontiguousarray(action.copy()), diagnostics

    def _actor_inference(self, observation, *, attention_diagnostics=False):
        return self._distributed_inference(
            observation,
            policy_kind=POLICY_ONLINE,
            attention_diagnostics=attention_diagnostics,
        )

    def _stage1_actor_inference(self, observation):
        if self.stage1_policy is None:
            raise ContextualTrainingFailure(
                "Stage 2 reached its Stage 1 prefix without metadata"
            )
        return self._distributed_inference(
            observation,
            policy_kind=POLICY_STAGE1,
            attention_diagnostics=False,
        )

    def _training_gate(self):
        central = self.coordinator_snapshot
        states = {
            "gate/mode_training": self.config.mode == "training",
            "gate/action_enabled": self.config.action_enabled,
            "gate/normalizers_frozen": self._normalizers_frozen(),
            "gate/minimum_replay": (
                central.replay_size_env_steps
                >= self.config.minimum_replay_env_steps
            ),
            "gate/minimum_action_enabled": (
                central.action_enabled_env_steps
                >= self.config.minimum_action_enabled_env_steps
            ),
            "gate/resume_refill_complete": True,
            "gate/resume_deterministic_episode_complete": True,
            "gate/resume_warmstart_complete": True,
            "gate/stage2_policy_active": not self._stage1_prefix_active(),
            "gate/not_failed": not self.training_failed,
            "gate/central_open": central.gate_open,
            "gate/local_updates_disabled": True,
        }
        # CentralCoordinator owns the only learner update loop.  Returning a
        # closed local gate is an invariant, not a reflection of central.gate.
        return states, False

    def _stage2_schedule_step(self) -> int:
        return int(self._distributed_schedule_step)

    def _save_runtime_checkpoint(self, _kind):
        return None

    def _maybe_checkpoint(
        self,
        *,
        force_latest=False,
        schedule_advanced=True,
    ):
        del force_latest, schedule_advanced
        return None

    def on_new_connection(self):
        super().on_new_connection()
        if self._connection_generation_started:
            self._generation += 1
        else:
            self._connection_generation_started = True
        self._request_seq = 0
        if self._distributed_registered:
            self.coordinator.set_worker_generation(
                self.worker_id, self.generation
            )
            self._adopt_coordinator_snapshot(
                self.coordinator.snapshot(), update_schedule=True
            )

    def Reset(self, pclient):
        super().Reset(pclient)
        if self._distributed_registered:
            # Simulator command=2 stays within the same TCP generation.  Its
            # new local episode ID is namespaced when the first transition is
            # submitted; no central inference cancellation is necessary.
            self._adopt_coordinator_snapshot(
                self.coordinator.snapshot(), update_schedule=True
            )

    def Algorithm(self, pclient):
        if self._distributed_registered:
            try:
                self._adopt_coordinator_snapshot(
                    self.coordinator.snapshot(), update_schedule=True
                )
            except Exception as error:
                self._enter_training_failure(error)
                raise
        stage2_active = not self._stage1_prefix_active()
        result = super().Algorithm(pclient)
        if stage2_active and not self.training_failed:
            try:
                snapshot = self.coordinator.advance_stage2_step(
                    self.worker_id,
                    self.generation,
                    self.episode_id,
                    self.episode_steps - 1,
                    self._action_scale(),
                )
                self._adopt_coordinator_snapshot(
                    snapshot, update_schedule=True
                )
            except Exception as error:
                # The action has already been applied to the local rail-cost
                # payload. Restore the same baseline fallback used by the
                # single-runtime learner failure path, then let AlgorithmAfter
                # stop only after the protocol sends that complete payload.
                self._enter_training_failure(error)
                if self._current_baseline is not None:
                    result = self._baseline_only_result(
                        pclient, self._current_baseline
                    )
        return result

    def _distributed_diagnostics(self) -> dict[str, float]:
        snapshot = self.coordinator_snapshot
        diagnostics = dict(snapshot.latest_diagnostics)
        diagnostics.update({
            "distributed/worker_id": float(self.worker_id),
            "distributed/generation": float(self.generation),
            "distributed/schedule_step": float(snapshot.schedule_step),
            "distributed/policy_version": float(snapshot.policy_version),
            "distributed/central_gate_open": float(snapshot.gate_open),
            "distributed/central_stopped": float(snapshot.stopped),
            "replay/size_env_steps": float(
                snapshot.replay_size_env_steps
            ),
            "replay/action_enabled_env_steps": float(
                snapshot.action_enabled_env_steps
            ),
            "update/learner_count": float(snapshot.learner_updates),
            "learner/updates": float(snapshot.learner_updates),
        })
        return diagnostics

    def log_wandb_tick(self):
        if (
            self.config.mode in {"training", "actor_inference"}
            and not self.training_failed
        ):
            diagnostics = dict(self.last_diagnostics)
            diagnostics.update(self._distributed_diagnostics())
            try:
                self.coordinator.submit_diagnostics(
                    self.worker_id,
                    self.generation,
                    diagnostics,
                )
            except Exception as error:
                self._enter_training_failure(error)

    def _enter_training_failure(self, error: Exception) -> None:
        first_failure = not self.training_failed
        super()._enter_training_failure(error)
        if first_failure and not self._reported_failure:
            self._reported_failure = True
            try:
                self.coordinator.report_worker_failure(
                    self.worker_id, error
                )
            except Exception:
                # Reporting must never replace the simulator/learner failure
                # already latched by the base runtime.
                pass


__all__ = (
    "DistributedCollectorClient",
    "DistributedCoordinator",
    "RemoteLearnerProxy",
    "RemoteReplayProxy",
    "distributed_worker_config",
)
