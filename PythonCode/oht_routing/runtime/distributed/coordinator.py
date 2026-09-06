"""Single-owner GPU learner and inference scheduler for simulator collectors."""

from __future__ import annotations

import copy
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
    REPLAY_SAMPLING_SNAPSHOT,
    contextual_algorithm_variant,
    load_frozen_contextual_policy,
    save_contextual_checkpoint,
)
from oht_routing.mdp.observation import (
    ContextualObservationBuilder,
    ObservationNormalizerConfig,
)
from oht_routing.mdp.reward.builder import ContextualRewardBuilder
from oht_routing.runtime.config_validation import make_reward_config
from oht_routing.runtime.stages import STAGE_TWO_STAGE1_POLICY_STEPS
from oht_routing.utils.wandb_logging import ContextualWandbLogger
from oht_routing.version import CONTEXTUAL_VERSION

from .policy import PackedPolicyInput, forward_policy_batch, merge_policy_inputs
from .types import (
    CoordinatorSnapshot,
    DistributedBootstrap,
    InferenceResult,
    POLICY_KINDS,
    POLICY_ONLINE,
    POLICY_STAGE1,
    Stage1PolicyMetadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class DistributedRuntimeError(RuntimeError):
    """Fatal central runtime failure shared by every collector."""


@dataclass(frozen=True)
class _RegistrationRequest:
    worker_id: int
    topology: object
    observation_builder: object
    cache_path: Path
    response: Future


@dataclass(frozen=True)
class _InferenceRequest:
    worker_id: int
    generation: int
    request_seq: int
    policy_kind: str
    packed: PackedPolicyInput
    attention_diagnostics: bool
    enqueued_at: float
    response: Future


@dataclass(frozen=True)
class _TransitionRequest:
    worker_id: int
    generation: int
    transition: object
    action_scale: float
    response: Future


@dataclass(frozen=True)
class _AdvanceRequest:
    worker_id: int
    generation: int
    episode_id: int
    episode_step: int
    action_scale: float
    response: Future


@dataclass(frozen=True)
class _DiagnosticsEvent:
    worker_id: int
    generation: int
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class _FailureEvent:
    worker_id: int
    error: Exception


def _normalizer_states(builder) -> dict[str, dict]:
    return {
        "local": copy.deepcopy(builder.local_normalizer.state_dict()),
        "global": copy.deepcopy(builder.global_normalizer.state_dict()),
        "critic": copy.deepcopy(builder.critic_normalizer.state_dict()),
    }


def _load_normalizer_states(builder, states: dict[str, dict]) -> None:
    builder.local_normalizer.load_state_dict(copy.deepcopy(states["local"]))
    builder.global_normalizer.load_state_dict(copy.deepcopy(states["global"]))
    builder.critic_normalizer.load_state_dict(copy.deepcopy(states["critic"]))


class CentralTrainingCoordinator:
    """Own every CUDA operation, replay mutation, update, and artifact write."""

    def __init__(self, config, *, microbatch_window_ms: float = 2.0):
        if int(config.num_sim) <= 1:
            raise ValueError("central coordinator requires num_sim > 1")
        self.config = config
        self.num_workers = int(config.num_sim)
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.microbatch_window_s = max(
            0.0, float(microbatch_window_ms) / 1000.0
        )
        self.stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._state_lock = threading.Lock()
        self._event_queue: queue.Queue = queue.Queue(
            maxsize=max(16, self.num_workers * 8)
        )
        self._failure_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._inference_queue: queue.Queue = queue.Queue(
            maxsize=max(1, self.num_workers)
        )
        self._worker_generations = [0] * self.num_workers
        self._transition_advanced_ticks: dict[int, tuple[int, int]] = {}
        self._pending_registrations: dict[int, _RegistrationRequest] = {}
        self._worker_states: dict[int, dict[str, int]] = {}
        self._canonical_topology = None
        self.observation_builder = None
        self.reward_builder = None
        self.replay = None
        self.learner = None
        self.stage1_policy = None
        self.stage1_metadata = None
        self._schedule_step = 0
        self._action_enabled_env_steps = 0
        self._aggregate_batch_id = 0
        self._latest_diagnostics: dict[str, float] = {}
        self._failure: Exception | None = None
        self._closed = False
        self._last_logged_step = -1
        self._latest_checkpoint_step = -1
        self._periodic_checkpoint_step = -1
        self.last_checkpoint_path: Path | None = None
        self.wandb_logger = ContextualWandbLogger(config)
        self.checkpoint_root = Path(
            config.checkpoint_root
            or PROJECT_ROOT / "checkpoints" / self._checkpoint_variant()
        )
        self._snapshot = CoordinatorSnapshot()

    def _checkpoint_variant(self) -> str:
        algorithm = contextual_algorithm_variant(
            self.config.sale_enabled, self.config.lap_enabled
        )
        return (
            f"ctx_td7_{CONTEXTUAL_VERSION}_dist{self.num_workers}_"
            f"{algorithm}_g{self.config.stage or 0}_"
            f"k{self.config.num_stacks}_i{self.config.stack_interval}_"
            f"r{self.config.reward_version}_{self.config.action_mode}_"
            f"{self.config.replay_sampling_mode}_"
            f"e{self.config.replay_eviction_mode}"
        )

    def _queue_request(self, target: queue.Queue, request) -> None:
        while not self.stop_event.is_set():
            try:
                target.put(request, timeout=0.25)
                self._wake_event.set()
                return
            except queue.Full:
                continue
        raise DistributedRuntimeError("central coordinator has stopped")

    def _wait(self, future: Future):
        while True:
            try:
                return future.result(timeout=0.25)
            except FutureTimeoutError:
                if self.stop_event.is_set() and not future.done():
                    raise DistributedRuntimeError(
                        "central coordinator stopped before replying"
                    ) from self._failure

    def register_worker(
        self, worker_id, topology, observation_builder, cache_path
    ) -> DistributedBootstrap:
        future = Future()
        request = _RegistrationRequest(
            int(worker_id), topology, observation_builder, Path(cache_path), future
        )
        self._queue_request(self._event_queue, request)
        return self._wait(future)

    def infer(
        self,
        worker_id,
        generation,
        request_seq,
        policy_kind,
        packed,
        attention_diagnostics=False,
    ) -> InferenceResult:
        if policy_kind not in POLICY_KINDS:
            raise ValueError(f"unknown policy kind: {policy_kind!r}")
        future = Future()
        request = _InferenceRequest(
            int(worker_id),
            int(generation),
            int(request_seq),
            str(policy_kind),
            packed,
            bool(attention_diagnostics),
            time.perf_counter(),
            future,
        )
        self._queue_request(self._inference_queue, request)
        return self._wait(future)

    def submit_transition(
        self, worker_id, generation, transition, action_scale
    ) -> CoordinatorSnapshot:
        future = Future()
        request = _TransitionRequest(
            int(worker_id),
            int(generation),
            transition,
            float(action_scale),
            future,
        )
        self._queue_request(self._event_queue, request)
        return self._wait(future)

    def advance_stage2_step(
        self,
        worker_id,
        generation,
        episode_id,
        episode_step,
        action_scale,
    ) -> CoordinatorSnapshot:
        future = Future()
        request = _AdvanceRequest(
            int(worker_id),
            int(generation),
            int(episode_id),
            int(episode_step),
            float(action_scale),
            future,
        )
        self._queue_request(self._event_queue, request)
        return self._wait(future)

    def set_worker_generation(self, worker_id, generation) -> None:
        worker = self._validate_worker_id(worker_id)
        with self._state_lock:
            self._worker_generations[worker] = int(generation)

    def submit_diagnostics(self, worker_id, generation, diagnostics) -> None:
        event = _DiagnosticsEvent(
            int(worker_id), int(generation), dict(diagnostics)
        )
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            return
        self._wake_event.set()

    def report_worker_failure(self, worker_id, error) -> None:
        event = _FailureEvent(int(worker_id), error)
        if self.stop_event.is_set():
            return
        # Fatal events bypass both bounded normal traffic and inference
        # priority. _fail() still owns checkpoint and stop publication on the
        # main coordinator thread.
        self._failure_queue.put(event)
        self._wake_event.set()

    def snapshot(self) -> CoordinatorSnapshot:
        with self._state_lock:
            value = self._snapshot
            return CoordinatorSnapshot(
                schedule_step=value.schedule_step,
                replay_size_env_steps=value.replay_size_env_steps,
                action_enabled_env_steps=value.action_enabled_env_steps,
                learner_updates=value.learner_updates,
                policy_version=value.policy_version,
                gate_open=value.gate_open,
                stopped=value.stopped,
                latest_diagnostics=dict(value.latest_diagnostics),
            )

    def _validate_worker_id(self, worker_id) -> int:
        worker = int(worker_id)
        if worker < 0 or worker >= self.num_workers:
            raise ValueError(
                f"worker_id must be in [0, {self.num_workers - 1}]"
            )
        return worker

    def _generation_matches(self, worker_id, generation) -> bool:
        worker = self._validate_worker_id(worker_id)
        with self._state_lock:
            return self._worker_generations[worker] == int(generation)

    def _publish_snapshot(self, *, stopped=None) -> CoordinatorSnapshot:
        learner_updates = int(
            getattr(self.learner, "learner_update_count", 0)
        )
        replay_size = int(
            getattr(self.replay, "size_env_steps", 0)
        )
        snapshot = CoordinatorSnapshot(
            schedule_step=int(self._schedule_step),
            replay_size_env_steps=replay_size,
            action_enabled_env_steps=int(self._action_enabled_env_steps),
            learner_updates=learner_updates,
            policy_version=learner_updates,
            gate_open=self._training_gate_open(),
            stopped=(
                self.stop_event.is_set() if stopped is None else bool(stopped)
            ),
            latest_diagnostics=dict(self._latest_diagnostics),
        )
        with self._state_lock:
            self._snapshot = snapshot
        return snapshot

    def _initialize_kernel(self, request: _RegistrationRequest) -> None:
        topology = request.topology
        self._canonical_topology = topology
        self.observation_builder = ContextualObservationBuilder(
            topology,
            cache_path=request.cache_path,
            normalizer_config=ObservationNormalizerConfig(
                freeze_after_env_steps=self.config.normalizer_freeze_steps
            ),
        )
        self.replay = ContextualStepReplayBuffer(
            topology,
            self.observation_builder,
            capacity_env_steps=self.config.replay_capacity_env_steps,
            seed=self.config.seed,
            lap_enabled=self.config.lap_enabled,
            reward_version=self.config.reward_version,
            sampling_mode=self.config.replay_sampling_mode,
            eviction_mode=self.config.replay_eviction_mode,
            state_capacity_margin=(
                self.config.replay_state_capacity_margin
            ),
            num_stacks=self.config.num_stacks,
            stack_interval=self.config.stack_interval,
        )
        learner_config = ContextualLearnerConfig(
            action_mode=self.config.action_mode,
            action_scale=self.config.action_scale,
            batch_size=self.config.batch_size,
            minimum_replay_env_steps=self.config.minimum_replay_env_steps,
            minimum_action_enabled_env_steps=(
                self.config.minimum_action_enabled_env_steps
            ),
            require_normalizer_frozen=True,
            sale_enabled=self.config.sale_enabled,
            lap_enabled=self.config.lap_enabled,
            critic_loss_mode=self.config.critic_loss_mode,
        )
        network_config = ContextualNetworkConfig(
            num_rails=len(topology.all_rail_ids),
            neighbor_count=int(topology.incoming_neighbor_ids.shape[1]),
            num_stacks=self.config.num_stacks,
            stack_interval=self.config.stack_interval,
            use_attention=self.config.use_attention,
        )
        self.learner = ContextualTD7Learner(
            self.replay,
            network_config=network_config,
            config=learner_config,
            device=self.device,
            seed=self.config.seed,
        )
        warm_start = bool(self.config.stage1_policy_warm_start)
        self.stage1_policy = load_frozen_contextual_policy(
            self.config.load_stage1_policy_path,
            self.learner,
            observation_builder=self.observation_builder,
            # Frozen prefix only: the reward version has to match solely when
            # the artifact also seeds trainable Stage 2 weights.
            expected_reward_version=(
                self.config.reward_version if warm_start else None
            ),
            initialize_fresh_learner_policy=warm_start,
        )
        self.stage1_metadata = Stage1PolicyMetadata(
            action_mode=self.stage1_policy.action_mode,
            applied_action_scale=self.stage1_policy.applied_action_scale,
            checkpoint_path=self.stage1_policy.checkpoint_path,
            checkpoint_sha256=self.stage1_policy.checkpoint_sha256,
        )
        self.reward_builder = ContextualRewardBuilder(
            topology, make_reward_config(self.config)
        )
        self._publish_snapshot()

    def _validate_topology(self, topology) -> None:
        expected = self._canonical_topology
        if expected is None:
            raise RuntimeError("canonical topology is unavailable")
        mismatches = []
        for name in ("topology_hash", "mapping_hash"):
            if getattr(topology, name) != getattr(expected, name):
                mismatches.append(name)
        for name in ("all_rail_ids", "controlled_rail_ids"):
            if not np.array_equal(
                getattr(topology, name), getattr(expected, name)
            ):
                mismatches.append(name)
        if topology.incoming_neighbor_ids.shape != (
            expected.incoming_neighbor_ids.shape
        ):
            mismatches.append("incoming_neighbor_shape")
        if mismatches:
            raise DistributedRuntimeError(
                "collector topology mismatch: " + ", ".join(mismatches)
            )

    def _handle_registration(self, request: _RegistrationRequest) -> None:
        worker = self._validate_worker_id(request.worker_id)
        if worker in self._pending_registrations:
            raise DistributedRuntimeError(
                f"worker {worker} registered more than once"
            )
        if self._canonical_topology is None:
            self._initialize_kernel(request)
        else:
            self._validate_topology(request.topology)
        self._pending_registrations[worker] = request
        print(
            "[distributed] collector registered: "
            f"worker={worker}, ready={len(self._pending_registrations)}/"
            f"{self.num_workers}",
            flush=True,
        )
        if len(self._pending_registrations) != self.num_workers:
            return
        states = _normalizer_states(self.observation_builder)
        if not all(bool(state.get("frozen")) for state in states.values()):
            raise DistributedRuntimeError(
                "distributed Stage 2 requires frozen Stage 1 normalizers"
            )
        snapshot = self._publish_snapshot()
        for registered in self._pending_registrations.values():
            _load_normalizer_states(registered.observation_builder, states)
            registered.response.set_result(DistributedBootstrap(
                normalizer_states=copy.deepcopy(states),
                stage1_policy=self.stage1_metadata,
                snapshot=snapshot,
            ))
        print(
            "[distributed] all collectors ready; central inference enabled",
            flush=True,
        )

    def _policy_modules(self, kind: str):
        if kind == POLICY_STAGE1:
            policy = self.stage1_policy
            return (
                policy.encoder,
                policy.actor,
                policy.sale_fixed,
                policy.encoder.config,
            )
        if kind == POLICY_ONLINE:
            return (
                self.learner.encoder,
                self.learner.actor,
                getattr(self.learner, "sale_fixed", None),
                self.learner.network_config,
            )
        raise ValueError(f"unknown policy kind: {kind!r}")

    def _collect_inference_batch(
        self, first: _InferenceRequest
    ) -> list[_InferenceRequest]:
        requests = [first]
        deadline = time.perf_counter() + self.microbatch_window_s
        while len(requests) < self.num_workers:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            try:
                requests.append(self._inference_queue.get(timeout=remaining))
            except queue.Empty:
                break
        return requests

    def _handle_inference_batch(
        self, requests: list[_InferenceRequest]
    ) -> None:
        if self.learner is None or len(self._pending_registrations) < self.num_workers:
            raise DistributedRuntimeError(
                "inference requested before all collectors registered"
            )
        valid = []
        for request in requests:
            if self._generation_matches(request.worker_id, request.generation):
                valid.append(request)
            elif not request.response.done():
                request.response.set_exception(DistributedRuntimeError(
                    "stale inference request after collector reconnect"
                ))
        if not valid:
            return
        service_started = time.perf_counter()
        self._aggregate_batch_id += 1
        batch_id = self._aggregate_batch_id
        schedule_step = self._schedule_step
        grouped: dict[str, list[_InferenceRequest]] = {}
        for request in valid:
            grouped.setdefault(request.policy_kind, []).append(request)
        completed: list[tuple[_InferenceRequest, np.ndarray, dict]] = []
        for kind, group in grouped.items():
            merged = merge_policy_inputs([item.packed for item in group])
            encoder, actor, sale_fixed, network_config = self._policy_modules(kind)
            actions, diagnostics = forward_policy_batch(
                merged,
                device=self.device,
                encoder=encoder,
                actor=actor,
                sale_fixed=sale_fixed,
                network_config=network_config,
                attention_diagnostics=any(
                    item.attention_diagnostics for item in group
                ),
            )
            for request, row_slice in zip(group, merged.row_slices):
                item_diagnostics = dict(diagnostics)
                item_diagnostics["runtime/tensor_conversion_ms"] = float(
                    request.packed.pack_ms
                )
                item_diagnostics["runtime/inference_queue_ms"] = (
                    service_started - request.enqueued_at
                ) * 1000.0
                item_diagnostics["runtime/inference_microbatch_size"] = float(
                    len(group)
                )
                completed.append((
                    request,
                    actions[row_slice].copy(),
                    item_diagnostics,
                ))
        policy_version = int(self.learner.learner_update_count)
        for request, action, diagnostics in completed:
            request.response.set_result(InferenceResult(
                action=action,
                diagnostics=diagnostics,
                policy_version=(
                    0 if request.policy_kind == POLICY_STAGE1 else policy_version
                ),
                aggregate_batch_id=batch_id,
                schedule_step=schedule_step,
            ))

    def _training_gate_open(self) -> bool:
        if self.replay is None or self.stop_event.is_set():
            return False
        minimum_replay = max(
            self.config.minimum_replay_env_steps,
            (
                self.config.batch_size
                if self.config.replay_sampling_mode == REPLAY_SAMPLING_SNAPSHOT
                else 0
            ),
        )
        return bool(
            self.replay.size_env_steps >= minimum_replay
            and self._action_enabled_env_steps
            >= self.config.minimum_action_enabled_env_steps
        )

    def _update_learner(self, action_scale: float) -> dict[str, float]:
        self.learner.set_applied_action_scale(float(action_scale))
        sample_started = time.perf_counter()
        batch = self.replay.sample(
            self.config.batch_size, device=self.device
        )
        replay_sample_ms = (time.perf_counter() - sample_started) * 1000.0
        learner_started = time.perf_counter()
        update = self.learner.update(batch)
        learner_update_ms = (time.perf_counter() - learner_started) * 1000.0
        diagnostics = dict(update.diagnostics)
        diagnostics.update({
            "runtime/replay_sample_ms": replay_sample_ms,
            "runtime/learner_update_ms": learner_update_ms,
        })
        return diagnostics

    def _update_for_current_tick(
        self, action_scale: float
    ) -> dict[str, float]:
        """Run the configured learner cadence once for one aggregate tick."""

        diagnostics = {}
        if (
            self._training_gate_open()
            and self._schedule_step
            % self.config.learn_every_env_steps == 0
        ):
            for _ in range(self.config.updates_per_env_step):
                diagnostics.update(self._update_learner(action_scale))
        return diagnostics

    def _complete_stage2_tick(self) -> None:
        """Advance the aggregate clock and checkpoint at its exact boundary."""

        self._schedule_step += 1
        self._maybe_checkpoint()

    def _handle_transition(self, request: _TransitionRequest) -> None:
        if not self._generation_matches(request.worker_id, request.generation):
            request.response.set_exception(DistributedRuntimeError(
                "stale transition after collector reconnect"
            ))
            return
        if int(request.transition.env_step) < STAGE_TWO_STAGE1_POLICY_STEPS:
            raise DistributedRuntimeError(
                "Stage 1 transition reached the distributed Stage 2 replay"
            )
        worker = self._validate_worker_id(request.worker_id)
        marker = self._transition_advanced_ticks.get(worker)
        if marker is not None:
            if marker[0] == request.generation:
                raise DistributedRuntimeError(
                    "collector submitted two transitions before advancing "
                    f"its Stage 2 tick: worker={worker}"
                )
            self._transition_advanced_ticks.pop(worker, None)
        learner_diagnostics = self._update_for_current_tick(
            request.action_scale
        )
        replay_started = time.perf_counter()
        self.replay.push_transition(request.transition)
        replay_push_ms = (time.perf_counter() - replay_started) * 1000.0
        self._action_enabled_env_steps += 1
        self.reward_builder.reward_steps = self._action_enabled_env_steps
        reward = request.transition.reward.total
        self._latest_diagnostics.update({
            **learner_diagnostics,
            "runtime/replay_push_ms": replay_push_ms,
            "replay/reward_mean": float(reward.mean()),
            "replay/reward_std": float(reward.std()),
            "replay/policy_action_std": float(
                request.transition.action.std()
            ),
            "replay/applied_action_std": float(
                request.transition.applied_action.std()
            ),
            "replay/done_ratio": float(request.transition.done),
        })
        self._latest_diagnostics.update(self.replay.diagnostics())
        # This transition completes on the collector's current observation
        # tick, whose zero-based step is previous env_step + 1. Advance here
        # so transitions queued by several collectors still receive distinct
        # aggregate learner clocks before any worker RPC is released.
        current_episode_step = int(request.transition.env_step) + 1
        self._transition_advanced_ticks[worker] = (
            int(request.generation), current_episode_step
        )
        self._complete_stage2_tick()
        snapshot = self._publish_snapshot()
        request.response.set_result(snapshot)

    def _handle_advance(self, request: _AdvanceRequest) -> None:
        if not self._generation_matches(request.worker_id, request.generation):
            request.response.set_exception(DistributedRuntimeError(
                "stale Stage 2 clock advance after collector reconnect"
            ))
            return
        worker = self._validate_worker_id(request.worker_id)
        marker = self._transition_advanced_ticks.get(worker)
        expected = (int(request.generation), int(request.episode_step))
        if marker == expected:
            self._transition_advanced_ticks.pop(worker, None)
        else:
            if marker is not None and marker[0] == request.generation:
                raise DistributedRuntimeError(
                    "collector transition/advance tick mismatch: "
                    f"worker={worker}, transition={marker[1]}, "
                    f"advance={request.episode_step}"
                )
            # A first Stage 2 tick (or post-reset tick) has no completed
            # transition, but the legacy runtime still performs one learner
            # cadence check. Preserve that per-environment-step contract.
            self._transition_advanced_ticks.pop(worker, None)
            self._latest_diagnostics.update(
                self._update_for_current_tick(request.action_scale)
            )
            self._complete_stage2_tick()
        self._worker_states[request.worker_id] = {
            "generation": request.generation,
            "episode_id": request.episode_id,
            "episode_step": request.episode_step,
        }
        request.response.set_result(self._publish_snapshot())

    def _handle_diagnostics(self, event: _DiagnosticsEvent) -> None:
        if not self._generation_matches(event.worker_id, event.generation):
            return
        step = int(self._schedule_step)
        if (
            step <= 0
            or step == self._last_logged_step
            or step % self.config.wandb_log_interval != 0
        ):
            return
        diagnostics = dict(event.diagnostics)
        diagnostics.update(self._latest_diagnostics)
        diagnostics.update({
            "env/step": float(step),
            "stage2/env_steps": float(step),
            "replay/size_env_steps": float(
                getattr(self.replay, "size_env_steps", 0)
            ),
            "update/learner_count": float(
                getattr(self.learner, "learner_update_count", 0)
            ),
            "learner/updates": float(
                getattr(self.learner, "learner_update_count", 0)
            ),
        })
        self.wandb_logger.log(diagnostics, step)
        self._last_logged_step = step

    def _runtime_checkpoint_metadata(self, kind: str) -> dict:
        stage1 = self.stage1_metadata
        return {
            "version": CONTEXTUAL_VERSION,
            "checkpoint_variant": self._checkpoint_variant(),
            "sale": self.config.sale_enabled,
            "lap": self.config.lap_enabled,
            "action_mode": self.config.action_mode,
            "reward_version": self.config.reward_version,
            "num_stacks": self.config.num_stacks,
            "stack_interval": self.config.stack_interval,
            "replay_eviction_mode": self.config.replay_eviction_mode,
            "dispatch_mode": self.config.dispatch_mode,
            "runtime_env_step": self._schedule_step,
            "episode_id": 0,
            "stage": self.config.stage,
            "stage2_env_steps": self._schedule_step,
            "checkpoint_schedule_step": self._schedule_step,
            "stage1_policy_prefix_steps": STAGE_TWO_STAGE1_POLICY_STEPS,
            "stage1_policy_path": str(stage1.checkpoint_path),
            "stage1_policy_sha256": stage1.checkpoint_sha256,
            "stage1_applied_action_scale": stage1.applied_action_scale,
            "stage2_policy_warm_started_from_stage1": True,
            "action_enabled_env_steps": self._action_enabled_env_steps,
            "normalizers_frozen": True,
            "checkpoint_kind": kind,
            "training_failed": self._failure is not None,
            "failure_type": (
                type(self._failure).__name__ if self._failure else None
            ),
            "failure_message": str(self._failure) if self._failure else None,
            "failure_env_step": (
                self._schedule_step if self._failure else None
            ),
            "distributed": True,
            "distributed_num_sim": self.num_workers,
            "distributed_worker_states": copy.deepcopy(self._worker_states),
            "distributed_full_resume_supported": False,
        }

    def _save_checkpoint(self, kind: str) -> Path | None:
        if self.learner is None:
            return None
        if kind == "latest":
            path = self.checkpoint_root / "latest" / "checkpoint.pt"
        elif kind == "periodic":
            path = (
                self.checkpoint_root
                / "periodic"
                / f"step_{self._schedule_step:05d}.pt"
            )
        elif kind == "crash":
            path = self.checkpoint_root / "crash" / "checkpoint.pt"
        else:
            raise ValueError(f"unknown checkpoint kind: {kind}")
        self.last_checkpoint_path = save_contextual_checkpoint(
            path,
            self.learner,
            observation_builder=self.observation_builder,
            reward_builder=self.reward_builder,
            runtime_metadata=self._runtime_checkpoint_metadata(kind),
            runtime_config=asdict(self.config),
            exploration_rng=None,
        )
        return self.last_checkpoint_path

    def _maybe_checkpoint(self) -> None:
        step = int(self._schedule_step)
        if step <= 0:
            return
        if (
            step % self.config.latest_checkpoint_interval == 0
            and step != self._latest_checkpoint_step
        ):
            self._save_checkpoint("latest")
            self._latest_checkpoint_step = step
        if (
            step % self.config.periodic_checkpoint_interval == 0
            and step != self._periodic_checkpoint_step
        ):
            self._save_checkpoint("periodic")
            self._periodic_checkpoint_step = step

    def _set_request_exception(self, request, error) -> None:
        future = getattr(request, "response", None)
        if isinstance(future, Future) and not future.done():
            future.set_exception(error)

    def _drain_pending(self, error: Exception) -> None:
        for request in self._pending_registrations.values():
            self._set_request_exception(request, error)
        for target in (self._inference_queue, self._event_queue):
            while True:
                try:
                    request = target.get_nowait()
                except queue.Empty:
                    break
                self._set_request_exception(request, error)

    def _fail(self, error: Exception) -> None:
        if self._failure is not None:
            return
        self._failure = error
        if all((
            self.learner is not None,
            self.observation_builder is not None,
            self.reward_builder is not None,
            self.stage1_metadata is not None,
        )):
            try:
                self._save_checkpoint("crash")
            except Exception as checkpoint_error:
                self._failure = DistributedRuntimeError(
                    f"distributed runtime failed: {error}; crash checkpoint "
                    f"failed: {checkpoint_error}"
                )
        self.stop_event.set()
        self._publish_snapshot(stopped=True)
        self._drain_pending(self._failure)
        self._wake_event.set()

    def _handle_event(self, event) -> None:
        if isinstance(event, _RegistrationRequest):
            self._handle_registration(event)
        elif isinstance(event, _TransitionRequest):
            self._handle_transition(event)
        elif isinstance(event, _AdvanceRequest):
            self._handle_advance(event)
        elif isinstance(event, _DiagnosticsEvent):
            self._handle_diagnostics(event)
        elif isinstance(event, _FailureEvent):
            raise DistributedRuntimeError(
                f"collector {event.worker_id} failed: {event.error}"
            ) from event.error
        else:
            raise TypeError(f"unknown coordinator event: {type(event)!r}")

    def run_once(self, *, timeout: float = 0.25) -> str:
        """Execute one deterministic scheduling quantum for tests and run()."""

        if self.stop_event.is_set():
            return "stopped"
        try:
            failure = self._failure_queue.get_nowait()
        except queue.Empty:
            failure = None
        if failure is not None:
            try:
                self._handle_event(failure)
            except Exception as error:
                self._fail(error)
            return "failure"
        try:
            first = self._inference_queue.get_nowait()
        except queue.Empty:
            first = None
        if first is not None:
            requests = self._collect_inference_batch(first)
            try:
                self._handle_inference_batch(requests)
            except Exception as error:
                for request in requests:
                    self._set_request_exception(request, error)
                self._fail(error)
            return "inference"
        try:
            event = self._event_queue.get_nowait()
        except queue.Empty:
            self._wake_event.wait(max(0.0, float(timeout)))
            self._wake_event.clear()
            return "idle"
        try:
            self._handle_event(event)
        except Exception as error:
            self._set_request_exception(event, error)
            self._fail(error)
        return "event"

    def run(self) -> None:
        """Run on the main thread until interrupted or a fatal error occurs."""

        while not self.stop_event.is_set():
            self.run_once(timeout=0.25)
        if self._failure is not None:
            raise DistributedRuntimeError(
                "central distributed runtime failed"
            ) from self._failure

    def request_stop(self) -> None:
        self.stop_event.set()
        self._publish_snapshot(stopped=True)
        self._wake_event.set()

    def close(self, *, status: str) -> None:
        if self._closed:
            return
        self._closed = True
        self.request_stop()
        self._drain_pending(
            DistributedRuntimeError("distributed runtime is closing")
        )
        if status == "interrupted":
            if self.learner is not None and self._failure is None:
                try:
                    self._save_checkpoint("latest")
                except Exception:
                    pass
            self.wandb_logger.finish_interrupted()
        elif status == "failed":
            self.wandb_logger.finish_failed(
                self._failure or "distributed runtime failed",
                self._schedule_step,
            )
        else:
            self.wandb_logger.finish_success()


__all__ = (
    "CentralTrainingCoordinator",
    "DistributedRuntimeError",
)
