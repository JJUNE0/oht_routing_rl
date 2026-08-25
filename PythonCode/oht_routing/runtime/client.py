"""Contextual action runtime with Phase 4B reward/transition alignment."""

from __future__ import annotations

import heapq
import os
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualActor,
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualObservationHistory,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
    DirectionalContextEncoder,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    contextual_algorithm_variant,
    encode_observation_stack,
    flatten_action_stack,
    flatten_state_stack,
    load_contextual_checkpoint,
    save_contextual_checkpoint,
)
from oht_routing.mdp.action import (
    EXP_RESIDUAL,
    REGION_B_RL,
    apply_controlled_action,
)
from oht_dispatching import (
    OHTDispatcher,
)
from oht_routing.mdp.observation import (
    ContextualObservationBuilder,
    ObservationNormalizerConfig,
)
from oht_routing.mdp.topology import load_cached_contextual_topology
from oht_routing.mdp.reward.builder import ContextualRewardBuilder
from oht_routing.version import CONTEXTUAL_VERSION
from oht_routing.utils.reward_diagnostic import (
    LeadingIndicatorTracker,
    RewardDiagnosticWriter,
    parse_diagnostic_windows,
)
from oht_routing.utils.environment_capture import ActorEnvironmentCapture
from oht_routing.mdp.transition import ContextualTransitionAligner
from oht_routing.utils.wandb_logging import ContextualWandbLogger
from oht_routing.runtime.console import print_entries, print_header
from oht_routing.runtime.diagnostics import ContextualRuntimeDiagnosticsMixin
from oht_routing.runtime.config import ContextualRuntimeConfig
from oht_routing.runtime.config_validation import make_reward_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_CACHE_DIR = (
    Path(__file__).resolve().parents[1] / "topology" / "cache"
)


class ContextualTrainingFailure(RuntimeError):
    """Fatal training failure requiring an explicit process restart."""


class _InferenceReplayContext:
    """Minimal replay contract required by learner construction and loading."""

    def __init__(
        self,
        topology,
        observation_builder,
        *,
        num_stacks,
        stack_interval,
        seed,
    ):
        self.topology = topology
        self.observation_builder = observation_builder
        self.num_stacks = int(num_stacks)
        self.stack_interval = int(stack_interval)
        self.rng = np.random.default_rng(seed)


class ClientAlgorithm(ContextualRuntimeDiagnosticsMixin):
    def __init__(self, config: ContextualRuntimeConfig | None = None):
        self.config = config or ContextualRuntimeConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        with torch.random.fork_rng(
            devices=[self.device] if self.device.type == "cuda" else []
        ):
            torch.manual_seed(self.config.seed)
            network_config = ContextualNetworkConfig(
                num_stacks=self.config.num_stacks,
                stack_interval=self.config.stack_interval,
            )
            self.encoder = DirectionalContextEncoder(network_config).to(self.device)
            self.actor = ContextualActor(network_config).to(self.device)
        self.encoder.eval()
        self.actor.eval()

        self.cache_path = Path(
            self.config.topology_cache_path
            or TOPOLOGY_CACHE_DIR / "contextual_topology_cache.npz"
        )
        self.audit_path = Path(
            self.config.topology_audit_path
            or TOPOLOGY_CACHE_DIR / "topology_neighbor_audit.json"
        )
        self.topology = None
        self.observation_builder = None
        self.parameterDw: dict[int, float] = {}
        self.parameterC: dict[int, float] = {}
        self.parameterPassTimes: dict[int, list[float]] = {}
        self.total_steps = 0
        self.episode_steps = 0
        self.tat_above_threshold_count = 0
        self._tat_terminal_penalty_applied = False
        self.last_observation = None
        self.observation_history = ContextualObservationHistory(
            self.config.num_stacks, self.config.stack_interval
        )
        self.last_controlled_action = None
        self.last_applied_action = None
        self.last_policy_action = None
        self.last_exploratory_action = None
        self.last_diagnostics: dict[str, float] = {}
        self._phase2_global_scales = deque(maxlen=1_000)
        self._phase2_local_scales = deque(maxlen=1_000)
        self._phase2_rail_active_scales = deque(maxlen=100_000)
        self._phase2_backlogs = deque(maxlen=1_000)
        self._phase2_good_rewards = deque(maxlen=10_000)
        self._phase2_bad_rewards = deque(maxlen=10_000)
        self._phase2_q_means = deque(maxlen=1_001)
        self.leading_indicator_tracker = LeadingIndicatorTracker()
        self.reward_builder = None
        self.transition_aligner = None
        self.episode_id = 0
        self.checkpoint_loaded = False
        self.replay_buffer = None
        self.learner = None
        self.action_enabled_env_steps = 0
        self.state_normalizer_loaded = False
        self.state_normalizer_saved = False
        self.warmup_episode_boundary_sent = False
        self.exploration_rng = np.random.default_rng(self.config.seed)
        self.wandb_logger = ContextualWandbLogger(self.config)
        self.environment_capture = (
            ActorEnvironmentCapture(
                PROJECT_ROOT / "results" / "environment_capture"
            )
            if self.config.save_data_enabled
            else None
        )
        self.training_failed = False
        self.pending_failure: Exception | None = None
        self.failure_type: str | None = None
        self.failure_message: str | None = None
        self.failure_traceback: str | None = None
        self.failure_env_step: int | None = None
        self.failure_episode_id: int | None = None
        self.last_checkpoint_path = None
        self._last_replay_summary: dict[str, float] = {}
        self._last_replay_push_ms = 0.0
        self._resume_requires_refill = False
        self._resume_deterministic_episode_active = False
        self._last_sim_time: float | None = None
        self._stale_sim_time_ticks = 0
        self._burnin_last_applied_action = None
        self._burnin_previous_applied_action = None
        self.dispatcher = OHTDispatcher(self.config.dispatch_mode)
        self.checkpoint_root = Path(
            self.config.checkpoint_root
            or PROJECT_ROOT / "checkpoints" / self.checkpoint_variant
        )
        self.rail_tat_diagnostic_path = (
            Path(self.config.rail_tat_diagnostic_path)
            if self.config.rail_tat_diagnostic_path else None
        )
        self.reward_diagnostic_writer = (
            RewardDiagnosticWriter(
                self.config.reward_diagnostic_dir,
                self.config.reward_diagnostic_windows,
            )
            if self.config.reward_diagnostic_dir
            or self.config.wandb_enabled else None
        )

    @property
    def algorithm_variant(self):
        return contextual_algorithm_variant(
            self.config.sale_enabled, self.config.lap_enabled
        )

    @property
    def runtime_variant(self):
        return (
            f"{CONTEXTUAL_VERSION}_{self.algorithm_variant}_"
            f"{self.config.action_mode}_reward_{self.config.reward_version}_"
            f"s{self.config.num_stacks}i{self.config.stack_interval}_"
            f"{self.config.replay_sampling_mode}_"
            f"curr{self.config.curriculum_scale_start:g}-"
            f"{self.config.curriculum_scale_end:g}-"
            f"{self.config.curriculum_end_step}-"
            f"{self.config.curriculum_shape}"
        )

    @property
    def checkpoint_variant(self):
        """Short, collision-resistant directory name for Windows paths."""
        return (
            f"ctx_td7_{CONTEXTUAL_VERSION}_s{int(self.config.sale_enabled)}_"
            f"l{int(self.config.lap_enabled)}_"
            f"k{self.config.num_stacks}_i{self.config.stack_interval}_"
            f"r{self.config.reward_version}_"
            f"{self.config.action_mode}_"
            f"{self.config.replay_sampling_mode}_"
            f"c{self.config.curriculum_scale_start:g}-"
            f"{self.config.curriculum_scale_end:g}-"
            f"{self.config.curriculum_end_step}"
        )

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _maybe_load_state_normalizer(self) -> None:
        source = self.config.load_state_normalizer_path
        if source is None or self.state_normalizer_loaded:
            return
        if self.observation_builder is None:
            raise ContextualTrainingFailure(
                "state normalizer load requested before observation builder "
                "initialization"
            )
        self.observation_builder.load_normalizers(
            source, require_frozen=True
        )
        self.state_normalizer_loaded = True
        print(
            "[state-normalizer] loaded compatible frozen snapshot: "
            f"path={source}, effective_warmup_steps="
            f"{self.config.effective_warmup_steps}",
            flush=True,
        )

    def _maybe_save_state_normalizer(self) -> None:
        target = self.config.save_state_normalizer_path
        if target is None or self.state_normalizer_saved:
            return
        if not self._state_normalizers_ready_for_bypass():
            return
        saved = self.observation_builder.save_normalizers(
            target, require_frozen=True
        )
        self.state_normalizer_saved = True
        print(
            "[state-normalizer] saved populated frozen snapshot: "
            f"path={saved}, env_steps={self.observation_builder.env_steps}",
            flush=True,
        )

    def _needs_learner_runtime(self) -> bool:
        return bool(
            self.config.mode == "training"
            or (
                self.config.mode == "actor_inference"
                and self.config.resume_checkpoint_path is not None
            )
        )

    def _ensure_initialized(self, pclient):
        if self.topology is not None:
            self._maybe_load_state_normalizer()
            if self.reward_builder is None:
                self.reward_builder = ContextualRewardBuilder(
                    self.topology,
                    make_reward_config(self.config),
                    completion_diagnostic_path=self.rail_tat_diagnostic_path,
                    global_step_provider=lambda: self.total_steps,
                    completion_diagnostic_max_global_step=(
                        self.config.rail_tat_diagnostic_max_step
                    ),
                    reward_diagnostic_writer=self.reward_diagnostic_writer,
                )
                self.transition_aligner = ContextualTransitionAligner(
                    self.topology, self.reward_builder
                )
                self.transition_aligner.episode_id = self.episode_id
            if self._needs_learner_runtime() and self.learner is None:
                self._initialize_learner_runtime()
            return
        self.topology = load_cached_contextual_topology(
            pclient.RAILLINE_DIC,
            expected_rail_count=int(pclient.RAILINE_COUNT),
            cache_path=self.cache_path,
        )
        self.observation_builder = ContextualObservationBuilder(
            self.topology,
            cache_path=self.cache_path,
            normalizer_config=ObservationNormalizerConfig(
                freeze_after_env_steps=self.config.normalizer_freeze_steps
            ),
        )
        self._maybe_load_state_normalizer()
        self.reward_builder = ContextualRewardBuilder(
            self.topology,
            make_reward_config(self.config),
            completion_diagnostic_path=self.rail_tat_diagnostic_path,
            global_step_provider=lambda: self.total_steps,
            completion_diagnostic_max_global_step=(
                self.config.rail_tat_diagnostic_max_step
            ),
            reward_diagnostic_writer=self.reward_diagnostic_writer,
        )
        self.transition_aligner = ContextualTransitionAligner(
            self.topology, self.reward_builder
        )
        self.transition_aligner.episode_id = self.episode_id
        if self._needs_learner_runtime():
            self._initialize_learner_runtime()

    def _initialize_learner_runtime(self):
        if self.learner is not None:
            return
        if self.config.mode == "training":
            self.replay_buffer = ContextualStepReplayBuffer(
                self.topology,
                self.observation_builder,
                capacity_env_steps=self.config.replay_capacity_env_steps,
                seed=self.config.seed,
                lap_enabled=self.config.lap_enabled,
                reward_version=self.config.reward_version,
                sampling_mode=self.config.replay_sampling_mode,
                num_stacks=self.config.num_stacks,
                stack_interval=self.config.stack_interval,
            )
            learner_replay = self.replay_buffer
        else:
            learner_replay = _InferenceReplayContext(
                self.topology,
                self.observation_builder,
                num_stacks=self.config.num_stacks,
                stack_interval=self.config.stack_interval,
                seed=self.config.seed,
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
            num_rails=len(self.topology.all_rail_ids),
            neighbor_count=int(self.topology.incoming_neighbor_ids.shape[1]),
            num_stacks=self.config.num_stacks,
            stack_interval=self.config.stack_interval,
        )
        self.learner = ContextualTD7Learner(
            learner_replay,
            network_config=network_config,
            config=learner_config,
            device=self.device,
            seed=self.config.seed,
        )
        # Runtime policy and learner online policy are the same objects.
        self.encoder = self.learner.encoder
        self.actor = self.learner.actor
        self.encoder.eval()
        self.actor.eval()
        self.transition_aligner.callback = (
            self._on_completed_transition
            if self.config.mode == "training"
            else None
        )
        if self.config.resume_checkpoint_path:
            runtime_metadata = load_contextual_checkpoint(
                self.config.resume_checkpoint_path,
                self.learner,
                observation_builder=self.observation_builder,
                reward_builder=self.reward_builder,
                exploration_rng=self.exploration_rng,
                exploration_seed=self.config.seed,
            )
            self.total_steps = int(
                runtime_metadata.get("runtime_env_step", self.total_steps)
            )
            self.episode_id = int(
                runtime_metadata.get("episode_id", self.episode_id)
            )
            self.transition_aligner.episode_id = self.episode_id
            self.checkpoint_loaded = True
            self.warmup_episode_boundary_sent = bool(
                runtime_metadata.get(
                    "warmup_episode_boundary_sent",
                    self.warmup_episode_boundary_sent,
                )
            )
            if (
                self.config.state_normalizer_warmup_bypass
                and not self._state_normalizers_ready_for_bypass()
            ):
                raise ContextualTrainingFailure(
                    "checkpoint warm-up bypass requires populated, frozen "
                    "observation state normalizers"
                )
            self._resume_requires_refill = self.config.mode == "training"
            self._resume_deterministic_episode_active = bool(
                self.config.mode == "training"
                and self.config.resume_deterministic_first_episode
            )
            self.action_enabled_env_steps = 0
            print_header("checkpoint-loaded")
            print_entries(
                (
                    ("version", CONTEXTUAL_VERSION),
                    ("mode", self.config.mode),
                    ("environment step", self.total_steps),
                    ("episode", self.episode_id),
                    ("action scale", self._action_scale()),
                    (
                        "normalizers restored",
                        self._state_normalizers_ready_for_bypass(),
                    ),
                    (
                        "learner updates enabled",
                        self.config.mode == "training",
                    ),
                    (
                        "replay collection enabled",
                        self.config.mode == "training",
                    ),
                ),
                indent=2,
            )

    def _on_completed_transition(self, transition):
        started = time.perf_counter()
        self.replay_buffer.push_transition(transition)
        self._last_replay_push_ms = (
            time.perf_counter() - started
        ) * 1000.0
        # `transition.env_step` is episode-local and restarts at zero after a
        # process resume. `total_steps` is restored from the checkpoint, so it
        # correctly identifies transitions produced by the resumed policy.
        if self.total_steps > self.config.effective_warmup_steps:
            self.action_enabled_env_steps += 1
        if (
            self._resume_requires_refill
            and self.action_enabled_env_steps
            >= self.config.minimum_action_enabled_env_steps
            and self.replay_buffer.size_env_steps
            >= self.config.resume_refill_target_env_steps
        ):
            self._resume_requires_refill = False
        self._last_replay_summary = {
            "replay/reward_mean": float(transition.reward.total.mean()),
            "replay/reward_std": float(transition.reward.total.std()),
            "replay/policy_action_std": float(transition.action.std()),
            "replay/applied_action_std": float(
                transition.applied_action.std()
            ),
            "replay/done_ratio": float(transition.done),
        }

    def _normalizers_frozen(self):
        return bool(
            getattr(self.observation_builder.local_normalizer, "frozen", False)
            and getattr(
                self.observation_builder.global_normalizer, "frozen", False
            )
            and getattr(
                self.observation_builder.critic_normalizer, "frozen", False
            )
        )

    def _state_normalizers_ready_for_bypass(self):
        return bool(
            self._normalizers_frozen()
            and int(
                getattr(
                    self.observation_builder.local_normalizer, "count", 0
                )
            ) > 0
            and int(
                getattr(
                    self.observation_builder.global_normalizer, "count", 0
                )
            ) > 0
            and int(
                getattr(
                    self.observation_builder.critic_normalizer, "count", 0
                )
            ) > 0
        )

    def _training_gate(self):
        states = {
            "gate/mode_training": self.config.mode == "training",
            "gate/action_enabled": self.config.action_enabled,
            "gate/normalizers_frozen": self._normalizers_frozen(),
            "gate/minimum_replay": (
                self.replay_buffer.size_env_steps
                >= max(
                    self.config.minimum_replay_env_steps,
                    (
                        self.config.batch_size
                        if self.config.replay_sampling_mode
                        == REPLAY_SAMPLING_SNAPSHOT
                        else 0
                    ),
                )
            ),
            "gate/minimum_action_enabled": (
                self.action_enabled_env_steps
                >= self.config.minimum_action_enabled_env_steps
            ),
            "gate/resume_refill_complete": (
                not self._resume_requires_refill
            ),
            "gate/resume_deterministic_episode_complete": (
                not self._resume_deterministic_episode_active
            ),
            "gate/not_failed": not self.training_failed,
        }
        return states, all(states.values())

    def _runtime_checkpoint_metadata(self, checkpoint_kind="latest"):
        return {
            "version": CONTEXTUAL_VERSION,
            "checkpoint_variant": self.checkpoint_variant,
            "sale": self.config.sale_enabled,
            "lap": self.config.lap_enabled,
            "action_mode": self.config.action_mode,
            "reward_version": self.reward_builder.reward_version,
            "num_stacks": self.config.num_stacks,
            "stack_interval": self.config.stack_interval,
            "dispatch_mode": self.config.dispatch_mode,
            "warmup_episode_boundary_sent": (
                self.warmup_episode_boundary_sent
            ),
            "runtime_env_step": self.total_steps,
            "episode_id": self.episode_id,
            "action_enabled_env_steps": self.action_enabled_env_steps,
            "normalizers_frozen": self._normalizers_frozen(),
            "checkpoint_kind": str(checkpoint_kind),
            "training_failed": bool(self.training_failed),
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "failure_traceback": self.failure_traceback,
            "failure_env_step": self.failure_env_step,
            "failure_episode_id": self.failure_episode_id,
        }

    def _save_runtime_checkpoint(self, kind):
        if self.learner is None:
            return None
        if kind == "latest":
            path = self.checkpoint_root / "latest" / "checkpoint.pt"
        elif kind == "periodic":
            path = (
                self.checkpoint_root / "periodic"
                / f"step_{self.total_steps:05d}.pt"
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
            exploration_rng=self.exploration_rng,
        )
        return self.last_checkpoint_path

    def _maybe_checkpoint(self, *, force_latest=False):
        if self.training_failed:
            return
        if force_latest or (
            self.total_steps
            and self.total_steps % self.config.latest_checkpoint_interval == 0
        ):
            self._save_runtime_checkpoint("latest")
        if self.total_steps and (
            self.total_steps % self.config.periodic_checkpoint_interval == 0
        ):
            self._save_runtime_checkpoint("periodic")

    def on_new_connection(self):
        if self.training_failed:
            raise ContextualTrainingFailure(
                "contextual training failed; operator restart required"
            )
        self.last_observation = None
        if hasattr(self, "observation_history"):
            self.observation_history.clear()
        self.last_controlled_action = None
        self.last_applied_action = None
        self.last_policy_action = None
        self.last_exploratory_action = None
        self._last_sim_time = None
        self._stale_sim_time_ticks = 0
        self._burnin_last_applied_action = None
        self._burnin_previous_applied_action = None
        self.tat_above_threshold_count = 0
        self._tat_terminal_penalty_applied = False
        if hasattr(self, "leading_indicator_tracker"):
            self.leading_indicator_tracker.reset_episode()
        if self.transition_aligner is not None:
            self.transition_aligner.reset(self.episode_id)
        if self.reward_builder is not None:
            self.reward_builder.reset_episode()
        observation_builder = getattr(self, "observation_builder", None)
        if observation_builder is not None:
            observation_builder.reset_episode()

    def _clear_training_temporal_state(self) -> None:
        """Idempotently remove state that could create a post-failure transition."""
        self.last_observation = None
        if hasattr(self, "observation_history"):
            self.observation_history.clear()
        self.last_controlled_action = None
        self.last_applied_action = None
        self.last_policy_action = None
        self.last_exploratory_action = None
        self._last_replay_summary = {}
        self._last_replay_push_ms = 0.0
        if self.transition_aligner is not None:
            self.transition_aligner.pending = None
            self.transition_aligner.previous_applied_action = None
            self.transition_aligner.last_completed = None

    def _sim_time_progress(self, pclient) -> tuple[float, int, bool]:
        """Track active-packet progress and detect a repeated raw snapshot."""
        value = getattr(pclient, "SimTime", None)
        if value is None:
            return 0.0, 0, False
        sim_time = float(value)
        if not np.isfinite(sim_time):
            return sim_time, self.config.max_stale_sim_time_ticks, True
        regressed = (
            self._last_sim_time is not None
            and sim_time < self._last_sim_time
        )
        if self._last_sim_time is None or sim_time > self._last_sim_time:
            self._stale_sim_time_ticks = 0
        elif sim_time == self._last_sim_time:
            self._stale_sim_time_ticks += 1
        else:
            self._stale_sim_time_ticks = self.config.max_stale_sim_time_ticks
        self._last_sim_time = sim_time
        stalled = (
            regressed
            or self._stale_sim_time_ticks
            >= self.config.max_stale_sim_time_ticks
        )
        return sim_time, self._stale_sim_time_ticks, stalled

    @staticmethod
    def _job_diagnostics(pclient) -> dict[str, float]:
        jobs = list(getattr(pclient, "JOB_DIC", {}).values())
        waiting_priorities = [
            max(0.0, float(getattr(job, "Priority", 0) or 0))
            for job in jobs
        ]
        reassign_counts = [
            max(0.0, float(getattr(job, "ReAssignCount", 0) or 0))
            for job in jobs
        ]
        return {
            "job/mean_wait_priority": (
                float(np.mean(waiting_priorities)) if waiting_priorities else 0.0
            ),
            "job/mean_reassign": (
                float(np.mean(reassign_counts)) if reassign_counts else 0.0
            ),
            "job/queued": float(
                getattr(pclient, "QueuedCommandCount", 0) or 0
            ),
        }

    def _enter_training_failure(self, error: Exception) -> None:
        if not self.training_failed:
            self.failure_type = type(error).__name__
            self.failure_message = str(error)
            self.failure_traceback = "".join(
                traceback.format_exception(
                    type(error), error, error.__traceback__
                )
            )
            self.failure_env_step = int(self.total_steps)
            self.failure_episode_id = int(self.episode_id)
        self.training_failed = True
        self.pending_failure = error
        self._clear_training_temporal_state()
        try:
            self._save_runtime_checkpoint("crash")
        except Exception as checkpoint_error:
            self.pending_failure = RuntimeError(
                f"training failed: {error}; crash checkpoint failed: "
                f"{checkpoint_error}"
            )

    def _baseline_only_result(self, pclient, baseline):
        controlled_count = len(self.topology.controlled_rail_ids)
        base_cost, congestion_cost, expected_baseline = self._cost_components(
            pclient
        )
        if not np.allclose(expected_baseline, baseline, rtol=0.0, atol=1e-12):
            raise RuntimeError("baseline fallback cost components changed")
        for physical_row, rail_id_value in enumerate(
            self.topology.all_rail_ids
        ):
            rail_id = int(rail_id_value)
            pclient.RAILLINECOST_DIC[rail_id].FRailLineCost = float(
                baseline[physical_row]
            )
        self.dispatcher.capture_cost_snapshot(
            self.topology.all_rail_ids,
            baseline,
            self.total_steps,
        )
        return apply_controlled_action(
            baseline,
            np.zeros(controlled_count, dtype=np.float32),
            self.topology,
            action_enabled=False,
            action_scale=0.0,
            action_mode=self.config.action_mode,
            base_cost=base_cost,
            congestion_cost=congestion_cost,
        )

    def Reset(self, pclient):
        if (
            getattr(self, "_resume_deterministic_episode_active", False)
            and self.episode_steps > 0
        ):
            self._resume_deterministic_episode_active = False
            print(
                "[checkpoint-resume] deterministic collection episode "
                "complete; learner updates and saved exploration resume "
                "from this episode",
                flush=True,
            )
        self.episode_id += 1
        self.episode_steps = 0
        self.tat_above_threshold_count = 0
        self._tat_terminal_penalty_applied = False
        self.last_observation = None
        if hasattr(self, "observation_history"):
            self.observation_history.clear()
        self.last_controlled_action = None
        self.last_applied_action = None
        self.last_policy_action = None
        self.last_exploratory_action = None
        self.last_diagnostics = {}
        self._last_sim_time = None
        self._stale_sim_time_ticks = 0
        self._burnin_last_applied_action = None
        self._burnin_previous_applied_action = None
        # Delay-estimator state is episode-local and affects baseline cost.
        self.parameterDw.clear()
        self.parameterPassTimes.clear()
        self.parameterC.clear()
        self.dispatcher.reset_episode()
        if hasattr(self, "leading_indicator_tracker"):
            self.leading_indicator_tracker.reset_episode()
        if self.transition_aligner is not None:
            self.transition_aligner.reset(self.episode_id)
        if self.reward_builder is not None:
            self.reward_builder.reset_episode()
        observation_builder = getattr(self, "observation_builder", None)
        if observation_builder is not None:
            observation_builder.reset_episode()

    def on_terminal(self):
        """Record terminal signal; no terminal state is available at protocol v=1."""
        if self.transition_aligner is not None:
            self.last_diagnostics["transition/terminal_without_observation"] = 1.0

    def capture_environment_tick(self, pclient):
        if (
            self.environment_capture is None
            or self.environment_capture.completed
        ):
            return
        started = time.perf_counter()
        captured = self.environment_capture.capture(pclient, self)
        self.last_diagnostics["runtime/data_capture_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        if captured and self.environment_capture.record_count == 1:
            print(
                "[environment-capture] started: "
                f"{self.environment_capture.output_dir}",
                flush=True,
            )
        if captured and self.environment_capture.completed:
            print(
                "[environment-capture] completed: "
                f"steps={self.environment_capture.record_count}, "
                f"path={self.environment_capture.output_dir}",
                flush=True,
            )

    def close_environment_capture(self, *, status="stopped"):
        if self.environment_capture is not None:
            self.environment_capture.close(status=status)

    def attach_replay_buffer(self, replay_buffer):
        """Attach Phase 5 storage without enabling any learner or updates."""
        if self.transition_aligner is None:
            raise RuntimeError("runtime must be initialized before replay attach")
        if replay_buffer.topology is not self.topology:
            raise RuntimeError("replay/runtime topology object mismatch")
        self.replay_buffer = replay_buffer
        self.transition_aligner.callback = replay_buffer.push_transition

    def compute_baseline_params(self, pclient):
        parameter_a = 0.1
        max_route_ahead = 10
        if not self.parameterDw:
            for rail_id in pclient.RAILLINECOST_DIC:
                self.parameterDw[rail_id] = 1.0
                self.parameterPassTimes[rail_id] = []
        parameter_c = {rail_id: 0.0 for rail_id in pclient.RAILLINECOST_DIC}

        for oht in pclient.OHT_DIC.values():
            limit = min(len(oht.RouteList) - 1, max_route_ahead)
            for index in range(1, max(0, limit) + 1):
                rail_id = oht.RouteList[index]
                if rail_id in parameter_c:
                    parameter_c[rail_id] += 1.0

        for oht in pclient.OHT_DIC.values():
            for pass_time in oht.PassTimes:
                rail_id = pass_time.ID
                if rail_id not in pclient.RAILLINE_DIC:
                    continue
                self.parameterPassTimes.setdefault(rail_id, [])
                oht_count = max(1, len(oht.FrontOhts.get(rail_id, [])))
                delay = (
                    pass_time.PassTime
                    - pclient.RAILLINE_DIC[rail_id].DistancePerVelocity
                ) / oht_count
                self.parameterPassTimes[rail_id].append(delay)

        for rail_id, values in self.parameterPassTimes.items():
            self.parameterDw.setdefault(rail_id, 1.0)
            for value in values:
                self.parameterDw[rail_id] += parameter_a * (
                    value - self.parameterDw[rail_id]
                )
            values.clear()
        self.parameterC = parameter_c

    def _action_scale(self) -> float:
        if self.config.action_mode == EXP_RESIDUAL:
            return float(self.config.action_scale)
        start = int(self.config.effective_warmup_steps)
        end = int(self.config.curriculum_end_step)
        scale_start = float(self.config.curriculum_scale_start)
        scale_end = float(self.config.curriculum_scale_end)
        step = int(self.total_steps)
        if step >= end:
            return scale_end
        if step <= start:
            return scale_start
        progress = (step - start) / max(1, end - start)
        if self.config.curriculum_shape == "linear":
            return float(scale_start + (scale_end - scale_start) * progress)
        return float(
            scale_start * (scale_end / scale_start) ** progress
        )

    def _exploration_noise_std(self) -> float:
        start = float(self.config.exploration_noise_std)
        if start == 0.0:
            return 0.0
        final = min(start, float(self.config.exploration_noise_final_std))
        post_warmup_step = max(
            0, self.total_steps - self.config.effective_warmup_steps
        )
        progress = np.clip(
            post_warmup_step / self.config.exploration_noise_anneal_steps,
            0.0,
            1.0,
        )
        return float(start + (final - start) * progress)

    def _burnin_active(self) -> bool:
        return bool(
            self.config.mode == "training"
            and self.episode_steps < self.config.episode_burnin_steps
        )

    def _has_trained_policy(self) -> bool:
        return bool(
            self.checkpoint_loaded
            or (
                self.learner is not None
                and int(getattr(self.learner, "learner_update_count", 0)) > 0
            )
        )

    def _advance_burnin_reward_history(self, pclient) -> None:
        """Advance reward/TAT state without staging or storing a transition."""
        if (
            self.config.episode_burnin_steps <= 0
            or self.episode_steps <= 0
            or self.episode_steps > self.config.episode_burnin_steps
            or self._burnin_last_applied_action is None
        ):
            return
        self.reward_builder.build(
            pclient,
            applied_action=self._burnin_last_applied_action,
            previous_applied_action=self._burnin_previous_applied_action,
            env_step=self.episode_steps - 1,
            episode_id=self.episode_id,
        )
        self.transition_aligner.previous_applied_action = (
            self._burnin_last_applied_action.reshape(-1, 1).copy()
        )

    @staticmethod
    def _temporal_delta(current, previous) -> tuple[float, float]:
        if previous is None:
            delta = np.zeros_like(current)
        else:
            previous = np.asarray(previous)
            if previous.shape != current.shape:
                raise RuntimeError("temporal action shape changed between ticks")
            delta = current - previous
        return float(delta.mean()), float(delta.std())

    def _cost_components(
        self, pclient
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = len(self.topology.all_rail_ids)
        base_cost = np.empty(count, dtype=np.float64)
        congestion_cost = np.empty(count, dtype=np.float64)
        for row, rail_id_value in enumerate(self.topology.all_rail_ids):
            rail_id = int(rail_id_value)
            base = float(pclient.RAILLINE_DIC[rail_id].DistancePerVelocity)
            weight = float(self.parameterDw.get(rail_id, 1.0))
            future = float(self.parameterC.get(rail_id, 0.0))
            base_cost[row] = base
            congestion_cost[row] = weight * future
        baseline = base_cost + 0.5 * congestion_cost
        if not all(
            np.isfinite(value).all()
            for value in (base_cost, congestion_cost, baseline)
        ):
            raise FloatingPointError("cost components contain NaN or Inf")
        return base_cost, congestion_cost, baseline

    def _baseline_cost(self, pclient) -> np.ndarray:
        return self._cost_components(pclient)[2]

    @staticmethod
    def _cpu_tensors(observation):
        frames = (
            tuple(observation)
            if isinstance(observation, (tuple, list))
            else (observation,)
        )
        if len(frames) == 1:
            frame = frames[0]
            return (
                torch.from_numpy(frame.center_local),
                torch.from_numpy(frame.incoming_local),
                torch.from_numpy(frame.outgoing_local),
                torch.from_numpy(frame.center_rail_index),
                torch.from_numpy(frame.incoming_rail_indices),
                torch.from_numpy(frame.outgoing_rail_indices),
                torch.from_numpy(frame.incoming_relation),
                torch.from_numpy(frame.outgoing_relation),
                torch.from_numpy(frame.global_state),
            )
        batch = int(frames[0].center_local.shape[0])
        values = tuple(
            torch.from_numpy(np.ascontiguousarray(np.stack([
                getattr(frame, name) for frame in frames
            ], axis=1)))
            for name in (
                "center_local",
                "incoming_local",
                "outgoing_local",
                "center_rail_index",
                "incoming_rail_indices",
                "outgoing_rail_indices",
                "incoming_relation",
                "outgoing_relation",
            )
        )
        global_stack = np.stack(
            [frame.global_state for frame in frames], axis=0
        )
        global_batch = np.broadcast_to(
            global_stack[None], (batch, *global_stack.shape)
        )
        return (*values, torch.from_numpy(np.ascontiguousarray(global_batch)))

    @staticmethod
    def _cpu_previous_applied_action(observation):
        frames = (
            tuple(observation)
            if isinstance(observation, (tuple, list))
            else (observation,)
        )
        if len(frames) == 1:
            return torch.from_numpy(frames[0].previous_applied_action)
        return torch.from_numpy(np.ascontiguousarray(np.stack([
            frame.previous_applied_action for frame in frames
        ], axis=1)))

    def _actor_inference(self, observation, *, attention_diagnostics=False):
        t0 = time.perf_counter()
        cpu_tensors = self._cpu_tensors(observation)
        cpu_previous_action = self._cpu_previous_applied_action(observation)
        tensor_conversion_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        device_tensors = tuple(
            tensor.to(self.device, non_blocking=False) for tensor in cpu_tensors
        )
        previous_action = cpu_previous_action.to(
            self.device, non_blocking=False
        )
        batch = device_tensors[0].shape[0]
        if device_tensors[0].ndim == 2:
            global_batch = device_tensors[-1].unsqueeze(0).expand(batch, -1)
            structured_batch = (*device_tensors[:-1], global_batch)
        else:
            structured_batch = device_tensors
        self._synchronize()
        host_to_device_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            encoded_stack = encode_observation_stack(
                self.encoder,
                structured_batch,
                return_attention=attention_diagnostics,
            )
            actor_state = flatten_state_stack(
                encoded_stack.state, self.config.num_stacks
            )
            actor_previous_action = flatten_action_stack(
                previous_action,
                num_stacks=self.config.num_stacks,
                action_dim=1,
            )
            sale_state = (
                self.learner.sale_fixed.state(structured_batch)
                if self.learner is not None
                and getattr(
                    getattr(self.learner, "config", None),
                    "sale_enabled", False
                )
                else None
            )
            actor_output = (
                self.actor(
                    actor_state,
                    sale_state,
                    previous_action=actor_previous_action,
                )
                if sale_state is not None
                else self.actor(
                    actor_state,
                    previous_action=actor_previous_action,
                )
            )
        self._synchronize()
        encoder_actor_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        controlled_action = (
            actor_output.action[:, 0].detach().to("cpu").numpy().copy()
        )
        self._synchronize()
        device_to_host_ms = (time.perf_counter() - t0) * 1000.0
        diagnostics = {
            "runtime/tensor_conversion_ms": tensor_conversion_ms,
            "runtime/host_to_device_ms": host_to_device_ms,
            "runtime/encoder_actor_ms": encoder_actor_ms,
            "runtime/device_to_host_ms": device_to_host_ms,
        }
        if attention_diagnostics:
            for direction, weights in (
                (
                    "incoming",
                    encoded_stack.flat_encoding.incoming_attention,
                ),
                (
                    "outgoing",
                    encoded_stack.flat_encoding.outgoing_attention,
                ),
            ):
                weights = weights.reshape(
                    batch, self.config.num_stacks, *weights.shape[1:]
                )[:, 0]
                probabilities = weights.clamp_min(1e-12)
                diagnostics[f"attention/{direction}_entropy"] = float(
                    (-(probabilities * probabilities.log()).sum(-1))
                    .mean().detach().cpu()
                )
                diagnostics[f"attention/{direction}_max_weight"] = float(
                    probabilities.max(-1).values.mean().detach().cpu()
                )
        return controlled_action, diagnostics

    def Algorithm(self, pclient):
        self._current_baseline = None
        try:
            return self._algorithm_impl(pclient)
        except Exception as error:
            if self.config.mode != "training" or self._current_baseline is None:
                if self.config.mode == "training":
                    self._enter_training_failure(error)
                raise
            baseline = self._current_baseline
            self._enter_training_failure(error)
            return self._baseline_only_result(pclient, baseline)

    def _update_tat_termination(self, total_tat: float) -> bool:
        if not self.config.tat_termination_enabled:
            self.tat_above_threshold_count = 0
            return False
        # The live protocol sends command=2 before the first active tick, and
        # Reset() advances episode_id from 0 to 1. Therefore episode_id=1 is
        # already the first human-numbered episode; adding one here enables an
        # episode-2 policy one episode too early. Direct unit/smoke runtimes
        # can omit that initial reset, so clamp their initial id 0 to episode 1.
        # A verified state-normalizer reuse run intentionally changes the
        # effective gate to episode 1 because its action warm-up is skipped.
        current_episode = max(1, int(self.episode_id))
        if (
            current_episode
            < self.config.effective_tat_termination_start_episode
        ):
            self.tat_above_threshold_count = 0
            return False
        if self.episode_steps < self.config.tat_termination_grace_steps:
            self.tat_above_threshold_count = 0
            return False
        above_threshold = (
            total_tat >= self.config.early_stop_tat_threshold
            if self.config.tat_termination_inclusive
            else total_tat > self.config.early_stop_tat_threshold
        )
        if above_threshold:
            self.tat_above_threshold_count += 1
        else:
            self.tat_above_threshold_count = 0
        return (
            self.tat_above_threshold_count
            >= self.config.tat_above_threshold_patience
        )

    def _update_warmup_episode_boundary(self) -> bool:
        """End the one collection episode before any policy action is applied."""
        if (
            self.config.mode != "training"
            or not self.config.terminate_on_warmup_complete
            or self.config.effective_warmup_steps <= 0
            or self.warmup_episode_boundary_sent
            or self.total_steps < self.config.effective_warmup_steps
        ):
            return False
        self.warmup_episode_boundary_sent = True
        return True

    def _algorithm_impl(self, pclient):
        total_start = time.perf_counter()
        self._ensure_initialized(pclient)
        sim_time, stale_sim_time_ticks, protocol_stalled = (
            self._sim_time_progress(pclient)
        )
        recent_tat = self.reward_builder.update_recent_completed_tat(pclient)
        # Expose the shared snapshot to capture/debug consumers without
        # recomputing lifecycle events in a second tracker.
        pclient.RecentCompletedTat300s = float(recent_tat.mean_s)
        pclient.RecentCompletedTat300sP90 = float(recent_tat.p90_s)
        pclient.RecentCompletedTat300sCount = int(recent_tat.event_count)
        pclient.RecentCompletedTat300sAvailable = bool(recent_tat.available)
        queued = float(getattr(pclient, "QueuedCommandCount", 0) or 0)
        total_tat = float(getattr(pclient, "TotalTat", 0.0) or 0.0)
        done_by_queue = queued > self.config.early_stop_queued_threshold
        done_by_tat = self._update_tat_termination(total_tat)
        done_by_warmup = self._update_warmup_episode_boundary()
        protocol_stalled = (
            self.config.mode == "training" and protocol_stalled
        )
        done = (
            done_by_queue
            or done_by_tat
            or done_by_warmup
            or protocol_stalled
        )
        termination_reason = (
            3.0 if protocol_stalled
            else 1.0 if done_by_queue
            else 2.0 if done_by_tat
            else 4.0 if done_by_warmup
            else 0.0
        )
        # Every active-data tick expects exactly one fixed-width termination
        # flag before the rail-cost payload.
        pclient.SendIsEnd(int(done))

        t0 = time.perf_counter()
        self.compute_baseline_params(pclient)
        base_cost, congestion_cost, baseline = self._cost_components(pclient)
        self._current_baseline = baseline.copy()
        state_collection_ms = (time.perf_counter() - t0) * 1000.0
        if protocol_stalled:
            error = ContextualTrainingFailure(
                "simulator active packets stopped advancing: "
                f"SimTime={sim_time}, repeated_ticks={stale_sim_time_ticks}"
            )
            self._enter_training_failure(error)
            return self._baseline_only_result(pclient, baseline)
        if self.training_failed:
            self._clear_training_temporal_state()
            return self._baseline_only_result(pclient, baseline)

        observation_calls = 0
        t0 = time.perf_counter()
        observation = self.observation_builder.build(
            pclient,
            next_10_route_oht_count=self.parameterC,
            recent_completed_tat_s=recent_tat.mean_s,
            recent_completed_tat_available=recent_tat.available,
            previous_applied_action=(
                np.zeros(
                    (len(self.topology.controlled_rail_ids), 1),
                    dtype=np.float32,
                )
                if self.last_applied_action is None
                else self.last_applied_action.reshape(-1, 1)
            ),
        )
        self._maybe_save_state_normalizer()
        observation_calls += 1
        observation_build_ms = (time.perf_counter() - t0) * 1000.0
        self.last_observation = observation

        if (
            self.config.mode == "training"
            and self.config.episode_burnin_steps > 0
            and self.episode_steps == self.config.episode_burnin_steps
        ):
            # Burn-in transitions are intentionally absent from replay. Start
            # the policy stack at the same boundary so online and replay
            # padding contracts remain identical.
            self.observation_history.clear()
        self.observation_history.append(
            observation,
            env_step=self.episode_steps,
            episode_id=self.episode_id,
        )

        burnin_active = self._burnin_active()
        has_trained_policy = self._has_trained_policy()
        self._advance_burnin_reward_history(pclient)
        completed = self.transition_aligner.complete_previous(
            observation=observation,
            pclient=pclient,
            env_step=self.episode_steps,
            episode_id=self.episode_id,
            done=done,
            dispatch=False,
        )
        if (
            completed is not None
            and done_by_tat
            and not self._tat_terminal_penalty_applied
        ):
            terminal_total = np.ascontiguousarray(
                completed.reward.total + self.config.terminal_tat_penalty,
                dtype=np.float64,
            )
            terminal_total.setflags(write=False)
            completed = replace(
                completed,
                reward=replace(
                    completed.reward,
                    total=terminal_total,
                    terminal_penalty=self.config.terminal_tat_penalty,
                ),
            )
            self.transition_aligner.last_completed = completed
            self._tat_terminal_penalty_applied = True

        timing = {
            "runtime/state_collection_ms": state_collection_ms,
            "runtime/observation_build_ms": observation_build_ms,
            "runtime/tensor_conversion_ms": 0.0,
            "runtime/host_to_device_ms": 0.0,
            "runtime/encoder_actor_ms": 0.0,
            "runtime/device_to_host_ms": 0.0,
            "runtime/send_cost_ms": 0.0,
        }
        deterministic_policy = np.zeros(
            len(self.topology.controlled_rail_ids), dtype=np.float32
        )
        use_actor = (
            self.config.mode == "actor_inference"
            or (
                self.config.mode == "training"
                and not done_by_warmup
                and (
                    (burnin_active and has_trained_policy)
                    or (
                        not burnin_active
                        and self.total_steps
                        >= self.config.effective_warmup_steps
                    )
                )
            )
        )
        if use_actor:
            deterministic_policy, inference_timing = self._actor_inference(
                self.observation_history.frames(),
                attention_diagnostics=(
                    self.config.mode == "training"
                    and self.total_steps % self.config.wandb_log_interval == 0
                ),
            )
            timing.update(inference_timing)
        controlled_action = deterministic_policy.copy()
        raw_exploration_noise = np.zeros_like(deterministic_policy)
        preclip_action = deterministic_policy.copy()
        exploration_noise_std = (
            self._exploration_noise_std()
            if (
                self.config.mode == "training"
                and not burnin_active
                and not self._resume_deterministic_episode_active
            )
            else 0.0
        )
        should_apply = (
            self.config.action_enabled
            and (
                (burnin_active and has_trained_policy)
                or (
                    not burnin_active
                    and self.total_steps
                    >= self.config.effective_warmup_steps
                )
            )
            and not self.training_failed
            and not done
        )
        if (
            self.config.mode == "training"
            and should_apply
            and not burnin_active
            and not self._resume_deterministic_episode_active
        ):
            raw_exploration_noise = self.exploration_rng.normal(
                0.0,
                exploration_noise_std,
                size=controlled_action.shape,
            )
            raw_exploration_noise = np.clip(
                raw_exploration_noise,
                -self.config.exploration_noise_clip,
                self.config.exploration_noise_clip,
            ).astype(np.float32)
            preclip_action = (
                deterministic_policy + raw_exploration_noise
            )
            controlled_action = np.clip(preclip_action, -1.0, 1.0).astype(
                np.float32
            )

        effective_noise = controlled_action - deterministic_policy
        policy_variance = float(np.var(deterministic_policy))
        noise_variance = float(np.var(raw_exploration_noise))
        positive_clip_mask = preclip_action > 1.0
        negative_clip_mask = preclip_action < -1.0
        clipped_mask = positive_clip_mask | negative_clip_mask

        policy_delta_mean, policy_delta_std = self._temporal_delta(
            deterministic_policy, self.last_policy_action
        )
        exploratory_delta_mean, exploratory_delta_std = self._temporal_delta(
            controlled_action, self.last_exploratory_action
        )
        if not np.array_equal(
            observation.controlled_rail_ids, self.topology.controlled_rail_ids
        ):
            raise RuntimeError("actor action/control rail row alignment failed")
        self.last_policy_action = deterministic_policy.copy()
        self.last_exploratory_action = controlled_action.copy()
        self.last_controlled_action = controlled_action.copy()
        current_action_scale = self._action_scale()
        t0 = time.perf_counter()
        action_result = apply_controlled_action(
            baseline,
            controlled_action,
            self.topology,
            action_enabled=should_apply,
            action_scale=current_action_scale,
            action_mode=self.config.action_mode,
            base_cost=base_cost,
            congestion_cost=congestion_cost,
        )
        applied_action = action_result.applied_controlled_action
        self.last_applied_action = applied_action.copy()
        for physical_row, rail_id_value in enumerate(self.topology.all_rail_ids):
            rail_id = int(rail_id_value)
            pclient.RAILLINECOST_DIC[rail_id].FRailLineCost = float(
                action_result.final_cost[physical_row]
            )
        self.dispatcher.capture_cost_snapshot(
            self.topology.all_rail_ids,
            action_result.final_cost,
            self.total_steps,
        )
        cost_apply_ms = (time.perf_counter() - t0) * 1000.0

        replay_sample_ms = 0.0
        learner_update_ms = 0.0
        learner_diagnostics = {}
        gate_states = {}
        gate_open = False
        if self.config.mode == "training":
            self.learner.set_applied_action_scale(current_action_scale)
            gate_states, gate_open = self._training_gate()
            try:
                if (
                    not burnin_active
                    and gate_open
                    and self.total_steps
                    % self.config.learn_every_env_steps == 0
                ):
                    for _ in range(self.config.updates_per_env_step):
                        sample_started = time.perf_counter()
                        batch = self.replay_buffer.sample(
                            self.config.batch_size, device=self.device
                        )
                        replay_sample_ms += (
                            time.perf_counter() - sample_started
                        ) * 1000.0
                        learner_started = time.perf_counter()
                        learner_result = self.learner.update(batch)
                        learner_update_ms += (
                            time.perf_counter() - learner_started
                        ) * 1000.0
                        learner_diagnostics.update(
                            learner_result.diagnostics
                        )
            except Exception as error:
                self._enter_training_failure(error)
                return self._baseline_only_result(pclient, baseline)

            # Replay commit is transactional with the learner work for this
            # tick. A failed tick never stores its completed transition.
            if completed is not None:
                self._on_completed_transition(completed)

        if not done and not burnin_active:
            self.transition_aligner.stage_current(
                observation=observation,
                controlled_action=deterministic_policy,
                applied_action=applied_action,
                baseline_cost=baseline,
                final_cost=action_result.final_cost,
                env_step=self.episode_steps,
                episode_id=self.episode_id,
            )
        else:
            self.transition_aligner.pending = None
        if burnin_active:
            self._burnin_previous_applied_action = (
                None
                if self._burnin_last_applied_action is None
                else self._burnin_last_applied_action.copy()
            )
            self._burnin_last_applied_action = applied_action.copy()
            self.transition_aligner.previous_applied_action = (
                applied_action.reshape(-1, 1).copy()
            )

        self.total_steps += 1
        self.episode_steps += 1
        self.last_diagnostics = {
            **action_result.diagnostics,
            **self.observation_builder.diagnostics(),
            **timing,
            "action/finite_ratio": float(
                np.isfinite(controlled_action).mean()
            ),
            "runtime/nonfinite_count": float(
                np.size(action_result.final_cost)
                - np.isfinite(action_result.final_cost).sum()
            ),
            "stack/num_stacks": float(self.config.num_stacks),
            "stack/interval": float(self.config.stack_interval),
            "stack/history_size": float(self.observation_history.size),
            "runtime/cost_apply_ms": cost_apply_ms,
            "runtime/observation_ms": observation_build_ms,
            "runtime/actor_inference_ms": (
                timing["runtime/encoder_actor_ms"]
            ),
            "runtime/replay_push_ms": self._last_replay_push_ms,
            "runtime/replay_sample_ms": replay_sample_ms,
            "runtime/learner_update_ms": learner_update_ms,
            "runtime/observation_build_calls_per_tick": float(observation_calls),
            **self.transition_aligner.diagnostics(completed),
            **learner_diagnostics,
            **{key: float(value) for key, value in gate_states.items()},
            "gate/open": float(gate_open),
            "action/policy_mean": float(deterministic_policy.mean()),
            "action/policy_std": float(deterministic_policy.std()),
            "action/policy_min": float(deterministic_policy.min()),
            "action/policy_max": float(deterministic_policy.max()),
            "action/policy_saturation_ratio": float(
                (np.abs(deterministic_policy) >= 0.999).mean()
            ),
            "action/cross_rail_policy_std": float(
                deterministic_policy.std()
            ),
            "action/cross_rail_applied_std": float(
                controlled_action.std()
            ),
            "action/cross_rail_noise_std": float(
                raw_exploration_noise.std()
            ),
            "action/cross_rail_noise_residual_std": float(
                effective_noise.std()
            ),
            "action/cross_rail_policy_range": float(
                np.ptp(deterministic_policy)
            ),
            "action/policy_to_noise_variance_ratio": float(
                policy_variance / (noise_variance + 1e-12)
            ),
            "action/noise_abs_mean": float(
                np.abs(raw_exploration_noise).mean()
            ),
            "action/noise_residual_abs_mean": float(
                np.abs(effective_noise).mean()
            ),
            "action/clipped_fraction": float(clipped_mask.mean()),
            "action/positive_clip_ratio": float(
                positive_clip_mask.mean()
            ),
            "action/negative_clip_ratio": float(
                negative_clip_mask.mean()
            ),
            "action/noise_suppressed_by_clip_mean": float(
                (
                    np.abs(raw_exploration_noise)
                    - np.abs(effective_noise)
                ).mean()
            ),
            "action/policy_positive_saturation_ratio": float(
                (deterministic_policy > 0.95).mean()
            ),
            "action/policy_negative_saturation_ratio": float(
                (deterministic_policy < -0.95).mean()
            ),
            "action/policy_temporal_delta_mean": policy_delta_mean,
            "action/policy_temporal_delta_std": policy_delta_std,
            "action/exploratory_mean": float(controlled_action.mean()),
            "action/exploratory_std": float(controlled_action.std()),
            "action/exploratory_temporal_delta_mean": exploratory_delta_mean,
            "action/exploratory_temporal_delta_std": exploratory_delta_std,
            "action/exploration_noise_std": exploration_noise_std,
            "action/applied_mean": float(applied_action.mean()),
            "action/applied_std": float(applied_action.std()),
            "action/applied_temporal_std": float(
                np.std(
                    applied_action
                    - (
                        self.transition_aligner.previous_applied_action.reshape(-1)
                        if self.transition_aligner.previous_applied_action
                        is not None
                        else applied_action
                    )
                )
            ),
            "replay/action_enabled_env_steps": float(
                self.action_enabled_env_steps
            ),
            "replay/size": float(
                self.replay_buffer.size_env_steps
                if self.replay_buffer is not None else 0
            ),
            "episode/step": float(self.episode_steps - 1),
            "burnin/active": float(burnin_active),
            "burnin/remaining_steps": float(
                max(
                    0,
                    self.config.episode_burnin_steps
                    - (self.episode_steps - 1),
                )
                if burnin_active else 0
            ),
            "burnin/has_trained_policy": float(has_trained_policy),
            "burnin/action_source": float(
                1 if burnin_active and has_trained_policy
                else 2 if burnin_active
                else 0
            ),
            "learner/updates": float(
                getattr(self.learner, "learner_update_count", 0)
            ),
            **self.dispatcher.diagnostics(self.total_steps),
        }
        if completed is not None:
            reward_diagnostics = self.reward_builder.diagnostics(
                completed.reward
            )
            self.last_diagnostics.update(reward_diagnostics)
            self._update_phase2_reward_diagnostics(completed.reward)
            self.last_diagnostics.update({
                # Compatibility aliases used by the per-rail/region dashboards.
                "reward/step_reward": reward_diagnostics["reward/total_mean"],
                "reward/global_norm": reward_diagnostics[
                    "reward/global/normalized"
                ],
                "reward/local_norm": reward_diagnostics[
                    "reward/local/normalized_mean"
                ],
                "reward/rail_tat": reward_diagnostics[
                    "reward/rail_tat_vector_mean"
                ],
                "reward/rail_tat_event_count": reward_diagnostics[
                    "reward/rail_tat_event_count"
                ],
                "reward/rail_tat_sum": reward_diagnostics[
                    "reward/rail_tat_sum"
                ],
                "reward/rail_tat_mean": reward_diagnostics[
                    "reward/rail_tat_mean"
                ],
                "reward/rail_tat_vector_mean": reward_diagnostics[
                    "reward/rail_tat_vector_mean"
                ],
                "reward/rail_tat_event_mean": reward_diagnostics[
                    "reward/rail_tat_event_mean"
                ],
                "reward/rail_tat_max": reward_diagnostics[
                    "reward/rail_tat_penalty_max"
                ],
                "reward/alpha": float(
                    self.reward_builder.config.global_alpha
                ),
                "reward/backlog": float(
                    (getattr(pclient, "WaitingCommandCount", 0) or 0)
                    + (getattr(pclient, "QueuedCommandCount", 0) or 0)
                ),
                "reward/smooth_mean": reward_diagnostics[
                    "reward/smooth_penalty_mean"
                ],
            })
        if self.replay_buffer is not None:
            self.last_diagnostics.update(self.replay_buffer.diagnostics())
            self.last_diagnostics.update(self._last_replay_summary)
        oht_states = [
            int(getattr(oht, "State", 0) or 0)
            for oht in getattr(pclient, "OHT_DIC", {}).values()
        ]
        self.last_diagnostics.update({
            "env/step": float(self.total_steps),
            "env/episode": float(self.episode_id),
            "env/tat": float(getattr(pclient, "TotalTat", 0.0)),
            "env/recent_completed_tat_300s_mean": float(recent_tat.mean_s),
            "env/recent_completed_tat_300s_p90": float(recent_tat.p90_s),
            "env/recent_completed_tat_300s_count": float(
                recent_tat.event_count
            ),
            "env/recent_completed_tat_300s_available": float(
                recent_tat.available
            ),
            "env/recent_completed_tat_300s_window_age": float(
                recent_tat.window_age_s
            ),
            "env/recent_completed_tat_events_added": float(
                recent_tat.events_added
            ),
            "env/recent_completed_tat_duplicate_events": float(
                recent_tat.duplicate_event_count
            ),
            "env/recent_completed_tat_missing_state5": float(
                recent_tat.missing_state5_tat_count
            ),
            "env/recent_completed_tat_ambiguous_entries": float(
                recent_tat.ambiguous_entry_count
            ),
            "env/operation_rate": float(
                getattr(pclient, "TotalOhtOperationRate", 0.0)
            ),
            "env/queued": float(getattr(pclient, "QueuedCommandCount", 0) or 0),
            "env/waiting": float(getattr(pclient, "WaitingCommandCount", 0) or 0),
            "env/completed": float(
                getattr(pclient, "CompletedCommandCount", 0) or 0
            ),
            "env/transferring": float(
                sum(
                    int(getattr(job, "State", 0) or 0) == 5
                    for job in getattr(pclient, "JOB_DIC", {}).values()
                )
            ),
            "env/sim_time": sim_time,
            "env/termination_reason": termination_reason,
            "termination/done": float(done),
            "termination/by_queue": float(done_by_queue),
            "termination/by_tat": float(done_by_tat),
            "termination/by_warmup": float(done_by_warmup),
            "warmup/episode_boundary_sent": float(
                self.warmup_episode_boundary_sent
            ),
            "protocol/stale_sim_time_ticks": float(stale_sim_time_ticks),
            "curriculum/action_scale": current_action_scale,
            "global/tat": float(getattr(pclient, "TotalTat", 0.0)),
            "global/recent_completed_tat_300s": float(recent_tat.mean_s),
            "global/recent_completed_tat_300s_available": float(
                recent_tat.available
            ),
            "global/op_rate": float(
                getattr(pclient, "TotalOhtOperationRate", 0.0)
            ),
            "global/queued": float(
                getattr(pclient, "QueuedCommandCount", 0) or 0
            ),
            "global/queued_jobs": float(
                getattr(pclient, "QueuedCommandCount", 0) or 0
            ),
            "global/waiting": float(
                getattr(pclient, "WaitingCommandCount", 0) or 0
            ),
            "global/transfer": float(oht_states.count(4)),
            "global/completed": float(
                getattr(pclient, "CompletedCommandCount", 0) or 0
            ),
            **self._job_diagnostics(pclient),
            "oht/idle_count": float(oht_states.count(0)),
            "oht/move_to_load": float(oht_states.count(2)),
            "oht/move_to_unload": float(oht_states.count(4)),
            "oht/loading": float(oht_states.count(3)),
            "oht/unloading": float(oht_states.count(5)),
        })
        predicted_oht = [
            float(
                getattr(
                    pclient.RAILLINE_DIC[int(rail_id)],
                    "PredictedOHTCount",
                    0.0,
                ) or 0.0
            )
            for rail_id in self.topology.controlled_rail_ids
            if int(rail_id) in getattr(pclient, "RAILLINE_DIC", {})
        ]
        self.last_diagnostics.update(
            self.leading_indicator_tracker.update(
                pclient,
                predicted_oht=predicted_oht,
                route_ratio_diagnostics=(
                    self.reward_builder.route_ratio_diagnostics()
                ),
                idle_reserve_target=(
                    self.reward_builder.config.idle_reserve_target
                ),
                idle_reserve_scale=(
                    self.reward_builder.config.idle_reserve_scale
                ),
            )
        )
        if completed is not None and self.reward_diagnostic_writer is not None:
            self._write_reward_step_diagnostic(pclient, completed)
            cycle_summary = self.reward_diagnostic_writer.cycle_summary(
                self.total_steps - 1
            )
            cycle_summary["reward/rail/mode"] = (
                self.reward_builder.config.rail_reward_mode
            )
            cycle_summary["reward/rail/free_flow_neutral_ratio"] = (
                self.reward_builder.config.rail_free_flow_neutral_ratio
            )
            self.last_diagnostics.update(cycle_summary)
        checkpoint_started = time.perf_counter()
        if self.config.mode == "training" and not self.training_failed:
            self._maybe_checkpoint(force_latest=done_by_warmup)
        checkpoint_ms = (time.perf_counter() - checkpoint_started) * 1000.0
        total_algorithm_ms = (time.perf_counter() - total_start) * 1000.0
        self.last_diagnostics.update({
            "runtime/checkpoint_ms": checkpoint_ms,
            "runtime/total_algorithm_ms": total_algorithm_ms,
            "runtime/total_ms": total_algorithm_ms,
        })
        return action_result


    def AlgorithmAfter(self, pclient):
        if self.pending_failure is not None:
            error = self.pending_failure
            self.pending_failure = None
            self.wandb_logger.finish_failed(error, self.failure_env_step)
            raise ContextualTrainingFailure(
                "contextual training stopped after baseline fallback; "
                "operator restart required"
            ) from error

    def UpdateDatas(self, pclient):
        pass

    def UpdateOHTRoute(self, pclient, ohtId, curLine, destination, route_list):
        pclient.OHT_DIC[ohtId].RouteList = route_list

    def DijkStraRoute(self, pclient, curLine, destination):
        distances = defaultdict(lambda: float("inf"))
        parent = defaultdict(int)
        distances[curLine] = 0.0
        parent[curLine] = -1
        heap = [(0.0, curLine)]
        while heap:
            weight, current = heapq.heappop(heap)
            for next_id in pclient.RAILLINE_DIC[current].DivergingLineIDList:
                new_weight = pclient.RAILLINE_DIC[next_id].Distance + weight
                if new_weight < distances[next_id] and new_weight < distances[destination]:
                    distances[next_id] = new_weight
                    parent[next_id] = current
                    if next_id != destination:
                        heapq.heappush(heap, (new_weight, next_id))
        next_id = destination
        route = deque([destination])
        while next_id != -1:
            next_id = parent[next_id]
            if next_id != -1:
                route.appendleft(next_id)
        return route

    def ReRoute(self, pclient, oht_list, from_nodes, to_nodes):
        return {
            oht_id: self.DijkStraRoute(pclient, from_nodes[index], to_nodes[index])
            for index, oht_id in enumerate(oht_list)
        }

    def Assign(self, pclient, job_list):
        assignments = self.dispatcher.assign(pclient.OHT_DIC, job_list)
        self.last_diagnostics.update(
            self.dispatcher.diagnostics(self.total_steps)
        )
        return assignments
