"""Contextual action runtime with Phase 4B reward/transition alignment."""

from __future__ import annotations

import heapq
import os
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from cocel_rl.algorithms.contextual_td7 import (
    ALGORITHM_VERSION,
    ContextualActor,
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
    DirectionalContextEncoder,
    REPLAY_SAMPLING_MODES,
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    REPLAY_SAMPLING_VERSION,
    contextual_algorithm_variant,
    load_contextual_checkpoint,
    save_contextual_checkpoint,
)
from contextual_action import (
    ACTION_MODES,
    EXP_RESIDUAL,
    EXPLORATION_SCHEDULE_VERSION,
    REGION_B_RL,
    action_version,
    apply_controlled_action,
)
from contextual_observation import (
    ContextualObservationBuilder,
    ObservationNormalizerConfig,
)
from contextual_topology import load_cached_contextual_topology
from contextual_reward import ContextualRewardBuilder, ContextualRewardConfig
from contextual_transition import ContextualTransitionAligner
from contextual_wandb import ContextualWandbLogger


class ContextualTrainingFailure(RuntimeError):
    """Fatal training failure requiring an explicit process restart."""


@dataclass(frozen=True)
class ContextualRuntimeConfig:
    mode: str = "baseline_only"
    action_enabled: bool = False
    action_mode: str = REGION_B_RL
    action_scale: float = 0.05
    curriculum_end_step: int = 20_000
    curriculum_scale_start: float = 0.05
    curriculum_scale_end: float = 1.0
    curriculum_shape: str = "geometric"
    smooth_b_rl_weight: float = 0.05
    smooth_exp_residual_weight: float = 0.5
    warmup_steps: int = 10_000
    episode_burnin_steps: int = 2_000
    normalizer_freeze_steps: int = 10_000
    tat_confidence_n0: float = 50.0
    tat_confidence_ramp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    topology_cache_path: str | None = None
    topology_audit_path: str | None = None
    exploration_noise_std: float = 0.10
    exploration_noise_final_std: float = 0.02
    exploration_noise_anneal_steps: int = 100_000
    exploration_noise_clip: float = 0.20
    replay_capacity_env_steps: int = 10_000
    replay_sampling_mode: str = REPLAY_SAMPLING_RAIL
    batch_size: int = 1_024
    minimum_replay_env_steps: int = 100
    minimum_action_enabled_env_steps: int = 100
    updates_per_env_step: int = 1
    learn_every_env_steps: int = 1
    latest_checkpoint_interval: int = 1_000
    periodic_checkpoint_interval: int = 5_000
    checkpoint_root: str | None = None
    resume_checkpoint_path: str | None = None
    rail_tat_diagnostic_path: str | None = None
    rail_tat_diagnostic_max_step: int = 1_000
    wandb_enabled: bool = False
    wandb_log_interval: int = 10
    sale_enabled: bool = True
    lap_enabled: bool = True
    critic_loss_mode: str = "auto"
    early_stop_queued_threshold: float = 500.0
    early_stop_tat_threshold: float = 500.0
    early_stop_min_episode_steps: int = 100
    max_stale_sim_time_ticks: int = 5

    def __post_init__(self):
        if self.mode not in {"baseline_only", "actor_inference", "training"}:
            raise ValueError(
                "mode must be baseline_only, actor_inference, or training"
            )
        if self.mode == "training" and not self.action_enabled:
            raise ValueError("training mode requires explicit action_enabled")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        if self.replay_sampling_mode not in REPLAY_SAMPLING_MODES:
            raise ValueError(
                f"replay_sampling_mode must be one of {REPLAY_SAMPLING_MODES}"
            )
        if (
            self.replay_sampling_mode
            in {REPLAY_SAMPLING_SNAPSHOT, REPLAY_SAMPLING_RANDOM_RAIL}
            and self.lap_enabled
        ):
            raise ValueError(
                f"{self.replay_sampling_mode} replay sampling requires "
                "lap_enabled=False"
            )
        if (
            not np.isfinite(self.action_scale)
            or self.action_scale < 0.0
            or self.action_scale > 1.0
        ):
            raise ValueError("action_scale must be finite and in [0, 1]")
        if (
            isinstance(self.episode_burnin_steps, bool)
            or not isinstance(self.episode_burnin_steps, Integral)
        ):
            raise ValueError("episode_burnin_steps must be an integer")
        if (
            self.warmup_steps < 0
            or self.episode_burnin_steps < 0
            or self.normalizer_freeze_steps < 0
        ):
            raise ValueError("warmup/burn-in/freeze steps must be non-negative")
        if self.mode == "training" and self.action_scale <= 0:
            raise ValueError("training mode requires positive action_scale")
        if (
            self.curriculum_end_step <= self.warmup_steps
            or not 0 < self.curriculum_scale_start <= 1
            or not 0 < self.curriculum_scale_end <= 1
            or self.curriculum_scale_start > self.curriculum_scale_end
            or self.curriculum_shape not in {"geometric", "linear"}
        ):
            raise ValueError("invalid action curriculum configuration")
        if (
            self.smooth_b_rl_weight < 0
            or self.smooth_exp_residual_weight < 0
        ):
            raise ValueError("smooth penalty weights must be non-negative")
        if (
            not np.isfinite(self.exploration_noise_std)
            or self.exploration_noise_std < 0
            or not np.isfinite(self.exploration_noise_final_std)
            or self.exploration_noise_final_std < 0
        ):
            raise ValueError("exploration noise stds must be finite and non-negative")
        positive = (
            self.replay_capacity_env_steps, self.batch_size,
            self.minimum_replay_env_steps,
            self.minimum_action_enabled_env_steps,
            self.updates_per_env_step, self.learn_every_env_steps,
            self.exploration_noise_anneal_steps,
            self.latest_checkpoint_interval,
            self.periodic_checkpoint_interval, self.wandb_log_interval,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("training counts/intervals must be positive")
        if (
            self.early_stop_queued_threshold <= 0
            or self.early_stop_tat_threshold <= 0
            or self.early_stop_min_episode_steps < 0
            or self.max_stale_sim_time_ticks <= 0
        ):
            raise ValueError("early-stop/watchdog settings are invalid")
        if self.rail_tat_diagnostic_max_step < 0:
            raise ValueError(
                "rail_tat_diagnostic_max_step must be non-negative"
            )
        if (
            not np.isfinite(self.tat_confidence_n0)
            or self.tat_confidence_n0 <= 0
        ):
            raise ValueError("tat_confidence_n0 must be finite and positive")


class ClientAlgorithm:
    def __init__(self, config: ContextualRuntimeConfig | None = None):
        self.config = config or ContextualRuntimeConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        with torch.random.fork_rng(
            devices=[self.device] if self.device.type == "cuda" else []
        ):
            torch.manual_seed(self.config.seed)
            network_config = ContextualNetworkConfig()
            self.encoder = DirectionalContextEncoder(network_config).to(self.device)
            self.actor = ContextualActor(network_config).to(self.device)
        self.encoder.eval()
        self.actor.eval()

        root = Path(__file__).resolve().parent.parent
        self.cache_path = Path(
            self.config.topology_cache_path
            or root / "contextual_topology_cache.npz"
        )
        self.audit_path = Path(
            self.config.topology_audit_path
            or root / "topology_neighbor_audit.json"
        )
        self.topology = None
        self.observation_builder = None
        self.parameterDw: dict[int, float] = {}
        self.parameterC: dict[int, float] = {}
        self.parameterPassTimes: dict[int, list[float]] = {}
        self.total_steps = 0
        self.episode_steps = 0
        self.last_observation = None
        self.last_controlled_action = None
        self.last_policy_action = None
        self.last_exploratory_action = None
        self.last_diagnostics: dict[str, float] = {}
        self.reward_builder = None
        self.transition_aligner = None
        self.episode_id = 0
        self.checkpoint_loaded = False
        self.replay_buffer = None
        self.learner = None
        self.action_enabled_env_steps = 0
        self.exploration_rng = np.random.default_rng(self.config.seed)
        self.wandb_logger = ContextualWandbLogger(self.config)
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
        self._last_sim_time: float | None = None
        self._stale_sim_time_ticks = 0
        self._burnin_last_applied_action = None
        self._burnin_previous_applied_action = None
        root = Path(__file__).resolve().parent.parent
        self.checkpoint_root = Path(
            self.config.checkpoint_root
            or root / "checkpoints" / self.runtime_variant
        )
        self.rail_tat_diagnostic_path = Path(
            self.config.rail_tat_diagnostic_path
            or self.checkpoint_root
            / "diagnostics"
            / "rail_tat_completion.jsonl"
        )

    @property
    def algorithm_variant(self):
        return contextual_algorithm_variant(
            self.config.sale_enabled, self.config.lap_enabled
        )

    @property
    def action_version(self):
        return action_version(self.config.action_mode)

    @property
    def runtime_variant(self):
        base = (
            f"{ALGORITHM_VERSION}_{self.algorithm_variant}_"
            f"{self.action_version}"
        )
        if self.config.replay_sampling_mode == REPLAY_SAMPLING_RAIL:
            return base
        return (
            f"{base}_{REPLAY_SAMPLING_VERSION}_"
            f"{self.config.replay_sampling_mode}"
        )

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _ensure_initialized(self, pclient):
        if self.topology is not None:
            if self.reward_builder is None:
                self.reward_builder = ContextualRewardBuilder(
                    self.topology,
                    ContextualRewardConfig(
                        freeze_after_env_steps=self.config.normalizer_freeze_steps,
                        action_mode=self.config.action_mode,
                        smooth_b_rl_weight=self.config.smooth_b_rl_weight,
                        smooth_exp_residual_weight=(
                            self.config.smooth_exp_residual_weight
                        ),
                        tat_confidence_n0=self.config.tat_confidence_n0,
                        tat_confidence_ramp=self.config.tat_confidence_ramp,
                    ),
                    completion_diagnostic_path=self.rail_tat_diagnostic_path,
                    global_step_provider=lambda: self.total_steps,
                    completion_diagnostic_max_global_step=(
                        self.config.rail_tat_diagnostic_max_step
                    ),
                )
                self.transition_aligner = ContextualTransitionAligner(
                    self.topology, self.reward_builder
                )
                self.transition_aligner.episode_id = self.episode_id
            if self.config.mode == "training" and self.learner is None:
                self._initialize_training()
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
        self.reward_builder = ContextualRewardBuilder(
            self.topology,
            ContextualRewardConfig(
                freeze_after_env_steps=self.config.normalizer_freeze_steps,
                action_mode=self.config.action_mode,
                smooth_b_rl_weight=self.config.smooth_b_rl_weight,
                smooth_exp_residual_weight=(
                    self.config.smooth_exp_residual_weight
                ),
                tat_confidence_n0=self.config.tat_confidence_n0,
                tat_confidence_ramp=self.config.tat_confidence_ramp,
            ),
            completion_diagnostic_path=self.rail_tat_diagnostic_path,
            global_step_provider=lambda: self.total_steps,
            completion_diagnostic_max_global_step=(
                self.config.rail_tat_diagnostic_max_step
            ),
        )
        self.transition_aligner = ContextualTransitionAligner(
            self.topology, self.reward_builder
        )
        self.transition_aligner.episode_id = self.episode_id
        if self.config.mode == "training":
            self._initialize_training()

    def _initialize_training(self):
        if self.learner is not None:
            return
        self.replay_buffer = ContextualStepReplayBuffer(
            self.topology,
            self.observation_builder,
            capacity_env_steps=self.config.replay_capacity_env_steps,
            seed=self.config.seed,
            lap_enabled=self.config.lap_enabled,
            action_version=self.action_version,
            sampling_mode=self.config.replay_sampling_mode,
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
        self.learner = ContextualTD7Learner(
            self.replay_buffer,
            network_config=ContextualNetworkConfig(),
            config=learner_config,
            device=self.device,
            seed=self.config.seed,
        )
        # Runtime policy and learner online policy are the same objects.
        self.encoder = self.learner.encoder
        self.actor = self.learner.actor
        self.encoder.eval()
        self.actor.eval()
        self.transition_aligner.callback = self._on_completed_transition
        if self.config.resume_checkpoint_path:
            runtime_metadata = load_contextual_checkpoint(
                self.config.resume_checkpoint_path,
                self.learner,
                observation_builder=self.observation_builder,
                reward_builder=self.reward_builder,
                expected_runtime_metadata={
                    "algorithm_version": self.runtime_variant,
                    "action_mode": self.config.action_mode,
                    "action_version": self.action_version,
                    "action_scale": self.config.action_scale,
                    "curriculum_end_step": self.config.curriculum_end_step,
                    "curriculum_scale_start": (
                        self.config.curriculum_scale_start
                    ),
                    "curriculum_scale_end": self.config.curriculum_scale_end,
                    "curriculum_shape": self.config.curriculum_shape,
                    "exploration_noise_std": self.config.exploration_noise_std,
                    "episode_burnin_steps": self.config.episode_burnin_steps,
                },
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
            self._resume_requires_refill = True
            self.action_enabled_env_steps = 0

    def _on_completed_transition(self, transition):
        started = time.perf_counter()
        self.replay_buffer.push_transition(transition)
        self._last_replay_push_ms = (
            time.perf_counter() - started
        ) * 1000.0
        # `transition.env_step` is episode-local and restarts at zero after a
        # process resume. `total_steps` is restored from the checkpoint, so it
        # correctly identifies transitions produced by the resumed policy.
        if self.total_steps > self.config.warmup_steps:
            self.action_enabled_env_steps += 1
        if (
            self._resume_requires_refill
            and self.action_enabled_env_steps
            >= self.config.minimum_action_enabled_env_steps
            and self.replay_buffer.size_env_steps
            >= self.config.minimum_replay_env_steps
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
            and getattr(self.reward_builder.local_normalizer, "frozen", False)
            and getattr(self.reward_builder.global_normalizer, "frozen", False)
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
            "gate/not_failed": not self.training_failed,
        }
        return states, all(states.values())

    def _runtime_checkpoint_metadata(self, checkpoint_kind="latest"):
        return {
            "algorithm_version": self.runtime_variant,
            "uniform_replay": not self.config.lap_enabled,
            "sale": self.config.sale_enabled,
            "lap": self.config.lap_enabled,
            "critic_loss_mode": self.config.critic_loss_mode,
            "action_mode": self.config.action_mode,
            "action_version": self.action_version,
            "action_scale": self.config.action_scale,
            "curriculum_end_step": self.config.curriculum_end_step,
            "curriculum_scale_start": self.config.curriculum_scale_start,
            "curriculum_scale_end": self.config.curriculum_scale_end,
            "curriculum_shape": self.config.curriculum_shape,
            "episode_burnin_steps": self.config.episode_burnin_steps,
            "exploration_noise_std": self.config.exploration_noise_std,
            "exploration_noise_final_std": (
                self.config.exploration_noise_final_std
            ),
            "exploration_noise_anneal_steps": (
                self.config.exploration_noise_anneal_steps
            ),
            "exploration_noise_anneal_start_step": self.config.warmup_steps,
            "exploration_noise_anneal_end_step": (
                self.config.warmup_steps
                + self.config.exploration_noise_anneal_steps
            ),
            "exploration_schedule_version": (
                EXPLORATION_SCHEDULE_VERSION
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

    def _maybe_checkpoint(self):
        if self.training_failed:
            return
        if self.total_steps and (
            self.total_steps % self.config.latest_checkpoint_interval == 0
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
        self.last_controlled_action = None
        self.last_policy_action = None
        self.last_exploratory_action = None
        self._last_sim_time = None
        self._stale_sim_time_ticks = 0
        self._burnin_last_applied_action = None
        self._burnin_previous_applied_action = None
        if self.transition_aligner is not None:
            self.transition_aligner.reset(self.episode_id)
        if self.reward_builder is not None:
            self.reward_builder.reset_episode()

    def _clear_training_temporal_state(self) -> None:
        """Idempotently remove state that could create a post-failure transition."""
        self.last_observation = None
        self.last_controlled_action = None
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
        self.episode_id += 1
        self.episode_steps = 0
        self.last_observation = None
        self.last_controlled_action = None
        self.last_policy_action = None
        self.last_exploratory_action = None
        self.last_diagnostics = {}
        self._last_sim_time = None
        self._stale_sim_time_ticks = 0
        self._burnin_last_applied_action = None
        self._burnin_previous_applied_action = None
        if self.transition_aligner is not None:
            self.transition_aligner.reset(self.episode_id)
        if self.reward_builder is not None:
            self.reward_builder.reset_episode()

    def on_terminal(self):
        """Record terminal signal; no terminal state is available at protocol v=1."""
        if self.transition_aligner is not None:
            self.last_diagnostics["transition/terminal_without_observation"] = 1.0

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
        start = int(self.config.warmup_steps)
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
            0, self.total_steps - self.config.warmup_steps
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
        return (
            torch.from_numpy(observation.center_local),
            torch.from_numpy(observation.incoming_local),
            torch.from_numpy(observation.outgoing_local),
            torch.from_numpy(observation.incoming_relation),
            torch.from_numpy(observation.outgoing_relation),
            torch.from_numpy(observation.global_state),
        )

    def _actor_inference(self, observation, *, attention_diagnostics=False):
        t0 = time.perf_counter()
        cpu_tensors = self._cpu_tensors(observation)
        tensor_conversion_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        device_tensors = tuple(
            tensor.to(self.device, non_blocking=False) for tensor in cpu_tensors
        )
        batch = device_tensors[0].shape[0]
        global_batch = device_tensors[-1].unsqueeze(0).expand(batch, -1)
        self._synchronize()
        host_to_device_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            encoding = self.encoder(
                *device_tensors[:-1],
                global_batch,
                return_attention=attention_diagnostics,
            )
            structured_batch = (*device_tensors[:-1], global_batch)
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
                self.actor(encoding.state, sale_state)
                if sale_state is not None
                else self.actor(encoding.state)
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
                ("incoming", encoding.incoming_attention),
                ("outgoing", encoding.outgoing_attention),
            ):
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

    def _algorithm_impl(self, pclient):
        total_start = time.perf_counter()
        self._ensure_initialized(pclient)
        sim_time, stale_sim_time_ticks, protocol_stalled = (
            self._sim_time_progress(pclient)
        )
        queued = float(getattr(pclient, "QueuedCommandCount", 0) or 0)
        total_tat = float(getattr(pclient, "TotalTat", 0.0) or 0.0)
        done_by_queue = queued > self.config.early_stop_queued_threshold
        done_by_tat = (
            total_tat > self.config.early_stop_tat_threshold
            and self.episode_steps > self.config.early_stop_min_episode_steps
        )
        protocol_stalled = (
            self.config.mode == "training" and protocol_stalled
        )
        done = done_by_queue or done_by_tat or protocol_stalled
        termination_reason = (
            3.0 if protocol_stalled
            else 1.0 if done_by_queue
            else 2.0 if done_by_tat
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
            parameter_dw=self.parameterDw,
            parameter_c=self.parameterC,
        )
        observation_calls += 1
        observation_build_ms = (time.perf_counter() - t0) * 1000.0
        self.last_observation = observation

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
                and (not burnin_active or has_trained_policy)
            )
        )
        if use_actor:
            deterministic_policy, inference_timing = self._actor_inference(
                observation,
                attention_diagnostics=(
                    self.config.mode == "training"
                    and self.total_steps % self.config.wandb_log_interval == 0
                ),
            )
            timing.update(inference_timing)
        controlled_action = deterministic_policy.copy()
        exploration_noise_std = (
            0.0 if burnin_active else self._exploration_noise_std()
        )
        should_apply = (
            self.config.action_enabled
            and (
                (burnin_active and has_trained_policy)
                or (
                    not burnin_active
                    and self.total_steps >= self.config.warmup_steps
                )
            )
            and not self.training_failed
            and not done
        )
        if (
            self.config.mode == "training"
            and should_apply
            and not burnin_active
        ):
            noise = self.exploration_rng.normal(
                0.0,
                exploration_noise_std,
                size=controlled_action.shape,
            )
            noise = np.clip(
                noise,
                -self.config.exploration_noise_clip,
                self.config.exploration_noise_clip,
            )
            controlled_action = np.clip(
                controlled_action + noise, -1.0, 1.0
            ).astype(np.float32)

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
        for physical_row, rail_id_value in enumerate(self.topology.all_rail_ids):
            rail_id = int(rail_id_value)
            pclient.RAILLINECOST_DIC[rail_id].FRailLineCost = float(
                action_result.final_cost[physical_row]
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
            **timing,
            "action/finite_ratio": float(
                np.isfinite(controlled_action).mean()
            ),
            "runtime/nonfinite_count": float(
                np.size(action_result.final_cost)
                - np.isfinite(action_result.final_cost).sum()
            ),
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
        }
        if completed is not None:
            reward_diagnostics = self.reward_builder.diagnostics(
                completed.reward
            )
            self.last_diagnostics.update(reward_diagnostics)
            self.last_diagnostics.update({
                # Compatibility aliases used by the per-rail/region dashboards.
                "reward/step_reward": reward_diagnostics["reward/total_mean"],
                "reward/global_norm": reward_diagnostics[
                    "reward/global_normalized"
                ],
                "reward/local_norm": reward_diagnostics[
                    "reward/local_normalized_mean"
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
                    "reward/rail_tat_vector_mean"
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
                "reward/marg_tat": reward_diagnostics[
                    "reward/marginal_tat_ema"
                ],
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
            "protocol/stale_sim_time_ticks": float(stale_sim_time_ticks),
            "curriculum/action_scale": current_action_scale,
            "global/tat": float(getattr(pclient, "TotalTat", 0.0)),
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
        checkpoint_started = time.perf_counter()
        if self.config.mode == "training" and not self.training_failed:
            self._maybe_checkpoint()
        checkpoint_ms = (time.perf_counter() - checkpoint_started) * 1000.0
        total_algorithm_ms = (time.perf_counter() - total_start) * 1000.0
        self.last_diagnostics.update({
            "runtime/checkpoint_ms": checkpoint_ms,
            "runtime/total_algorithm_ms": total_algorithm_ms,
            "runtime/total_ms": total_algorithm_ms,
        })
        return action_result

    def record_send_cost_ms(self, elapsed_ms: float):
        send_ms = float(elapsed_ms)
        self.last_diagnostics["runtime/send_cost_ms"] = send_ms
        self.last_diagnostics["runtime/total_ms"] = (
            float(self.last_diagnostics.get("runtime/total_algorithm_ms", 0.0))
            + send_ms
        )

    def log_wandb_tick(self):
        if (
            self.config.mode == "training"
            and not self.training_failed
            and self.total_steps % self.config.wandb_log_interval == 0
        ):
            self.wandb_logger.log(self.last_diagnostics, self.total_steps)

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
        assigned = defaultdict(int)
        used = set()
        for job in job_list:
            for oht_id, oht in pclient.OHT_DIC.items():
                carrier_ok = any(
                    value in job.CarrierTypes for value in oht.CarrierTypes
                )
                area_ok = (
                    oht.RunningAreaType in job.RunningAreaTyes
                    or oht.RunningAreaType == 0
                    or job.RunningAreaTyes[0] == 0
                )
                if oht.State == 3 or not carrier_ok or not area_ok or oht_id in used:
                    continue
                if job.FromNode in list(oht.RouteList)[:16]:
                    assigned[job.ID] = oht_id
                    used.add(oht_id)
                    break
        return assigned
