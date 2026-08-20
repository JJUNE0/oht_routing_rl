"""Immutable one-step contextual transition alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from oht_routing.mdp.observation import ContextualObservationBatch
from oht_routing.mdp.reward.builder import (
    ContextualRewardBuilder,
    ControlledRewardBatch,
)
from oht_routing.mdp.topology import ContextualTopology


class ContextualTransitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PendingContextualStep:
    observation: ContextualObservationBatch
    controlled_action: np.ndarray
    applied_action: np.ndarray
    baseline_cost: np.ndarray
    final_cost: np.ndarray
    env_step: int
    episode_id: int
    topology_hash: str
    mapping_hash: str


@dataclass(frozen=True)
class CompletedContextualTransition:
    state: ContextualObservationBatch
    action: np.ndarray
    applied_action: np.ndarray
    reward: ControlledRewardBatch
    next_state: ContextualObservationBatch
    done: bool
    env_step: int
    next_env_step: int
    episode_id: int
    topology_hash: str
    mapping_hash: str


def _immutable(values, shape, name):
    result = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(shape).copy())
    if not np.isfinite(result).all():
        raise ContextualTransitionError(f"{name} contains NaN or Inf")
    result.setflags(write=False)
    return result


class ContextualTransitionAligner:
    def __init__(
        self,
        topology: ContextualTopology,
        reward_builder: ContextualRewardBuilder,
        callback: Callable[[CompletedContextualTransition], None] | None = None,
    ):
        self.topology = topology
        self.reward_builder = reward_builder
        self.callback = callback
        self.pending: PendingContextualStep | None = None
        self.previous_applied_action: np.ndarray | None = None
        self.completed_count = 0
        self.last_completed: CompletedContextualTransition | None = None
        self.episode_id = 0

    def reset(self, episode_id: int | None = None) -> None:
        self.pending = None
        self.previous_applied_action = None
        self.episode_id = self.episode_id + 1 if episode_id is None else int(episode_id)
        self.reward_builder.reset_episode()

    def _identity(self, observation: ContextualObservationBatch) -> None:
        if observation.topology_hash != self.topology.topology_hash:
            raise ContextualTransitionError("topology hash mismatch")
        if observation.mapping_hash != self.topology.mapping_hash:
            raise ContextualTransitionError("mapping hash mismatch")
        if not np.array_equal(
            observation.controlled_rail_ids, self.topology.controlled_rail_ids
        ):
            raise ContextualTransitionError("controlled rail row mismatch")

    def advance(
        self,
        *,
        observation: ContextualObservationBatch,
        pclient,
        controlled_action,
        applied_action,
        baseline_cost,
        final_cost,
        env_step: int,
        episode_id: int,
        done: bool = False,
    ) -> CompletedContextualTransition | None:
        completed = self.complete_previous(
            observation=observation,
            pclient=pclient,
            env_step=env_step,
            episode_id=episode_id,
            done=done,
        )
        if not done:
            self.stage_current(
                observation=observation,
                controlled_action=controlled_action,
                applied_action=applied_action,
                baseline_cost=baseline_cost,
                final_cost=final_cost,
                env_step=env_step,
                episode_id=episode_id,
            )
        return completed

    def complete_previous(
        self,
        *,
        observation: ContextualObservationBatch,
        pclient,
        env_step: int,
        episode_id: int,
        done: bool = False,
        dispatch: bool = True,
    ) -> CompletedContextualTransition | None:
        self._identity(observation)
        if int(episode_id) != self.episode_id:
            raise ContextualTransitionError("episode ID changed without reset")
        completed = None
        if self.pending is not None:
            if self.pending.episode_id != int(episode_id):
                raise ContextualTransitionError("cross-episode transition")
            if int(env_step) != self.pending.env_step + 1:
                raise ContextualTransitionError("environment step gap is not one")
            reward = self.reward_builder.build(
                pclient,
                applied_action=self.pending.applied_action,
                previous_applied_action=self.previous_applied_action,
                env_step=self.pending.env_step,
                episode_id=episode_id,
            )
            completed = CompletedContextualTransition(
                state=self.pending.observation,
                action=self.pending.controlled_action,
                applied_action=self.pending.applied_action,
                reward=reward,
                next_state=observation,
                done=bool(done),
                env_step=self.pending.env_step,
                next_env_step=int(env_step),
                episode_id=int(episode_id),
                topology_hash=self.topology.topology_hash,
                mapping_hash=self.topology.mapping_hash,
            )
            self.previous_applied_action = self.pending.applied_action
            self.pending = None
            self.completed_count += 1
            self.last_completed = completed
            if dispatch and self.callback is not None:
                self.callback(completed)

        if done:
            self.pending = None
        return completed

    def stage_current(
        self,
        *,
        observation: ContextualObservationBatch,
        controlled_action,
        applied_action,
        baseline_cost,
        final_cost,
        env_step: int,
        episode_id: int,
    ) -> None:
        self._identity(observation)
        if self.pending is not None:
            raise ContextualTransitionError(
                "previous pending step must be completed before staging current"
            )
        if int(episode_id) != self.episode_id:
            raise ContextualTransitionError("episode ID changed without reset")
        count = len(self.topology.controlled_rail_ids)
        physical = len(self.topology.all_rail_ids)
        self.pending = PendingContextualStep(
            observation=observation,
            controlled_action=_immutable(controlled_action, (count, 1), "action"),
            applied_action=_immutable(applied_action, (count, 1), "applied_action"),
            baseline_cost=_immutable(baseline_cost, (physical,), "baseline_cost"),
            final_cost=_immutable(final_cost, (physical,), "final_cost"),
            env_step=int(env_step),
            episode_id=int(episode_id),
            topology_hash=self.topology.topology_hash,
            mapping_hash=self.topology.mapping_hash,
        )

    def diagnostics(self, completed=None) -> dict[str, float]:
        transition = completed or self.last_completed
        return {
            "transition/pending_valid": float(self.pending is not None),
            "transition/completed_count": float(self.completed_count),
            "transition/done": float(transition.done) if transition else 0.0,
            "transition/env_step_gap": (
                float(transition.next_env_step - transition.env_step)
                if transition else 0.0
            ),
            "transition/episode_id": float(self.episode_id),
            "transition/boundary_count": 0.0,
            "transition/observation_identity_reused": (
                float(
                    transition is not None
                    and transition.next_state is (
                        self.pending.observation if self.pending else transition.next_state
                    )
                )
            ),
        }
