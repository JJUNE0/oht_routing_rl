"""Memory-efficient state-ring replay for contextual per-rail transitions."""

from __future__ import annotations

import time

import numpy as np
import torch

from oht_routing.mdp.action import ACTION_VERSION
from oht_routing.mdp.observation import (
    GLOBAL_DIM,
    LOCAL_DIM,
    OBSERVATION_VERSION,
)
from oht_routing.mdp.reward.config import (
    REWARD_VERSION,
    canonical_reward_version,
)
from oht_routing.mdp.topology import ContextualTopology
from oht_routing.mdp.transition import CompletedContextualTransition

from .replay_types import (
    ContextualReplayBatch,
    ContextualStepSnapshot,
    ReplaySampleKey,
)
from .stacking import stack_offsets, validate_stack_config


class ContextualReplayError(RuntimeError):
    pass


REPLAY_VERSION = "contextual_step_snapshot_previous_applied_action_v4"
REPLAY_SAMPLING_VERSION = "contextual_replay_sampling_modes_v2"
REPLAY_SAMPLING_RAIL = "rail"
REPLAY_SAMPLING_SNAPSHOT = "snapshot"
REPLAY_SAMPLING_RANDOM_RAIL = "random_rail"
REPLAY_SAMPLING_MODES = (
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    REPLAY_SAMPLING_RANDOM_RAIL,
)
LAP_VERSION = "contextual_snapshot_lap_hierarchical_v1"
LAP_PERFORMANCE_VERSION = "contextual_lap_cached_vectorized_v2"


def _copy_array(name, values, shape, dtype=np.float32):
    array = np.asarray(values, dtype=dtype)
    if array.shape != shape:
        raise ContextualReplayError(
            f"{name} shape mismatch: actual={array.shape}, expected={shape}"
        )
    if not np.isfinite(array).all():
        raise ContextualReplayError(f"{name} contains NaN or Inf")
    result = np.ascontiguousarray(array.copy())
    result.setflags(write=False)
    return result


def snapshot_from_transition(
    transition: CompletedContextualTransition,
    *,
    action_version: str = ACTION_VERSION,
    reward_version: str = REWARD_VERSION,
) -> ContextualStepSnapshot:
    """Copy the raw state and controlled vectors from a completed transition."""
    state_raw = transition.state.physical_local_raw
    state_global = transition.state.global_raw
    next_raw = transition.next_state.physical_local_raw
    next_global = transition.next_state.global_raw
    if any(value is None for value in (state_raw, state_global, next_raw, next_global)):
        raise ContextualReplayError(
            "transition observations do not contain raw snapshot exports"
        )
    physical_count = int(np.asarray(state_raw).shape[0])
    controlled_count = int(transition.action.shape[0])
    return ContextualStepSnapshot(
        physical_local_state=_copy_array(
            "physical_local_state", state_raw, (physical_count, LOCAL_DIM)
        ),
        global_state=_copy_array("global_state", state_global, (GLOBAL_DIM,)),
        previous_applied_action=_copy_array(
            "previous_applied_action",
            transition.state.previous_applied_action,
            (controlled_count, 1),
        ),
        policy_action=_copy_array(
            "policy_action", transition.action, (controlled_count, 1)
        ),
        applied_action=_copy_array(
            "applied_action", transition.applied_action, (controlled_count, 1)
        ),
        reward=_copy_array(
            "reward", transition.reward.total, (controlled_count,)
        ),
        next_physical_local_state=_copy_array(
            "next_physical_local_state", next_raw, (physical_count, LOCAL_DIM)
        ),
        next_global_state=_copy_array(
            "next_global_state", next_global, (GLOBAL_DIM,)
        ),
        next_previous_applied_action=_copy_array(
            "next_previous_applied_action",
            transition.next_state.previous_applied_action,
            (controlled_count, 1),
        ),
        done=bool(transition.done),
        env_step=int(transition.env_step),
        next_env_step=int(transition.next_env_step),
        episode_id=int(transition.episode_id),
        topology_hash=transition.topology_hash,
        mapping_hash=transition.mapping_hash,
        observation_version=OBSERVATION_VERSION,
        reward_version=str(reward_version),
        action_version=str(action_version),
    )


class ContextualStepReplayBuffer:
    """Fixed-capacity CPU ring indexed logically by (step slot, rail row)."""

    def __init__(
        self,
        topology: ContextualTopology,
        observation_builder,
        *,
        capacity_env_steps: int = 1_000,
        seed: int = 0,
        lap_enabled: bool = False,
        lap_alpha: float = 0.4,
        lap_min_priority: float = 1.0,
        action_version: str = ACTION_VERSION,
        reward_version: str = REWARD_VERSION,
        sampling_mode: str = REPLAY_SAMPLING_RAIL,
        num_stacks: int = 1,
        stack_interval: int = 1,
    ):
        if int(capacity_env_steps) <= 0:
            raise ValueError("capacity_env_steps must be positive")
        if sampling_mode not in REPLAY_SAMPLING_MODES:
            raise ValueError(
                f"sampling_mode must be one of {REPLAY_SAMPLING_MODES}"
            )
        if (
            sampling_mode
            in {REPLAY_SAMPLING_SNAPSHOT, REPLAY_SAMPLING_RANDOM_RAIL}
            and lap_enabled
        ):
            raise ValueError(
                f"{sampling_mode} replay sampling requires LAP disabled"
            )
        self.topology = topology
        self.observation_builder = observation_builder
        self.capacity = int(capacity_env_steps)
        # Worst case is one completed transition per episode: no adjacent
        # state can be shared, so C valid transitions require 2C state slots.
        self.state_capacity = 2 * self.capacity
        self.controlled_count = len(topology.controlled_rail_ids)
        self.physical_count = len(topology.all_rail_ids)
        self.rng = np.random.default_rng(seed)
        self.lap_enabled = bool(lap_enabled)
        self.lap_alpha = float(lap_alpha)
        self.lap_min_priority = float(lap_min_priority)
        self.action_version = str(action_version)
        self.reward_version = canonical_reward_version(reward_version)
        self.sampling_mode = str(sampling_mode)
        self.num_stacks, self.stack_interval = validate_stack_config(
            num_stacks, stack_interval
        )
        self._stack_offsets = stack_offsets(
            self.num_stacks, self.stack_interval
        )
        self._priority = (
            np.zeros((self.capacity, self.controlled_count), np.float32)
            if self.lap_enabled else None
        )
        self._priority_sum = (
            np.zeros(self.capacity, np.float64)
            if self.lap_enabled else None
        )
        self._priority_sq_sum = (
            np.zeros(self.capacity, np.float64)
            if self.lap_enabled else None
        )
        self._priority_min = (
            np.zeros(self.capacity, np.float32)
            if self.lap_enabled else None
        )
        self._priority_max = (
            np.zeros(self.capacity, np.float32)
            if self.lap_enabled else None
        )
        self.max_priority = float(self.lap_min_priority)
        self.priority_update_count = 0

        # Adjacent transitions share states. The 2C allocation also preserves
        # full transition capacity under pathological reset frequency.
        self._physical = np.empty(
            (self.state_capacity, self.physical_count, LOCAL_DIM), np.float32
        )
        self._global = np.empty(
            (self.state_capacity, GLOBAL_DIM), np.float32
        )
        self._previous_applied_action = np.empty(
            (self.state_capacity, self.controlled_count), np.float32
        )
        self._state_generation = np.zeros(self.state_capacity, np.int64)
        self._state_valid = np.zeros(self.state_capacity, bool)
        self._state_keys: list[tuple[int, int] | None] = [
            None
        ] * self.state_capacity
        self._key_to_state: dict[tuple[int, int], tuple[int, int]] = {}
        self._state_cursor = 0

        shape = (self.capacity, self.controlled_count)
        self._policy_action = np.empty(shape, np.float32)
        self._applied_action = np.empty(shape, np.float32)
        self._reward = np.empty(shape, np.float32)
        self._done = np.empty(self.capacity, np.float32)
        self._env_step = np.empty(self.capacity, np.int64)
        self._episode_id = np.empty(self.capacity, np.int64)
        self._state_slot = np.empty(self.capacity, np.int64)
        self._state_gen_ref = np.empty(self.capacity, np.int64)
        self._next_state_slot = np.empty(self.capacity, np.int64)
        self._next_state_gen_ref = np.empty(self.capacity, np.int64)
        self._transition_generation = np.zeros(self.capacity, np.int64)
        self._transition_valid = np.zeros(self.capacity, bool)
        self._transition_keys: list[tuple[int, int] | None] = [
            None
        ] * self.capacity
        self._key_to_transition: dict[tuple[int, int], tuple[int, int]] = {}
        self._episode_first_step: dict[int, int] = {}
        self._transition_cursor = 0

        id_to_physical = {
            int(rail_id): row
            for row, rail_id in enumerate(topology.all_rail_ids)
        }
        self._center_rows = np.ascontiguousarray(
            topology.controlled_row_to_physical_index.astype(np.int64)
        )
        self._incoming_rows = np.ascontiguousarray(np.asarray([
            [id_to_physical[int(rail_id)] for rail_id in row]
            for row in topology.incoming_neighbor_ids
        ], dtype=np.int64))
        self._outgoing_rows = np.ascontiguousarray(np.asarray([
            [id_to_physical[int(rail_id)] for rail_id in row]
            for row in topology.outgoing_neighbor_ids
        ], dtype=np.int64))

        self.push_count = 0
        self.sample_count = 0
        self.overwrite_count = 0
        self.stale_key_reject_count = 0
        self.hash_mismatch_count = 0
        self.episode_crossing_count = 0
        self._last_sample_diag: dict[str, float] = {}

    def _validate_snapshot(self, snapshot: ContextualStepSnapshot) -> None:
        expected_versions = (
            (snapshot.observation_version, OBSERVATION_VERSION, "observation"),
            (snapshot.reward_version, self.reward_version, "reward"),
            (snapshot.action_version, self.action_version, "action"),
        )
        if snapshot.topology_hash != self.topology.topology_hash:
            self.hash_mismatch_count += 1
            raise ContextualReplayError("topology hash mismatch")
        if snapshot.mapping_hash != self.topology.mapping_hash:
            self.hash_mismatch_count += 1
            raise ContextualReplayError("mapping hash mismatch")
        for actual, expected, name in expected_versions:
            if actual != expected:
                raise ContextualReplayError(
                    f"{name} version mismatch: {actual} != {expected}"
                )
        if snapshot.next_env_step != snapshot.env_step + 1:
            raise ContextualReplayError("next_env_step must equal env_step + 1")
        _copy_array("physical_local_state", snapshot.physical_local_state,
                    (self.physical_count, LOCAL_DIM))
        _copy_array("global_state", snapshot.global_state, (GLOBAL_DIM,))
        _copy_array("previous_applied_action", snapshot.previous_applied_action,
                    (self.controlled_count, 1))
        _copy_array("policy_action", snapshot.policy_action,
                    (self.controlled_count, 1))
        _copy_array("applied_action", snapshot.applied_action,
                    (self.controlled_count, 1))
        _copy_array("reward", snapshot.reward, (self.controlled_count,))
        _copy_array("next_physical_local_state",
                    snapshot.next_physical_local_state,
                    (self.physical_count, LOCAL_DIM))
        _copy_array("next_global_state", snapshot.next_global_state,
                    (GLOBAL_DIM,))
        _copy_array("next_previous_applied_action",
                    snapshot.next_previous_applied_action,
                    (self.controlled_count, 1))
        if not np.array_equal(
            snapshot.next_previous_applied_action,
            snapshot.applied_action,
        ):
            raise ContextualReplayError(
                "next_previous_applied_action must equal applied_action"
            )

    def _store_state(
        self, key, physical, global_state, previous_applied_action
    ) -> tuple[int, int]:
        existing = self._key_to_state.get(key)
        if existing is not None:
            slot, generation = existing
            if (
                self._state_valid[slot]
                and self._state_generation[slot] == generation
            ):
                if not (
                    np.array_equal(self._physical[slot], physical)
                    and np.array_equal(self._global[slot], global_state)
                    and np.array_equal(
                        self._previous_applied_action[slot],
                        np.asarray(previous_applied_action).reshape(-1),
                    )
                ):
                    raise ContextualReplayError(
                        "same episode/env step has different raw state"
                    )
                return slot, generation

        slot = self._state_cursor
        self._state_cursor = (slot + 1) % self.state_capacity
        old_key = self._state_keys[slot]
        if old_key is not None and self._key_to_state.get(old_key, (None,))[0] == slot:
            self._key_to_state.pop(old_key, None)
        generation = int(self._state_generation[slot]) + 1
        self._physical[slot] = physical
        self._global[slot] = global_state
        self._previous_applied_action[slot] = np.asarray(
            previous_applied_action, dtype=np.float32
        ).reshape(-1)
        self._state_generation[slot] = generation
        self._state_valid[slot] = True
        self._state_keys[slot] = key
        self._key_to_state[key] = (slot, generation)
        return slot, generation

    def _state_reference(self, episode: int, step: int) -> tuple[int, int] | None:
        origin = self._episode_first_step.get(int(episode))
        if origin is None:
            return None
        requested = max(int(step), int(origin))
        reference = self._key_to_state.get((int(episode), requested))
        if reference is None:
            return None
        slot, generation = reference
        if not (
            self._state_valid[slot]
            and self._state_generation[slot] == generation
        ):
            return None
        return int(slot), int(generation)

    def _transition_reference(
        self, episode: int, step: int, *, pad_episode_start: bool = False
    ) -> tuple[int, int] | None:
        origin = self._episode_first_step.get(int(episode))
        if origin is None:
            return None
        requested = int(step)
        if requested < int(origin):
            if not pad_episode_start:
                return None
            requested = int(origin)
        reference = self._key_to_transition.get((int(episode), requested))
        if reference is None:
            return None
        slot, generation = reference
        if not (
            self._transition_valid[slot]
            and self._transition_generation[slot] == generation
        ):
            return None
        return int(slot), int(generation)

    def push(self, snapshot: ContextualStepSnapshot) -> ReplaySampleKey:
        self._validate_snapshot(snapshot)
        state_key = (int(snapshot.episode_id), int(snapshot.env_step))
        next_key = (int(snapshot.episode_id), int(snapshot.next_env_step))
        if state_key[0] != next_key[0]:
            self.episode_crossing_count += 1
            raise ContextualReplayError("episode-crossing snapshot")
        self._episode_first_step.setdefault(state_key[0], state_key[1])
        state_slot, state_gen = self._store_state(
            state_key,
            snapshot.physical_local_state,
            snapshot.global_state,
            snapshot.previous_applied_action,
        )
        next_slot, next_gen = self._store_state(
            next_key, snapshot.next_physical_local_state,
            snapshot.next_global_state,
            snapshot.next_previous_applied_action,
        )

        slot = self._transition_cursor
        self._transition_cursor = (slot + 1) % self.capacity
        if self._transition_valid[slot]:
            self.overwrite_count += 1
            old_key = self._transition_keys[slot]
            if old_key is not None:
                current = self._key_to_transition.get(old_key)
                if current is not None and current[0] == slot:
                    self._key_to_transition.pop(old_key, None)
        generation = int(self._transition_generation[slot]) + 1
        self._policy_action[slot] = snapshot.policy_action[:, 0]
        self._applied_action[slot] = snapshot.applied_action[:, 0]
        self._reward[slot] = snapshot.reward
        self._done[slot] = float(snapshot.done)
        self._env_step[slot] = snapshot.env_step
        self._episode_id[slot] = snapshot.episode_id
        self._state_slot[slot], self._state_gen_ref[slot] = state_slot, state_gen
        self._next_state_slot[slot] = next_slot
        self._next_state_gen_ref[slot] = next_gen
        self._transition_generation[slot] = generation
        self._transition_valid[slot] = True
        self._transition_keys[slot] = state_key
        self._key_to_transition[state_key] = (slot, generation)
        if self.lap_enabled:
            self._priority[slot].fill(self.max_priority)
            self._priority_sum[slot] = (
                self.max_priority * self.controlled_count
            )
            self._priority_sq_sum[slot] = (
                self.max_priority * self.max_priority * self.controlled_count
            )
            self._priority_min[slot] = self.max_priority
            self._priority_max[slot] = self.max_priority
        self.push_count += 1
        return ReplaySampleKey(slot, generation, 0)

    def push_transition(
        self, transition: CompletedContextualTransition
    ) -> ReplaySampleKey:
        return self.push(
            snapshot_from_transition(
                transition,
                action_version=self.action_version,
                reward_version=self.reward_version,
            )
        )

    def _valid_transition_slots(self) -> np.ndarray:
        candidates = np.flatnonzero(self._transition_valid)
        if candidates.size == 0:
            return candidates
        state_ok = (
            self._state_valid[self._state_slot[candidates]]
            & (self._state_generation[self._state_slot[candidates]]
               == self._state_gen_ref[candidates])
        )
        next_ok = (
            self._state_valid[self._next_state_slot[candidates]]
            & (self._state_generation[self._next_state_slot[candidates]]
               == self._next_state_gen_ref[candidates])
        )
        candidates = candidates[state_ok & next_ok]
        if self.num_stacks == 1 or candidates.size == 0:
            return candidates

        # A retained episode origin may be duplicated for its initial stack.
        # If that origin has already fallen out of the circular buffer, only
        # sample steps at least one complete history horizon after the oldest
        # retained transition for that episode.
        eligible = np.zeros(candidates.size, dtype=bool)
        episodes = self._episode_id[candidates]
        steps = self._env_step[candidates]
        history_horizon = self._stack_offsets[-1]
        for episode in np.unique(episodes):
            positions = np.flatnonzero(episodes == episode)
            episode_steps = steps[positions]
            origin = self._episode_first_step.get(int(episode))
            origin_retained = bool(
                origin is not None and np.any(episode_steps == int(origin))
            )
            if origin_retained:
                eligible[positions] = True
            else:
                oldest_retained = int(episode_steps.min())
                eligible[positions] = (
                    episode_steps >= oldest_retained + history_horizon
                )
        return candidates[eligible]

    def validate_sample_keys(self, keys) -> None:
        keys = tuple(keys)
        if not keys:
            return
        slots = np.fromiter(
            (key.step_slot for key in keys), dtype=np.int64, count=len(keys)
        )
        generations = np.fromiter(
            (key.generation for key in keys), dtype=np.int64, count=len(keys)
        )
        rows = np.fromiter(
            (key.controlled_row for key in keys),
            dtype=np.int64,
            count=len(keys),
        )
        in_bounds = (
            (slots >= 0) & (slots < self.capacity)
            & (rows >= 0) & (rows < self.controlled_count)
        )
        valid = in_bounds.copy()
        bounded = np.flatnonzero(valid)
        if bounded.size:
            checked_slots = slots[bounded]
            valid[bounded] &= (
                self._transition_valid[checked_slots]
                & (
                    self._transition_generation[checked_slots]
                    == generations[bounded]
                )
            )
        bounded = np.flatnonzero(valid)
        if bounded.size and self.num_stacks > 1:
            valid_slots = self._valid_transition_slots()
            valid[bounded] &= np.isin(slots[bounded], valid_slots)
        bounded = np.flatnonzero(valid)
        if bounded.size:
            checked_slots = slots[bounded]
            state_slots = self._state_slot[checked_slots]
            next_slots = self._next_state_slot[checked_slots]
            valid[bounded] &= (
                self._state_valid[state_slots]
                & (
                    self._state_generation[state_slots]
                    == self._state_gen_ref[checked_slots]
                )
                & self._state_valid[next_slots]
                & (
                    self._state_generation[next_slots]
                    == self._next_state_gen_ref[checked_slots]
                )
            )
        if not valid.all():
            self.stale_key_reject_count += 1
            bad = int(np.flatnonzero(~valid)[0])
            raise ContextualReplayError(f"stale replay key: {keys[bad]}")

    def update_priorities(self, sample_keys, priorities) -> None:
        self.validate_sample_keys(sample_keys)
        if not self.lap_enabled:
            raise NotImplementedError(
                "priority update is unsupported when LAP is disabled"
            )
        values = np.asarray(priorities, dtype=np.float64).reshape(-1)
        if values.shape != (len(sample_keys),) or not np.isfinite(values).all():
            raise ContextualReplayError("priorities must be finite [sample keys]")
        values = np.maximum(
            np.power(np.maximum(values, 0.0), self.lap_alpha),
            self.lap_min_priority,
        ).astype(np.float32)
        slots = np.fromiter(
            (key.step_slot for key in sample_keys),
            dtype=np.int64,
            count=len(sample_keys),
        )
        rows = np.fromiter(
            (key.controlled_row for key in sample_keys),
            dtype=np.int64,
            count=len(sample_keys),
        )
        # A sampled pair may occur more than once. Preserve the former
        # sequential-update contract by retaining the last supplied value.
        flat = slots * self.controlled_count + rows
        _, reverse_positions = np.unique(flat[::-1], return_index=True)
        positions = len(flat) - 1 - reverse_positions
        slots = slots[positions]
        rows = rows[positions]
        values = values[positions]
        old = self._priority[slots, rows].astype(np.float64)
        values64 = values.astype(np.float64)
        self._priority[slots, rows] = values
        np.add.at(self._priority_sum, slots, values64 - old)
        np.add.at(
            self._priority_sq_sum,
            slots,
            np.square(values64) - np.square(old),
        )
        affected_slots = np.unique(slots)
        affected = self._priority[affected_slots]
        self._priority_min[affected_slots] = affected.min(axis=1)
        self._priority_max[affected_slots] = affected.max(axis=1)
        valid_slots = self._valid_transition_slots()
        self.max_priority = (
            max(
                self.lap_min_priority,
                float(self._priority_max[valid_slots].max(initial=0.0)),
            )
            if valid_slots.size else self.lap_min_priority
        )
        self.priority_update_count += 1

    @staticmethod
    def _readonly_normalize(normalizer, values, name):
        # normalize() is read-only; update() is intentionally never called.
        return normalizer.normalize(values, name=name)

    def _history_slot_arrays(self, transition_slots: np.ndarray):
        unique_slots, inverse = np.unique(
            np.asarray(transition_slots, dtype=np.int64), return_inverse=True
        )
        shape = (unique_slots.size, self.num_stacks)
        state_history = np.empty(shape, np.int64)
        next_state_history = np.empty(shape, np.int64)
        action_history = np.full(shape, -1, np.int64)
        next_action_history = np.full(shape, -1, np.int64)
        for row, transition_slot in enumerate(unique_slots):
            episode = int(self._episode_id[transition_slot])
            step = int(self._env_step[transition_slot])
            origin = self._episode_first_step.get(episode)
            if origin is None:
                raise ContextualReplayError("sampled episode has no stack origin")
            for index, offset in enumerate(self._stack_offsets):
                state_reference = self._state_reference(
                    episode, step - offset
                )
                next_state_reference = self._state_reference(
                    episode, step + 1 - offset
                )
                if state_reference is None or next_state_reference is None:
                    raise ContextualReplayError(
                        "sampled transition lost required state history"
                    )
                state_history[row, index] = state_reference[0]
                next_state_history[row, index] = next_state_reference[0]

                action_step = step - offset
                action_reference = self._transition_reference(
                    episode, action_step, pad_episode_start=True
                )
                if action_reference is None:
                    raise ContextualReplayError(
                        "sampled transition lost required action history"
                    )
                action_history[row, index] = action_reference[0]

                if index > 0:
                    next_action_step = step + 1 - offset
                    next_action_reference = self._transition_reference(
                        episode,
                        next_action_step,
                        pad_episode_start=True,
                    )
                    if next_action_reference is None:
                        raise ContextualReplayError(
                            "sampled transition lost required next-action history"
                        )
                    next_action_history[row, index] = next_action_reference[0]
        return (
            state_history[inverse],
            next_state_history[inverse],
            action_history[inverse],
            next_action_history[inverse],
        )

    def sample(
        self, batch_size: int, *, device: str | torch.device = "cpu"
    ) -> ContextualReplayBatch:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        valid_slots = self._valid_transition_slots()
        if valid_slots.size == 0:
            raise ContextualReplayError("cannot sample an empty replay")
        started = time.perf_counter()
        if self.sampling_mode == REPLAY_SAMPLING_RANDOM_RAIL:
            logical_count = int(valid_slots.size * self.controlled_count)
            if int(batch_size) > logical_count:
                raise ContextualReplayError(
                    "random-rail batch size exceeds the number of valid "
                    f"logical transitions: batch={int(batch_size)}, "
                    f"valid={logical_count}"
                )
            # Treat the complete (valid step, controlled rail) Cartesian
            # product as one flat replay pool. Sampling without replacement
            # guarantees that a logical rail transition appears at most once
            # in an optimizer batch, while allowing any number of different
            # rails to come from the same environment step.
            logical_indices = self.rng.choice(
                logical_count,
                size=int(batch_size),
                replace=False,
                shuffle=False,
            )
            transition_slots = valid_slots[
                logical_indices // self.controlled_count
            ]
            controlled_rows = (
                logical_indices % self.controlled_count
            ).astype(np.int64, copy=False)
            sample_probabilities = np.full(
                int(batch_size),
                1.0 / logical_count,
                np.float64,
            )
        elif self.sampling_mode == REPLAY_SAMPLING_SNAPSHOT:
            if int(batch_size) > valid_slots.size:
                raise ContextualReplayError(
                    "snapshot batch size exceeds the number of valid "
                    f"environment steps: batch={int(batch_size)}, "
                    f"valid={int(valid_slots.size)}"
                )
            # A snapshot batch means distinct environment steps, with every
            # controlled rail from each selected step included exactly once.
            snapshot_slots = self.rng.choice(
                valid_slots, size=int(batch_size), replace=False
            )
            transition_slots = np.repeat(
                snapshot_slots, self.controlled_count
            )
            controlled_rows = np.tile(
                np.arange(self.controlled_count, dtype=np.int64),
                int(batch_size),
            )
            sample_probabilities = np.full(
                transition_slots.size,
                1.0 / len(valid_slots),
                np.float64,
            )
        # Rail sampling is the original behavior. With replacement is
        # explicit: any positive batch is allowed once one snapshot exists.
        elif self.lap_enabled:
            step_sums = self._priority_sum[valid_slots]
            total_priority = float(step_sums.sum())
            if not np.isfinite(total_priority) or total_priority <= 0:
                raise ContextualReplayError("invalid LAP total priority")
            transition_slots = self.rng.choice(
                valid_slots, size=int(batch_size), replace=True,
                p=step_sums / total_priority,
            )
            controlled_rows = np.empty(int(batch_size), np.int64)
            unique_slots, inverse = np.unique(
                transition_slots, return_inverse=True
            )
            for group, slot in enumerate(unique_slots):
                batch_rows = np.flatnonzero(inverse == group)
                cdf = np.cumsum(
                    self._priority[slot], dtype=np.float64
                )
                thresholds = self.rng.random(batch_rows.size) * cdf[-1]
                controlled_rows[batch_rows] = np.searchsorted(
                    cdf, thresholds, side="right"
                )
            sample_probabilities = (
                self._priority[transition_slots, controlled_rows].astype(
                    np.float64
                )
                / total_priority
            )
        else:
            transition_slots = self.rng.choice(
                valid_slots, size=int(batch_size), replace=True
            )
            controlled_rows = self.rng.integers(
                0, self.controlled_count, size=int(batch_size), dtype=np.int64
            )
            sample_probabilities = np.full(
                int(batch_size),
                1.0 / (len(valid_slots) * self.controlled_count),
                np.float64,
            )
        (
            state_slots,
            next_slots,
            action_slots,
            next_action_slots,
        ) = self._history_slot_arrays(transition_slots)
        batch_count = int(transition_slots.size)
        state_global_raw = self._global[state_slots]
        next_global_raw = self._global[next_slots]
        previous_applied_action = self._previous_applied_action[
            state_slots, controlled_rows[:, None]
        ].astype(np.float32, copy=True)
        next_previous_applied_action = self._previous_applied_action[
            next_slots, controlled_rows[:, None]
        ].astype(np.float32, copy=True)
        global_norm = self._readonly_normalize(
            self.observation_builder.global_normalizer,
            state_global_raw.reshape(-1, GLOBAL_DIM), "replay_state_global"
        ).reshape(batch_count, self.num_stacks, GLOBAL_DIM)
        next_global_norm = self._readonly_normalize(
            self.observation_builder.global_normalizer,
            next_global_raw.reshape(-1, GLOBAL_DIM), "replay_next_global"
        ).reshape(batch_count, self.num_stacks, GLOBAL_DIM)
        center_rows = self._center_rows[controlled_rows]
        incoming_rows = self._incoming_rows[controlled_rows]
        outgoing_rows = self._outgoing_rows[controlled_rows]
        current_center_raw = self._physical[
            state_slots, center_rows[:, None]
        ]
        current_incoming_raw = self._physical[
            state_slots[:, :, None], incoming_rows[:, None, :]
        ]
        current_outgoing_raw = self._physical[
            state_slots[:, :, None], outgoing_rows[:, None, :]
        ]
        next_center_raw = self._physical[
            next_slots, center_rows[:, None]
        ]
        next_incoming_raw = self._physical[
            next_slots[:, :, None], incoming_rows[:, None, :]
        ]
        next_outgoing_raw = self._physical[
            next_slots[:, :, None], outgoing_rows[:, None, :]
        ]
        local_normalizer = self.observation_builder.local_normalizer
        center = self._readonly_normalize(
            local_normalizer,
            current_center_raw.reshape(-1, LOCAL_DIM),
            "replay_center_local",
        ).reshape(batch_count, self.num_stacks, LOCAL_DIM)
        incoming = self._readonly_normalize(
            local_normalizer,
            current_incoming_raw.reshape(-1, LOCAL_DIM),
            "replay_incoming_local",
        ).reshape(current_incoming_raw.shape)
        outgoing = self._readonly_normalize(
            local_normalizer,
            current_outgoing_raw.reshape(-1, LOCAL_DIM),
            "replay_outgoing_local",
        ).reshape(current_outgoing_raw.shape)
        next_center = self._readonly_normalize(
            local_normalizer,
            next_center_raw.reshape(-1, LOCAL_DIM),
            "replay_next_center_local",
        ).reshape(batch_count, self.num_stacks, LOCAL_DIM)
        next_incoming = self._readonly_normalize(
            local_normalizer,
            next_incoming_raw.reshape(-1, LOCAL_DIM),
            "replay_next_incoming_local",
        ).reshape(next_incoming_raw.shape)
        next_outgoing = self._readonly_normalize(
            local_normalizer,
            next_outgoing_raw.reshape(-1, LOCAL_DIM),
            "replay_next_outgoing_local",
        ).reshape(next_outgoing_raw.shape)

        relation_shape = (
            batch_count,
            self.num_stacks,
            self._incoming_rows.shape[1],
            self.observation_builder._incoming_relation.shape[-1],
        )
        incoming_relation = np.broadcast_to(
            self.observation_builder._incoming_relation[controlled_rows, None],
            relation_shape,
        )
        outgoing_relation = np.broadcast_to(
            self.observation_builder._outgoing_relation[controlled_rows, None],
            relation_shape,
        )

        safe_action_slots = np.maximum(action_slots, 0)
        safe_next_action_slots = np.maximum(next_action_slots, 0)
        action_rows = controlled_rows[:, None]
        policy_action = self._policy_action[
            safe_action_slots, action_rows
        ].astype(np.float32, copy=True)
        applied_action = self._applied_action[
            safe_action_slots, action_rows
        ].astype(np.float32, copy=True)
        next_applied_action = self._applied_action[
            safe_next_action_slots, action_rows
        ].astype(np.float32, copy=True)
        policy_action[action_slots < 0] = 0.0
        applied_action[action_slots < 0] = 0.0
        next_applied_action[next_action_slots < 0] = 0.0

        if self.num_stacks == 1:
            center = center[:, 0]
            incoming = incoming[:, 0]
            outgoing = outgoing[:, 0]
            incoming_relation = incoming_relation[:, 0]
            outgoing_relation = outgoing_relation[:, 0]
            global_norm = global_norm[:, 0]
            previous_applied_action = previous_applied_action[:, 0, None]
            next_center = next_center[:, 0]
            next_incoming = next_incoming[:, 0]
            next_outgoing = next_outgoing[:, 0]
            next_global_norm = next_global_norm[:, 0]
            next_previous_applied_action = (
                next_previous_applied_action[:, 0, None]
            )
            policy_action = policy_action[:, 0, None]
            applied_action = applied_action[:, 0, None]
            next_applied_action = next_applied_action[:, 0, None]
        else:
            previous_applied_action = previous_applied_action[:, :, None]
            policy_action = policy_action[:, :, None]
            applied_action = applied_action[:, :, None]
            next_previous_applied_action = (
                next_previous_applied_action[:, :, None]
            )
            next_applied_action = next_applied_action[:, :, None]

        arrays = {
            "center_local": center,
            "incoming_local": incoming,
            "outgoing_local": outgoing,
            "incoming_relation": incoming_relation,
            "outgoing_relation": outgoing_relation,
            "global_state": global_norm,
            "previous_applied_action": previous_applied_action,
            "policy_action": policy_action,
            "applied_action": applied_action,
            "reward": self._reward[transition_slots, controlled_rows][:, None],
            "next_center_local": next_center,
            "next_incoming_local": next_incoming,
            "next_outgoing_local": next_outgoing,
            "next_incoming_relation": incoming_relation,
            "next_outgoing_relation": outgoing_relation,
            "next_global_state": next_global_norm,
            "next_previous_applied_action": next_previous_applied_action,
            "next_applied_action": next_applied_action,
            "done": self._done[transition_slots][:, None],
            "controlled_rail_id": self.topology.controlled_rail_ids[
                controlled_rows
            ],
            "env_step": self._env_step[transition_slots],
            "episode_id": self._episode_id[transition_slots],
        }
        materialize_ms = (time.perf_counter() - started) * 1000
        host_started = time.perf_counter()
        target = torch.device(device)
        tensors = {}
        for name, values in arrays.items():
            if name in {"controlled_rail_id", "env_step", "episode_id"}:
                array = np.array(
                    values, dtype=np.int64, order="C", copy=True
                )
            else:
                array = np.array(
                    values, dtype=np.float32, order="C", copy=True
                )
            tensors[name] = torch.from_numpy(array).to(target)
        host_to_device_ms = (time.perf_counter() - host_started) * 1000
        keys = tuple(
            ReplaySampleKey(
                int(slot), int(self._transition_generation[slot]), int(row)
            )
            for slot, row in zip(transition_slots, controlled_rows)
        )
        self.sample_count += 1
        self._last_sample_diag = {
            "replay/sample_reward_mean": float(arrays["reward"].mean()),
            "replay/sample_reward_std": float(arrays["reward"].std()),
            "replay/sample_done_ratio": float(arrays["done"].mean()),
            "replay/sample_policy_action_std": float(
                arrays["policy_action"].std()
            ),
            "replay/sample_applied_action_std": float(
                arrays["applied_action"].std()
            ),
            "stack/num_stacks": float(self.num_stacks),
            "stack/interval": float(self.stack_interval),
            "replay/sample_materialize_ms": materialize_ms,
            "replay/host_to_device_ms": host_to_device_ms,
            "lap/sample_probability_max": float(
                sample_probabilities.max(initial=0)
            ),
            "lap/effective_sample_size": float(
                1.0 / np.square(
                    sample_probabilities / sample_probabilities.sum()
                ).sum()
            ),
        }
        return ContextualReplayBatch(**tensors, sample_keys=keys)

    @property
    def size_env_steps(self) -> int:
        return int(self._valid_transition_slots().size)

    @staticmethod
    def estimate_capacity_bytes(
        capacity_env_steps: int,
        physical_count: int = 4_999,
        controlled_count: int = 4_996,
        lap_enabled: bool = False,
    ) -> int:
        c = int(capacity_env_steps)
        state = (2 * c) * (
            physical_count * LOCAL_DIM + GLOBAL_DIM + controlled_count
        ) * 4
        vectors = c * controlled_count * 3 * 4
        metadata = c * (7 * 8 + 4 + 1) + (2 * c) * (8 + 1)
        static_mapping = controlled_count * (1 + 2 * 10) * 8
        priority = (
            c * controlled_count * 4 + c * (2 * 8 + 2 * 4)
            if lap_enabled else 0
        )
        return int(state + vectors + metadata + static_mapping + priority)

    @property
    def storage_bytes(self) -> int:
        total = 0
        for value in self.__dict__.values():
            if isinstance(value, np.ndarray):
                total += value.nbytes
        return int(total)

    def diagnostics(self) -> dict[str, float]:
        result = {
            "replay/capacity_env_steps": float(self.capacity),
            "replay/capacity_logical_transitions": float(
                self.capacity * self.controlled_count
            ),
            "replay/size_env_steps": float(self.size_env_steps),
            "replay/size_logical_transitions": float(
                self.size_env_steps * self.controlled_count
            ),
            "replay/push_count": float(self.push_count),
            "replay/sample_count": float(self.sample_count),
            "replay/overwrite_count": float(self.overwrite_count),
            "replay/stale_key_reject_count": float(
                self.stale_key_reject_count
            ),
            "replay/boundary_transition_count": 0.0,
            "replay/episode_crossing_count": float(self.episode_crossing_count),
            "replay/hash_mismatch_count": float(self.hash_mismatch_count),
            "replay/stack_boundary_excluded_env_steps": float(
                np.count_nonzero(self._transition_valid)
                - self.size_env_steps
            ),
            "replay/storage_bytes": float(self.storage_bytes),
            "replay/estimated_capacity_bytes": float(
                self.estimate_capacity_bytes(
                    self.capacity, self.physical_count, self.controlled_count
                    , self.lap_enabled
                )
            ),
            "lap/enabled": float(self.lap_enabled),
            "lap/stale_update_reject_count": float(
                self.stale_key_reject_count
            ),
            "lap/new_transition_max_priority": float(self.max_priority),
            "stack/num_stacks": float(self.num_stacks),
            "stack/interval": float(self.stack_interval),
            "stack/history_horizon": float(
                (self.num_stacks - 1) * self.stack_interval
            ),
        }
        if self.lap_enabled:
            slots = self._valid_transition_slots()
            count = slots.size * self.controlled_count
            total = float(self._priority_sum[slots].sum())
            total_sq = float(self._priority_sq_sum[slots].sum())
            mean = total / count if count else 0.0
            variance = (
                max(0.0, total_sq / count - mean * mean)
                if count else 0.0
            )
            result.update({
                "lap/priority_mean": mean,
                "lap/priority_std": float(np.sqrt(variance)),
                "lap/priority_min": (
                    float(self._priority_min[slots].min())
                    if slots.size else 0.0
                ),
                "lap/priority_max": (
                    float(self._priority_max[slots].max())
                    if slots.size else 0.0
                ),
            })
        result.update(self._last_sample_diag)
        return result
