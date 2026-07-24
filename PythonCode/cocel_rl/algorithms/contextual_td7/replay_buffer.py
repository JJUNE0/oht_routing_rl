"""Memory-efficient state-ring replay for contextual per-rail transitions."""

from __future__ import annotations

import time
from dataclasses import fields

import numpy as np
import torch

from contextual_action import ACTION_VERSION
from contextual_observation import (
    GLOBAL_DIM,
    LOCAL_DIM,
    OBSERVATION_VERSION,
)
from contextual_reward import REWARD_VERSION
from contextual_topology import ContextualTopology
from contextual_transition import CompletedContextualTransition

from .replay_types import (
    ContextualReplayBatch,
    ContextualStepSnapshot,
    ReplaySampleKey,
)


class ContextualReplayError(RuntimeError):
    pass


REPLAY_VERSION = "contextual_step_snapshot_uniform_v1"
LAP_VERSION = "contextual_snapshot_lap_hierarchical_v1"


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
        done=bool(transition.done),
        env_step=int(transition.env_step),
        next_env_step=int(transition.next_env_step),
        episode_id=int(transition.episode_id),
        topology_hash=transition.topology_hash,
        mapping_hash=transition.mapping_hash,
        observation_version=OBSERVATION_VERSION,
        reward_version=REWARD_VERSION,
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
    ):
        if int(capacity_env_steps) <= 0:
            raise ValueError("capacity_env_steps must be positive")
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
        self._priority = (
            np.zeros((self.capacity, self.controlled_count), np.float32)
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
            (snapshot.reward_version, REWARD_VERSION, "reward"),
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

    def _store_state(self, key, physical, global_state) -> tuple[int, int]:
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
        self._state_generation[slot] = generation
        self._state_valid[slot] = True
        self._state_keys[slot] = key
        self._key_to_state[key] = (slot, generation)
        return slot, generation

    def push(self, snapshot: ContextualStepSnapshot) -> ReplaySampleKey:
        self._validate_snapshot(snapshot)
        state_key = (int(snapshot.episode_id), int(snapshot.env_step))
        next_key = (int(snapshot.episode_id), int(snapshot.next_env_step))
        if state_key[0] != next_key[0]:
            self.episode_crossing_count += 1
            raise ContextualReplayError("episode-crossing snapshot")
        state_slot, state_gen = self._store_state(
            state_key, snapshot.physical_local_state, snapshot.global_state
        )
        next_slot, next_gen = self._store_state(
            next_key, snapshot.next_physical_local_state,
            snapshot.next_global_state
        )

        slot = self._transition_cursor
        self._transition_cursor = (slot + 1) % self.capacity
        if self._transition_valid[slot]:
            self.overwrite_count += 1
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
        if self.lap_enabled:
            self._priority[slot].fill(self.max_priority)
        self.push_count += 1
        return ReplaySampleKey(slot, generation, 0)

    def push_transition(
        self, transition: CompletedContextualTransition
    ) -> ReplaySampleKey:
        return self.push(
            snapshot_from_transition(
                transition, action_version=self.action_version
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
        return candidates[state_ok & next_ok]

    def validate_sample_keys(self, keys) -> None:
        for key in keys:
            valid = (
                0 <= key.step_slot < self.capacity
                and self._transition_valid[key.step_slot]
                and self._transition_generation[key.step_slot] == key.generation
                and 0 <= key.controlled_row < self.controlled_count
                and key.step_slot in self._valid_transition_slots()
            )
            if not valid:
                self.stale_key_reject_count += 1
                raise ContextualReplayError(f"stale replay key: {key}")

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
        for key, value in zip(sample_keys, values):
            self._priority[key.step_slot, key.controlled_row] = value
        valid_slots = self._valid_transition_slots()
        self.max_priority = (
            max(
                self.lap_min_priority,
                float(self._priority[valid_slots].max(initial=0.0)),
            )
            if valid_slots.size else self.lap_min_priority
        )
        self.priority_update_count += 1

    @staticmethod
    def _readonly_normalize(normalizer, values, name):
        # normalize() is read-only; update() is intentionally never called.
        return normalizer.normalize(values, name=name)

    def sample(
        self, batch_size: int, *, device: str | torch.device = "cpu"
    ) -> ContextualReplayBatch:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        valid_slots = self._valid_transition_slots()
        if valid_slots.size == 0:
            raise ContextualReplayError("cannot sample an empty replay")
        started = time.perf_counter()
        # With replacement is explicit: any positive batch is allowed once one
        # environment snapshot exists.
        if self.lap_enabled:
            step_sums = self._priority[valid_slots].sum(axis=1, dtype=np.float64)
            total_priority = float(step_sums.sum())
            if not np.isfinite(total_priority) or total_priority <= 0:
                raise ContextualReplayError("invalid LAP total priority")
            transition_slots = self.rng.choice(
                valid_slots, size=int(batch_size), replace=True,
                p=step_sums / total_priority,
            )
            controlled_rows = np.empty(int(batch_size), np.int64)
            sample_probabilities = np.empty(int(batch_size), np.float64)
            for index, slot in enumerate(transition_slots):
                row_priority = self._priority[slot].astype(np.float64)
                row_sum = float(row_priority.sum())
                controlled_rows[index] = self.rng.choice(
                    self.controlled_count, p=row_priority / row_sum
                )
                sample_probabilities[index] = (
                    self._priority[slot, controlled_rows[index]]
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
        state_slots = self._state_slot[transition_slots]
        next_slots = self._next_state_slot[transition_slots]
        state_global_raw = self._global[state_slots]
        next_global_raw = self._global[next_slots]
        global_norm = self._readonly_normalize(
            self.observation_builder.global_normalizer,
            state_global_raw, "replay_state_global"
        )
        next_global_norm = self._readonly_normalize(
            self.observation_builder.global_normalizer,
            next_global_raw, "replay_next_global"
        )
        center_rows = self._center_rows[controlled_rows]
        incoming_rows = self._incoming_rows[controlled_rows]
        outgoing_rows = self._outgoing_rows[controlled_rows]
        current_center_raw = self._physical[state_slots, center_rows]
        current_incoming_raw = self._physical[
            state_slots[:, None], incoming_rows
        ]
        current_outgoing_raw = self._physical[
            state_slots[:, None], outgoing_rows
        ]
        next_center_raw = self._physical[next_slots, center_rows]
        next_incoming_raw = self._physical[
            next_slots[:, None], incoming_rows
        ]
        next_outgoing_raw = self._physical[
            next_slots[:, None], outgoing_rows
        ]
        local_normalizer = self.observation_builder.local_normalizer
        center = self._readonly_normalize(
            local_normalizer, current_center_raw, "replay_center_local"
        )
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
            local_normalizer, next_center_raw, "replay_next_center_local"
        )
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

        arrays = {
            "center_local": center,
            "incoming_local": incoming,
            "outgoing_local": outgoing,
            "incoming_relation": self.observation_builder._incoming_relation[
                controlled_rows
            ],
            "outgoing_relation": self.observation_builder._outgoing_relation[
                controlled_rows
            ],
            "global_state": global_norm,
            "policy_action": self._policy_action[
                transition_slots, controlled_rows
            ][:, None],
            "applied_action": self._applied_action[
                transition_slots, controlled_rows
            ][:, None],
            "reward": self._reward[transition_slots, controlled_rows][:, None],
            "next_center_local": next_center,
            "next_incoming_local": next_incoming,
            "next_outgoing_local": next_outgoing,
            "next_incoming_relation": self.observation_builder._incoming_relation[
                controlled_rows
            ],
            "next_outgoing_relation": self.observation_builder._outgoing_relation[
                controlled_rows
            ],
            "next_global_state": next_global_norm,
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
                array = np.ascontiguousarray(values, dtype=np.int64)
            else:
                array = np.ascontiguousarray(values, dtype=np.float32)
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
        state = (2 * c) * (physical_count * LOCAL_DIM + GLOBAL_DIM) * 4
        vectors = c * controlled_count * 3 * 4
        metadata = c * (7 * 8 + 4 + 1) + (2 * c) * (8 + 1)
        static_mapping = controlled_count * (1 + 2 * 10) * 8
        priority = c * controlled_count * 4 if lap_enabled else 0
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
        }
        if self.lap_enabled:
            slots = self._valid_transition_slots()
            active = self._priority[slots].reshape(-1)
            result.update({
                "lap/priority_mean": float(active.mean()) if active.size else 0.0,
                "lap/priority_std": float(active.std()) if active.size else 0.0,
                "lap/priority_min": float(active.min()) if active.size else 0.0,
                "lap/priority_max": float(active.max()) if active.size else 0.0,
            })
        result.update(self._last_sample_diag)
        return result
