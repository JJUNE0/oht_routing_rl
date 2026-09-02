import copy
import unittest
from collections import Counter
from dataclasses import replace

import numpy as np

from oht_routing.algorithms.rl.contextual_td7.replay_buffer import (
    ContextualReplayError,
    ContextualStepReplayBuffer,
    REPLAY_EVICTION_RANDOM,
    REPLAY_SAMPLING_SNAPSHOT,
    STATIC_PHYSICAL_FEATURE_INDICES,
    UINT16_PHYSICAL_FEATURE_INDICES,
    UINT8_PHYSICAL_FEATURE_INDICES,
)
from oht_routing.mdp.observation import CRITIC_EXTRA_DIM, GLOBAL_DIM
from test_contextual_observation import CONTROLLED_COUNT, make_topology
from test_contextual_replay import FakeObservationBuilder, make_snapshot


class ContextualReplayMemoryTests(unittest.TestCase):
    def setUp(self):
        self.topology = make_topology()
        self.builder = FakeObservationBuilder(self.topology)

    def test_ring_overwrite_and_stale_generation_rejection(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        stale = replay.push(make_snapshot(self.topology, 0))
        replay.push(make_snapshot(self.topology, 1))
        replay.push(make_snapshot(self.topology, 2))
        self.assertEqual(replay.size_env_steps, 2)
        self.assertEqual(replay.overwrite_count, 1)
        with self.assertRaises(ContextualReplayError):
            replay.validate_sample_keys([stale])
        self.assertEqual(replay.stale_key_reject_count, 1)

    def test_random_eviction_fills_before_seeded_non_fifo_replacement(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=4,
            seed=0,
            eviction_mode=REPLAY_EVICTION_RANDOM,
        )
        initial = [
            replay.push(make_snapshot(self.topology, step))
            for step in range(replay.capacity)
        ]

        self.assertEqual(
            [key.step_slot for key in initial], list(range(replay.capacity))
        )
        self.assertEqual([key.generation for key in initial], [1] * 4)
        self.assertEqual(replay.size_env_steps, replay.capacity)
        self.assertEqual(replay.overwrite_count, 0)

        reference_rng = np.random.default_rng()
        reference_rng.bit_generator.state = copy.deepcopy(
            replay.eviction_rng.bit_generator.state
        )
        expected_victim = int(reference_rng.integers(replay.capacity))
        self.assertNotEqual(expected_victim, 0)

        replacement = replay.push(make_snapshot(self.topology, 4))
        live_keys = {
            replay._transition_keys[slot]
            for slot in np.flatnonzero(replay._transition_valid)
        }
        expected_live = {(0, step) for step in range(4)}
        expected_live.remove((0, expected_victim))
        expected_live.add((0, 4))

        self.assertEqual(replacement.step_slot, expected_victim)
        self.assertEqual(live_keys, expected_live)
        self.assertIn((0, 0), live_keys)
        self.assertEqual(replay.size_env_steps, replay.capacity)
        self.assertEqual(replay.overwrite_count, 1)

    def test_random_eviction_stales_only_the_selected_victim(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=4,
            seed=0,
            eviction_mode=REPLAY_EVICTION_RANDOM,
        )
        initial = [
            replay.push(make_snapshot(self.topology, step))
            for step in range(replay.capacity)
        ]
        reference_rng = np.random.default_rng()
        reference_rng.bit_generator.state = copy.deepcopy(
            replay.eviction_rng.bit_generator.state
        )
        expected_victim = int(reference_rng.integers(replay.capacity))

        replacement = replay.push(make_snapshot(self.topology, 4))
        stale = initial[expected_victim]
        survivors = [
            key for index, key in enumerate(initial)
            if index != expected_victim
        ]

        self.assertEqual(replacement.step_slot, expected_victim)
        self.assertEqual(replacement.generation, stale.generation + 1)
        with self.assertRaises(ContextualReplayError):
            replay.validate_sample_keys([stale])
        replay.validate_sample_keys(survivors + [replacement])
        self.assertEqual(replay.stale_key_reject_count, 1)

    def test_random_eviction_keeps_capacity_and_state_references_valid(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=4,
            seed=23,
            sampling_mode=REPLAY_SAMPLING_SNAPSHOT,
            eviction_mode=REPLAY_EVICTION_RANDOM,
        )
        allocated_bytes = replay.storage_bytes
        for episode in range(24):
            replay.push(
                make_snapshot(
                    self.topology,
                    episode,
                    episode=episode,
                    done=True,
                )
            )
            self.assertEqual(
                replay.size_env_steps,
                min(episode + 1, replay.capacity),
            )
            self.assertEqual(replay.storage_bytes, allocated_bytes)

        valid_slots = replay._valid_transition_slots()
        self.assertEqual(valid_slots.size, replay.capacity)
        self.assertEqual(
            np.count_nonzero(replay._transition_valid), replay.capacity
        )
        for slot in valid_slots:
            transition_key = replay._transition_keys[int(slot)]
            self.assertIsNotNone(transition_key)
            episode, step = transition_key
            state_slot = int(replay._state_slot[slot])
            next_state_slot = int(replay._next_state_slot[slot])
            self.assertTrue(replay._state_valid[state_slot])
            self.assertTrue(replay._state_valid[next_state_slot])
            self.assertEqual(
                int(replay._state_generation[state_slot]),
                int(replay._state_gen_ref[slot]),
            )
            self.assertEqual(
                int(replay._state_generation[next_state_slot]),
                int(replay._next_state_gen_ref[slot]),
            )
            self.assertEqual(replay._state_keys[state_slot], (episode, step))
            self.assertEqual(
                replay._state_keys[next_state_slot], (episode, step + 1)
            )

        expected_refcounts = np.zeros(
            replay.state_capacity, dtype=np.uint8
        )
        np.add.at(
            expected_refcounts,
            replay._state_slot[valid_slots],
            1,
        )
        np.add.at(
            expected_refcounts,
            replay._next_state_slot[valid_slots],
            1,
        )
        np.testing.assert_array_equal(
            replay._state_refcount, expected_refcounts
        )
        free_slots = replay._free_state_slots[:replay._free_state_count]
        expected_free = np.flatnonzero(expected_refcounts == 0)
        np.testing.assert_array_equal(
            np.sort(free_slots), expected_free
        )

        batch = replay.sample(replay.capacity)
        live_episodes = {
            int(replay._episode_id[slot]) for slot in valid_slots
        }
        self.assertEqual(set(batch.episode_id.tolist()), live_episodes)
        np.testing.assert_array_equal(
            batch.critic_total_tat[:, 0].numpy(),
            1_000.0 + 10.0 * batch.env_step.numpy(),
        )

    def test_random_eviction_prunes_multi_episode_metadata(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=5,
            seed=31,
            eviction_mode=REPLAY_EVICTION_RANDOM,
        )
        total_pushes = 20
        for episode in range(total_pushes):
            replay.push(
                make_snapshot(
                    self.topology,
                    episode,
                    episode=episode,
                    done=True,
                )
            )

        live_keys = [
            replay._transition_keys[int(slot)]
            for slot in replay._valid_transition_slots()
        ]
        live_counts = Counter(key[0] for key in live_keys)
        self.assertEqual(replay._episode_transition_counts, dict(live_counts))
        self.assertEqual(
            set(replay._episode_first_step), set(live_counts)
        )
        for episode in live_counts:
            self.assertEqual(replay._episode_first_step[episode], episode)
        self.assertEqual(replay.overwrite_count, total_pushes - replay.capacity)

    def test_random_eviction_is_seeded_and_does_not_advance_sampling_rng(self):
        first = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=4,
            seed=47,
            eviction_mode=REPLAY_EVICTION_RANDOM,
        )
        second = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=4,
            seed=47,
            eviction_mode=REPLAY_EVICTION_RANDOM,
        )
        sampling_rng_state = copy.deepcopy(first.rng.bit_generator.state)
        for step in range(20):
            snapshot = make_snapshot(self.topology, step)
            first.push(snapshot)
            second.push(snapshot)
            self.assertEqual(
                first.rng.bit_generator.state, sampling_rng_state
            )

        self.assertEqual(first._transition_keys, second._transition_keys)
        np.testing.assert_array_equal(
            first._transition_generation, second._transition_generation
        )
        first_batch = first.sample(64)
        second_batch = second.sample(64)
        self.assertEqual(first_batch.sample_keys, second_batch.sample_keys)
        np.testing.assert_array_equal(
            first_batch.reward.numpy(), second_batch.reward.numpy()
        )

        sampled = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=4,
            seed=71,
            eviction_mode=REPLAY_EVICTION_RANDOM,
        )
        unsampled = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=4,
            seed=71,
            eviction_mode=REPLAY_EVICTION_RANDOM,
        )
        for step in range(20):
            snapshot = make_snapshot(self.topology, step)
            sampled.push(snapshot)
            unsampled.push(snapshot)
            sampled.sample(1)
        self.assertEqual(
            sampled._transition_keys, unsampled._transition_keys
        )
        np.testing.assert_array_equal(
            sampled._transition_generation,
            unsampled._transition_generation,
        )

    def test_invalid_random_push_does_not_evict_or_advance_rng(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=2,
            seed=53,
            eviction_mode=REPLAY_EVICTION_RANDOM,
        )
        for step in range(replay.capacity):
            replay.push(make_snapshot(self.topology, step))

        transition_keys = replay._transition_keys.copy()
        transition_generation = replay._transition_generation.copy()
        state_refcount = replay._state_refcount.copy()
        free_state_count = replay._free_state_count
        overwrite_count = replay.overwrite_count
        push_count = replay.push_count
        eviction_rng_state = copy.deepcopy(
            replay.eviction_rng.bit_generator.state
        )
        invalid = make_snapshot(self.topology, 2)
        physical = invalid.physical_local_state.copy()
        physical[0, UINT8_PHYSICAL_FEATURE_INDICES[0]] = 1.5

        with self.assertRaisesRegex(
            ContextualReplayError, "must be exactly integral"
        ):
            replay.push(replace(invalid, physical_local_state=physical))

        self.assertEqual(replay._transition_keys, transition_keys)
        np.testing.assert_array_equal(
            replay._transition_generation, transition_generation
        )
        np.testing.assert_array_equal(replay._state_refcount, state_refcount)
        self.assertEqual(replay._free_state_count, free_state_count)
        self.assertEqual(replay.overwrite_count, overwrite_count)
        self.assertEqual(replay.push_count, push_count)
        self.assertEqual(
            replay.eviction_rng.bit_generator.state, eviction_rng_state
        )

    def test_random_eviction_rejects_stacked_replay(self):
        with self.assertRaisesRegex(
            ValueError, "random replay eviction.*num_stacks=1"
        ):
            ContextualStepReplayBuffer(
                self.topology,
                self.builder,
                capacity_env_steps=4,
                eviction_mode=REPLAY_EVICTION_RANDOM,
                num_stacks=2,
            )

    def test_multi_episode_has_no_linkage(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=8
        )
        replay.push(make_snapshot(self.topology, 0, episode=0))
        replay.push(make_snapshot(self.topology, 0, episode=1))
        batch = replay.sample(200)
        self.assertTrue(set(batch.episode_id.tolist()).issubset({0, 1}))
        self.assertEqual(
            replay.diagnostics()["replay/episode_crossing_count"], 0
        )

    def test_full_capacity_survives_one_transition_per_episode(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=4,
            sampling_mode=REPLAY_SAMPLING_SNAPSHOT,
        )
        for episode in range(4):
            replay.push(
                make_snapshot(
                    self.topology, 0, episode=episode, done=True
                )
            )
        self.assertEqual(replay.size_env_steps, 4)
        self.assertEqual(replay.state_capacity, 2 * replay.capacity)
        sampled = replay.sample(4)
        self.assertEqual(set(sampled.episode_id.tolist()), set(range(4)))

    def test_storage_estimate_and_actual_allocation(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=3
        )
        estimated = replay.estimate_capacity_bytes(3)
        actual = replay.storage_bytes
        self.assertGreater(estimated, 0)
        self.assertGreater(actual, 0)
        self.assertLess(abs(actual - estimated) / estimated, 0.01)
        for capacity in (1_000, 5_000, 10_000, 45_000):
            self.assertGreater(replay.estimate_capacity_bytes(capacity), 0)

        estimate_100k = replay.estimate_capacity_bytes(
            100_000, lap_enabled=True
        )
        estimate_100k_gib = estimate_100k / (1024.0 ** 3)
        self.assertLess(estimate_100k_gib, 20.0)
        self.assertAlmostEqual(estimate_100k_gib, 16.772765, places=5)
        estimate_100k_no_lap_gib = replay.estimate_capacity_bytes(
            100_000, lap_enabled=False
        ) / (1024.0 ** 3)
        self.assertAlmostEqual(
            estimate_100k_no_lap_gib, 15.840325, places=5
        )

    def test_packed_storage_dtypes_match_capacity_estimate_contract(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=3,
            lap_enabled=True,
        )
        self.assertEqual(STATIC_PHYSICAL_FEATURE_INDICES, (0, 1, 2, 3))
        self.assertEqual(
            UINT8_PHYSICAL_FEATURE_INDICES, (4, 6, 7, 8, 9, 10, 11, 13)
        )
        self.assertEqual(UINT16_PHYSICAL_FEATURE_INDICES, (5, 12))
        self.assertEqual(replay._physical_static.dtype, np.float32)
        self.assertEqual(replay._physical_uint8.dtype, np.uint8)
        self.assertEqual(replay._physical_uint16.dtype, np.uint16)
        self.assertEqual(replay._global.dtype, np.float32)
        self.assertEqual(replay._critic_total_tat.dtype, np.float32)
        self.assertEqual(replay._previous_applied_action.dtype, np.int16)
        self.assertEqual(replay._policy_action.dtype, np.int16)
        self.assertEqual(replay._applied_action.dtype, np.int16)
        self.assertEqual(replay._reward.dtype, np.float32)
        self.assertEqual(replay._priority.dtype, np.float16)
        self.assertEqual(
            replay._physical_uint8.shape,
            (2 * replay.capacity, len(self.topology.all_rail_ids), 8),
        )
        self.assertEqual(
            replay._physical_uint16.shape,
            (2 * replay.capacity, len(self.topology.all_rail_ids), 2),
        )
        self.assertEqual(
            replay._global.shape,
            (2 * replay.capacity, GLOBAL_DIM),
        )
        self.assertEqual(
            replay._critic_total_tat.shape,
            (2 * replay.capacity, CRITIC_EXTRA_DIM),
        )

    def test_repeated_overwrite_has_fixed_storage(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=4
        )
        before = replay.storage_bytes
        for step in range(100):
            replay.push(make_snapshot(self.topology, step))
            if step % 10 == 0:
                replay.sample(32)
        self.assertEqual(replay.storage_bytes, before)
        self.assertEqual(replay.size_env_steps, 4)
        self.assertEqual(
            replay.diagnostics()["replay/size_logical_transitions"],
            4 * CONTROLLED_COUNT,
        )


if __name__ == "__main__":
    unittest.main()
