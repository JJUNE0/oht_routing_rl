import unittest

import numpy as np

from oht_routing.algorithms.rl.contextual_td7.replay_buffer import (
    ContextualReplayError,
    ContextualStepReplayBuffer,
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
        self.assertAlmostEqual(estimate_100k_gib, 16.772020, places=5)
        estimate_100k_no_lap_gib = replay.estimate_capacity_bytes(
            100_000, lap_enabled=False
        ) / (1024.0 ** 3)
        self.assertAlmostEqual(
            estimate_100k_no_lap_gib, 15.839580, places=5
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
