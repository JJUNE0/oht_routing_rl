import unittest

from cocel_rl.algorithms.contextual_td7.replay_buffer import (
    ContextualReplayError,
    ContextualStepReplayBuffer,
)
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
            self.topology, self.builder, capacity_env_steps=4
        )
        for episode in range(4):
            replay.push(make_snapshot(self.topology, 0, episode=episode))
        self.assertEqual(replay.size_env_steps, 4)

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
