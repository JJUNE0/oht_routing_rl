import unittest

import numpy as np

from cocel_rl.algorithms.contextual_td7.replay_buffer import (
    ContextualReplayError,
    ContextualStepReplayBuffer,
    ReplaySampleKey,
)
from test_contextual_observation import make_topology
from test_contextual_replay import FakeObservationBuilder, make_snapshot


class ContextualLAPTests(unittest.TestCase):
    def replay(self, seed=1):
        topology = make_topology()
        return ContextualStepReplayBuffer(
            topology, FakeObservationBuilder(topology),
            capacity_env_steps=2, seed=seed, lap_enabled=True,
            lap_alpha=0.4, lap_min_priority=1.0,
        )

    def test_higher_td_error_samples_more_and_minimum_applies(self):
        replay = self.replay(seed=8)
        replay.push(make_snapshot(replay.topology, 0))
        generation = int(replay._transition_generation[0])
        hot = ReplaySampleKey(0, generation, 7)
        cold = ReplaySampleKey(0, generation, 8)
        replay.update_priorities([hot, cold], [1e6, 0.0])
        self.assertEqual(replay._priority[0, 8], 1.0)
        counts = {7: 0, 8: 0}
        for _ in range(30):
            batch = replay.sample(512)
            for key in batch.sample_keys:
                if key.controlled_row in counts:
                    counts[key.controlled_row] += 1
        self.assertGreater(counts[7], counts[8] * 10)

    def test_equal_priority_is_uniform_and_seed_reproducible(self):
        first, second = self.replay(22), self.replay(22)
        first.push(make_snapshot(first.topology, 0))
        second.push(make_snapshot(second.topology, 0))
        self.assertEqual(first.sample(100).sample_keys,
                         second.sample(100).sample_keys)

    def test_new_transition_uses_current_max_and_overwrite_resets(self):
        replay = self.replay()
        replay.push(make_snapshot(replay.topology, 0))
        key = ReplaySampleKey(0, int(replay._transition_generation[0]), 0)
        replay.update_priorities([key], [1e5])
        maximum = replay.max_priority
        replay.push(make_snapshot(replay.topology, 1))
        self.assertTrue(np.all(replay._priority[1] == maximum))
        replay.push(make_snapshot(replay.topology, 2))
        self.assertTrue(np.all(replay._priority[0] == maximum))

    def test_stale_generation_priority_update_rejected(self):
        replay = self.replay()
        stale = replay.push(make_snapshot(replay.topology, 0))
        replay.push(make_snapshot(replay.topology, 1))
        replay.push(make_snapshot(replay.topology, 2))
        with self.assertRaises(ContextualReplayError):
            replay.update_priorities([stale], [2.0])

    def test_uniform_mode_priority_update_is_explicit_error(self):
        topology = make_topology()
        replay = ContextualStepReplayBuffer(
            topology, FakeObservationBuilder(topology),
            capacity_env_steps=2, lap_enabled=False,
        )
        key = replay.push(make_snapshot(topology, 0))
        with self.assertRaises(NotImplementedError):
            replay.update_priorities([key], [2.0])
        self.assertIsNone(replay._priority)


if __name__ == "__main__":
    unittest.main()
