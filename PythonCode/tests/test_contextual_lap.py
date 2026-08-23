import copy
import unittest

import numpy as np

from oht_routing.algorithms.rl.contextual_td7.replay_buffer import (
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

    def test_vectorized_rail_draw_matches_former_grouped_rng_contract(self):
        replay = self.replay(seed=27)
        for step in range(2):
            replay.push(make_snapshot(replay.topology, step))
        keys = []
        priorities = []
        for slot in range(2):
            generation = int(replay._transition_generation[slot])
            for row, priority in ((1, 2.0), (7, 30.0), (31, 5.0)):
                keys.append(ReplaySampleKey(slot, generation, row))
                priorities.append(priority + slot)
        replay.update_priorities(keys, priorities)

        reference_rng = np.random.default_rng()
        reference_rng.bit_generator.state = copy.deepcopy(
            replay.rng.bit_generator.state
        )
        valid_slots = replay._valid_transition_slots()
        step_sums = replay._priority_sum[valid_slots]
        total_priority = float(step_sums.sum())
        transition_slots = reference_rng.choice(
            valid_slots,
            size=64,
            replace=True,
            p=step_sums / total_priority,
        )
        expected_rows = np.empty(64, np.int64)
        unique_slots, inverse = np.unique(
            transition_slots, return_inverse=True
        )
        for group, slot in enumerate(unique_slots):
            batch_rows = np.flatnonzero(inverse == group)
            cdf = np.cumsum(replay._priority[slot], dtype=np.float64)
            thresholds = reference_rng.random(batch_rows.size) * cdf[-1]
            expected_rows[batch_rows] = np.searchsorted(
                cdf, thresholds, side="right"
            )

        batch = replay.sample(64)
        actual = [
            (key.step_slot, key.controlled_row)
            for key in batch.sample_keys
        ]
        expected = list(zip(
            transition_slots.tolist(), expected_rows.tolist()
        ))
        self.assertEqual(actual, expected)

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

    def test_cached_priority_statistics_match_storage_and_duplicates_use_last(self):
        replay = self.replay()
        replay.push(make_snapshot(replay.topology, 0))
        self.assertEqual(replay._priority.dtype, np.float16)
        self.assertEqual(replay._priority_min.dtype, np.float16)
        self.assertEqual(replay._priority_max.dtype, np.float16)
        self.assertEqual(replay._priority_sum.dtype, np.float64)
        self.assertEqual(replay._priority_sq_sum.dtype, np.float64)
        generation = int(replay._transition_generation[0])
        key = ReplaySampleKey(0, generation, 3)
        replay.update_priorities([key, key], [100.0, 10.0])
        expected = float(np.float16(max(
            10.0 ** replay.lap_alpha, replay.lap_min_priority
        )))
        self.assertEqual(float(replay._priority[0, 3]), expected)
        self.assertEqual(
            replay._priority_sum[0],
            replay._priority[0].sum(dtype=np.float64),
        )
        self.assertEqual(
            replay._priority_sq_sum[0],
            np.square(replay._priority[0].astype(np.float64)).sum(),
        )
        diagnostics = replay.diagnostics()
        active = replay._priority[
            replay._valid_transition_slots()
        ].astype(np.float64)
        self.assertAlmostEqual(
            diagnostics["lap/priority_mean"], float(active.mean()), places=6
        )
        self.assertAlmostEqual(
            diagnostics["lap/priority_std"], float(active.std()), places=6
        )

    def test_priority_overflow_is_rejected_before_float16_storage(self):
        replay = self.replay()
        replay.push(make_snapshot(replay.topology, 0))
        key = ReplaySampleKey(
            0, int(replay._transition_generation[0]), 0
        )
        with self.assertRaisesRegex(
            ContextualReplayError, "exceeds finite float16 replay storage"
        ):
            replay.update_priorities([key], [1e20])

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
