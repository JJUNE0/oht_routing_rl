import unittest
from dataclasses import replace

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7.replay_buffer import (
    ContextualReplayError,
    ContextualStepReplayBuffer,
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    snapshot_from_transition,
)
from oht_routing.algorithms.rl.contextual_td7.replay_types import ContextualStepSnapshot
from oht_routing.mdp.observation import (
    GLOBAL_DIM,
    LOCAL_DIM,
)
from oht_routing.mdp.reward.config import REWARD_VERSION
from test_contextual_observation import (
    BOUNDARY_IDS,
    CONTROLLED_COUNT,
    PHYSICAL_COUNT,
    make_topology,
)
from oht_routing.mdp.reward.builder import ContextualRewardBuilder, ContextualRewardConfig
from oht_routing.mdp.transition import ContextualTransitionAligner
from oht_routing.version import CONTEXTUAL_VERSION
from test_contextual_reward import reward_client
from test_contextual_transition import observation


class IdentityNormalizer:
    def __init__(self):
        self.count = 123
        self.update_calls = 7

    def normalize(self, values, *, name):
        return np.ascontiguousarray(np.asarray(values, np.float32))


class FakeObservationBuilder:
    def __init__(self, topology):
        self.local_normalizer = IdentityNormalizer()
        self.global_normalizer = IdentityNormalizer()
        shape = topology.incoming_neighbor_ids.shape + (2,)
        self._incoming_relation = np.arange(
            np.prod(shape), dtype=np.float32
        ).reshape(shape)
        self._outgoing_relation = self._incoming_relation + 100.0


def make_snapshot(topology, step, episode=0, done=False):
    physical = np.empty((PHYSICAL_COUNT, LOCAL_DIM), np.float32)
    for row, rail_id in enumerate(topology.all_rail_ids):
        physical[row] = float(rail_id) + np.arange(LOCAL_DIM) / 100
    rows = np.arange(CONTROLLED_COUNT, dtype=np.float32)
    return ContextualStepSnapshot(
        physical_local_state=physical + step,
        global_state=np.arange(GLOBAL_DIM, dtype=np.float32) + step,
        previous_applied_action=(rows / CONTROLLED_COUNT * 0.25)[:, None],
        policy_action=(rows / CONTROLLED_COUNT)[:, None],
        applied_action=(rows / CONTROLLED_COUNT * 0.25)[:, None],
        reward=rows + step * 10,
        next_physical_local_state=physical + (step + 1),
        next_global_state=np.arange(GLOBAL_DIM, dtype=np.float32) + step + 1.0,
        next_previous_applied_action=(
            rows / CONTROLLED_COUNT * 0.25
        )[:, None],
        done=done,
        env_step=step,
        next_env_step=step + 1,
        episode_id=episode,
        topology_hash=topology.topology_hash,
        mapping_hash=topology.mapping_hash,
        version=CONTEXTUAL_VERSION,
        reward_version=REWARD_VERSION,
    )


class ContextualReplayTests(unittest.TestCase):
    def setUp(self):
        self.topology = make_topology()
        self.builder = FakeObservationBuilder(self.topology)

    def test_one_push_is_4996_logical_rows_and_sample_contract(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=4, seed=5
        )
        replay.push(make_snapshot(self.topology, 0, done=True))
        self.assertEqual(replay.size_env_steps, 1)
        self.assertEqual(
            replay.diagnostics()["replay/size_logical_transitions"],
            CONTROLLED_COUNT,
        )
        batch = replay.sample(16)
        expected_shapes = {
            "center_local": (16, 8),
            "incoming_local": (16, 10, 8),
            "outgoing_local": (16, 10, 8),
            "incoming_relation": (16, 10, 2),
            "outgoing_relation": (16, 10, 2),
            "global_state": (16, 6),
            "previous_applied_action": (16, 1),
            "policy_action": (16, 1),
            "applied_action": (16, 1),
            "reward": (16, 1),
            "next_center_local": (16, 8),
            "next_incoming_local": (16, 10, 8),
            "next_outgoing_local": (16, 10, 8),
            "next_previous_applied_action": (16, 1),
            "done": (16, 1),
            "controlled_rail_id": (16,),
        }
        for name, shape in expected_shapes.items():
            value = getattr(batch, name)
            self.assertEqual(tuple(value.shape), shape)
            self.assertTrue(value.is_contiguous())
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.all(batch.done == 1))

        for index, key in enumerate(batch.sample_keys):
            row = key.controlled_row
            rail_id = int(self.topology.controlled_rail_ids[row])
            physical_row = int(
                self.topology.controlled_row_to_physical_index[row]
            )
            self.assertEqual(int(batch.controlled_rail_id[index]), rail_id)
            self.assertAlmostEqual(
                float(batch.center_local[index, 0]), float(physical_row)
            )
            self.assertAlmostEqual(
                float(batch.policy_action[index, 0]),
                row / CONTROLLED_COUNT,
            )
            self.assertAlmostEqual(float(batch.reward[index, 0]), float(row))

    def test_neighbor_gather_uses_id_mapping_and_boundary_can_be_source(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2, seed=1
        )
        replay.push(make_snapshot(self.topology, 0))
        # Find a sampled row 0, whose first incoming neighbor is boundary 3250.
        found = False
        for _ in range(100):
            batch = replay.sample(256)
            for i, key in enumerate(batch.sample_keys):
                if key.controlled_row == 0:
                    self.assertEqual(
                        float(batch.incoming_local[i, 0, 0]),
                        float(BOUNDARY_IDS[0]),
                    )
                    self.assertNotIn(
                        int(batch.controlled_rail_id[i]), BOUNDARY_IDS
                    )
                    found = True
                    break
            if found:
                break
        self.assertTrue(found)

    def test_sampling_does_not_update_normalizers(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        replay.push(make_snapshot(self.topology, 0))
        before = (
            self.builder.local_normalizer.count,
            self.builder.local_normalizer.update_calls,
            self.builder.global_normalizer.count,
            self.builder.global_normalizer.update_calls,
        )
        replay.sample(32)
        after = (
            self.builder.local_normalizer.count,
            self.builder.local_normalizer.update_calls,
            self.builder.global_normalizer.count,
            self.builder.global_normalizer.update_calls,
        )
        self.assertEqual(before, after)

    def test_completed_transition_converts_to_exactly_one_snapshot_push(self):
        reward_builder = ContextualRewardBuilder(
            self.topology,
            ContextualRewardConfig(),
        )
        aligner = ContextualTransitionAligner(self.topology, reward_builder)
        zeros = np.zeros((CONTROLLED_COUNT, 1), np.float32)
        costs = np.ones(PHYSICAL_COUNT, np.float32)
        aligner.advance(
            observation=observation(self.topology, 0),
            pclient=reward_client(), controlled_action=zeros,
            applied_action=zeros, baseline_cost=costs, final_cost=costs,
            env_step=0, episode_id=0,
        )
        completed = aligner.advance(
            observation=observation(self.topology, 1),
            pclient=reward_client(), controlled_action=zeros,
            applied_action=zeros, baseline_cost=costs, final_cost=costs,
            env_step=1, episode_id=0,
        )
        snapshot = snapshot_from_transition(completed)
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        replay.push(snapshot)
        self.assertEqual(replay.push_count, 1)
        self.assertEqual(replay.size_env_steps, 1)

    def test_hash_mapping_version_nonfinite_and_step_fail_fast(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        base = make_snapshot(self.topology, 0)
        for changed in (
            replace(base, topology_hash="wrong"),
            replace(base, mapping_hash="wrong"),
            replace(base, version="wrong"),
            replace(base, reward_version="wrong"),
            replace(base, next_env_step=3),
            replace(
                base,
                next_previous_applied_action=np.zeros_like(
                    base.next_previous_applied_action
                ),
            ),
        ):
            with self.assertRaises(ContextualReplayError):
                replay.push(changed)
        bad_reward = base.reward.copy()
        bad_reward[0] = np.nan
        with self.assertRaises(ContextualReplayError):
            replay.push(replace(base, reward=bad_reward))

    def test_replay_is_locked_to_reward_n(self):
        with self.assertRaisesRegex(ValueError, "only reward_version='N'"):
            ContextualStepReplayBuffer(
                self.topology,
                self.builder,
                capacity_env_steps=2,
                reward_version="T",
            )
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        snapshot = make_snapshot(self.topology, 0)
        replay.push(snapshot)
        self.assertEqual(replay.size_env_steps, 1)

    def test_empty_sampling_and_uniform_priority_contract(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        with self.assertRaises(ContextualReplayError):
            replay.sample(1)
        key = replay.push(make_snapshot(self.topology, 0))
        with self.assertRaises(NotImplementedError):
            replay.update_priorities([key], [1.0])

    def test_cpu_dtype_and_fixed_seed_reproducibility(self):
        first = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=4, seed=91
        )
        second = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=4, seed=91
        )
        for step in range(3):
            first.push(make_snapshot(self.topology, step))
            second.push(make_snapshot(self.topology, step))
        a, b = first.sample(32), second.sample(32)
        self.assertEqual(a.sample_keys, b.sample_keys)
        self.assertTrue(torch.equal(a.reward, b.reward))
        self.assertEqual(a.reward.dtype, torch.float32)
        self.assertEqual(a.env_step.dtype, torch.int64)
        self.assertEqual(a.reward.device.type, "cpu")

    def test_snapshot_sampling_returns_every_rail_for_distinct_steps(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=4,
            seed=37,
            sampling_mode=REPLAY_SAMPLING_SNAPSHOT,
        )
        for step in range(3):
            replay.push(make_snapshot(self.topology, step))

        batch = replay.sample(2)
        expected_rows = 2 * CONTROLLED_COUNT
        self.assertEqual(tuple(batch.center_local.shape), (expected_rows, 8))
        self.assertEqual(
            tuple(batch.incoming_local.shape), (expected_rows, 10, 8)
        )
        self.assertEqual(len(batch.sample_keys), expected_rows)

        sampled_steps = batch.env_step.numpy().reshape(2, CONTROLLED_COUNT)
        sampled_rails = batch.controlled_rail_id.numpy().reshape(
            2, CONTROLLED_COUNT
        )
        self.assertEqual(len(np.unique(sampled_steps[:, 0])), 2)
        for index in range(2):
            self.assertTrue(np.all(sampled_steps[index] == sampled_steps[index, 0]))
            self.assertTrue(np.array_equal(
                sampled_rails[index], self.topology.controlled_rail_ids
            ))
            self.assertEqual(
                [key.controlled_row for key in batch.sample_keys[
                    index * CONTROLLED_COUNT:(index + 1) * CONTROLLED_COUNT
                ]],
                list(range(CONTROLLED_COUNT)),
            )

    def test_snapshot_sampling_requires_enough_steps_and_disables_lap(self):
        with self.assertRaises(ValueError):
            ContextualStepReplayBuffer(
                self.topology,
                self.builder,
                capacity_env_steps=2,
                lap_enabled=True,
                sampling_mode=REPLAY_SAMPLING_SNAPSHOT,
            )
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=2,
            sampling_mode=REPLAY_SAMPLING_SNAPSHOT,
        )
        replay.push(make_snapshot(self.topology, 0))
        with self.assertRaises(ContextualReplayError):
            replay.sample(2)

    def test_random_rail_sampling_uses_flat_unique_logical_pool(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=2,
            seed=73,
            sampling_mode=REPLAY_SAMPLING_RANDOM_RAIL,
        )
        replay.push(make_snapshot(self.topology, 0))

        batch = replay.sample(1_024)
        pairs = [
            (key.step_slot, key.controlled_row)
            for key in batch.sample_keys
        ]
        self.assertEqual(len(pairs), 1_024)
        self.assertEqual(len(set(pairs)), 1_024)
        self.assertTrue(torch.all(batch.env_step == 0))
        self.assertEqual(
            len(torch.unique(batch.controlled_rail_id)), 1_024
        )
        for index, key in enumerate(batch.sample_keys):
            self.assertAlmostEqual(
                float(batch.reward[index, 0]),
                float(key.controlled_row),
            )

    def test_random_rail_sampling_capacity_and_lap_contract(self):
        with self.assertRaises(ValueError):
            ContextualStepReplayBuffer(
                self.topology,
                self.builder,
                capacity_env_steps=1,
                lap_enabled=True,
                sampling_mode=REPLAY_SAMPLING_RANDOM_RAIL,
            )
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=1,
            sampling_mode=REPLAY_SAMPLING_RANDOM_RAIL,
        )
        replay.push(make_snapshot(self.topology, 0))
        with self.assertRaises(ContextualReplayError):
            replay.sample(CONTROLLED_COUNT + 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_device_sample_smoke(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        replay.push(make_snapshot(self.topology, 0))
        batch = replay.sample(4, device="cuda")
        self.assertEqual(batch.reward.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
