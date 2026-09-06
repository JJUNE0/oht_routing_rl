import unittest
from dataclasses import replace

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7.replay_buffer import (
    ACTION_FIXED_POINT_MAX_ABS_ERROR,
    ContextualReplayError,
    ContextualStepReplayBuffer,
    REPLAY_EVICTION_FIFO,
    REPLAY_EVICTION_RANDOM,
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
    snapshot_from_transition,
)
from oht_routing.algorithms.rl.contextual_td7.replay_types import ContextualStepSnapshot
from oht_routing.mdp.observation import (
    CRITIC_EXTRA_DIM,
    GLOBAL_DIM,
    LOCAL_PHYSICAL_DIM,
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
        self.critic_normalizer = IdentityNormalizer()
        shape = topology.incoming_neighbor_ids.shape + (2,)
        self._incoming_relation = np.arange(
            np.prod(shape), dtype=np.float32
        ).reshape(shape)
        self._outgoing_relation = self._incoming_relation + 100.0


def make_physical_local_state(topology, step):
    """Build a packing-valid, exactly reconstructable physical snapshot."""
    physical_count = len(topology.all_rail_ids)
    rows = np.arange(physical_count, dtype=np.int64)
    physical = np.zeros((physical_count, LOCAL_PHYSICAL_DIM), np.float32)

    # Static rail fields remain bit-identical for every environment step.
    physical[:, 0] = np.asarray(topology.all_rail_ids, dtype=np.float32)
    physical[:, 1] = 1 + rows % 4
    physical[:, 2] = rows % 3
    physical[:, 3] = rows % 2

    # Packed integer fields exercise both uint8 and uint16 storage.
    physical[:, 4] = (rows + int(step)) % 251
    physical[:, 5] = (11 * rows + 2 * int(step)) % 50_001
    occupancy = ((rows + int(step)) % 4).astype(np.uint8)
    state = (rows + int(step)) % 6
    physical[rows, 6 + state] = occupancy
    stopped = np.where((rows + int(step)) % 5 == 0, occupancy, 0)
    physical[:, 12] = stopped * ((rows + int(step)) % 251)
    physical[:, 13] = stopped
    return np.ascontiguousarray(physical)


def make_snapshot(topology, step, episode=0, done=False):
    physical = make_physical_local_state(topology, step)
    next_physical = make_physical_local_state(topology, step + 1)
    rows = np.arange(CONTROLLED_COUNT, dtype=np.float32)
    total_tat = np.float32(1_000.0 + 10.0 * step)
    next_total_tat = np.float32(1_000.0 + 10.0 * (step + 1))
    global_state = np.arange(GLOBAL_DIM, dtype=np.float32) + step
    next_global_state = np.arange(GLOBAL_DIM, dtype=np.float32) + step + 1.0
    global_state[0] = total_tat
    next_global_state[0] = next_total_tat
    return ContextualStepSnapshot(
        physical_local_state=physical,
        global_state=np.ascontiguousarray(global_state),
        critic_total_tat=np.asarray([total_tat], dtype=np.float32),
        previous_applied_action=(rows / CONTROLLED_COUNT * 0.25)[:, None],
        policy_action=(rows / CONTROLLED_COUNT)[:, None],
        applied_action=(rows / CONTROLLED_COUNT * 0.25)[:, None],
        reward=rows + step * 10,
        next_physical_local_state=next_physical,
        next_global_state=np.ascontiguousarray(next_global_state),
        next_critic_total_tat=np.asarray([next_total_tat], dtype=np.float32),
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

    def _fill(self, replay, pushes, episode_length):
        step = episode = 0
        for _ in range(pushes):
            done = (step + 1) % episode_length == 0
            replay.push(make_snapshot(
                self.topology, step, episode=episode, done=done
            ))
            if done:
                episode += 1
                step = 0
            else:
                step += 1

    def test_random_eviction_keeps_the_full_2x_state_reservation(self):
        """Random eviction scatters the live set; measured need is up to 2x."""
        for margin in (0, 5, 10_000):
            replay = ContextualStepReplayBuffer(
                self.topology, self.builder, capacity_env_steps=32, seed=3,
                eviction_mode=REPLAY_EVICTION_RANDOM,
                state_capacity_margin=margin,
            )
            self.assertEqual(replay.state_capacity, 64)

    def test_fifo_state_capacity_follows_the_margin(self):
        for margin, expected in ((0, 32), (8, 40), (10_000, 64)):
            replay = ContextualStepReplayBuffer(
                self.topology, self.builder, capacity_env_steps=32, seed=3,
                eviction_mode=REPLAY_EVICTION_FIFO,
                state_capacity_margin=margin,
            )
            # The margin is clamped to the capacity, so 10_000 lands on 2x.
            self.assertEqual(replay.state_capacity, expected)

    def test_fifo_margin_keeps_every_transition_samplable(self):
        """One state per episode boundary is all FIFO needs above capacity."""
        for episode_length in (16, 64, 10_000):
            with self.subTest(episode_length=episode_length):
                replay = ContextualStepReplayBuffer(
                    self.topology, self.builder, capacity_env_steps=64, seed=3,
                    eviction_mode=REPLAY_EVICTION_FIFO,
                    state_capacity_margin=16,
                )
                self.assertEqual(replay.state_capacity, 80)
                self._fill(replay, 400, episode_length)
                live = int(np.count_nonzero(replay._transition_valid))
                self.assertEqual(live, 64)
                self.assertEqual(replay._valid_transition_slots().size, 64)
                self.assertEqual(
                    replay.diagnostics()["replay/unsamplable_env_steps"], 0.0
                )

    def test_undersized_fifo_margin_degrades_without_raising(self):
        """A too-small margin must lose samples, never corrupt or crash."""
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=64, seed=3,
            eviction_mode=REPLAY_EVICTION_FIFO, state_capacity_margin=2,
        )
        self._fill(replay, 400, 1)          # every push ends its episode
        live = int(np.count_nonzero(replay._transition_valid))
        samplable = replay._valid_transition_slots().size
        self.assertEqual(live, 64)
        self.assertLess(samplable, live)
        self.assertGreater(samplable, 0)
        self.assertEqual(
            replay.diagnostics()["replay/unsamplable_env_steps"],
            float(live - samplable),
        )
        batch = replay.sample(8)            # sampling still works
        self.assertEqual(batch.policy_action.shape, (8, 1))

    def test_estimate_matches_allocated_state_capacity(self):
        for eviction in (REPLAY_EVICTION_FIFO, REPLAY_EVICTION_RANDOM):
            for margin in (4, 10_000):
                replay = ContextualStepReplayBuffer(
                    self.topology, self.builder, capacity_env_steps=32,
                    seed=3, eviction_mode=eviction,
                    state_capacity_margin=margin,
                )
                estimated = ContextualStepReplayBuffer.estimate_capacity_bytes(
                    32,
                    physical_count=replay.physical_count,
                    controlled_count=replay.controlled_count,
                    neighbor_count=replay.neighbor_count,
                    lap_enabled=replay.lap_enabled,
                    eviction_mode=eviction,
                    state_capacity_margin=margin,
                )
                self.assertGreaterEqual(estimated, replay.storage_bytes)
                self.assertEqual(
                    replay.diagnostics()["replay/state_capacity"],
                    float(replay.state_capacity),
                )

    def test_negative_state_capacity_margin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "state_capacity_margin"):
            ContextualStepReplayBuffer(
                self.topology, self.builder, capacity_env_steps=8, seed=3,
                state_capacity_margin=-1,
            )

    def test_one_push_is_4996_logical_rows_and_sample_contract(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=4, seed=5
        )
        snapshot = make_snapshot(self.topology, 0, done=True)
        replay.push(snapshot)
        self.assertEqual(replay.size_env_steps, 1)
        self.assertEqual(
            replay.diagnostics()["replay/size_logical_transitions"],
            CONTROLLED_COUNT,
        )
        batch = replay.sample(16)
        self.assertEqual(LOCAL_PHYSICAL_DIM, 14)
        self.assertEqual(GLOBAL_DIM, 6)
        self.assertEqual(CRITIC_EXTRA_DIM, 1)
        self.assertEqual(self.topology.incoming_neighbor_ids.shape[1], 15)
        self.assertEqual(self.topology.outgoing_neighbor_ids.shape[1], 15)
        expected_shapes = {
            "center_local": (16, 14),
            "incoming_local": (16, 15, 14),
            "outgoing_local": (16, 15, 14),
            "center_rail_index": (16,),
            "incoming_rail_indices": (16, 15),
            "outgoing_rail_indices": (16, 15),
            "incoming_relation": (16, 15, 2),
            "outgoing_relation": (16, 15, 2),
            "global_state": (16, 6),
            "critic_total_tat": (16, 1),
            "previous_applied_action": (16, 1),
            "policy_action": (16, 1),
            "applied_action": (16, 1),
            "reward": (16, 1),
            "next_center_local": (16, 14),
            "next_incoming_local": (16, 15, 14),
            "next_outgoing_local": (16, 15, 14),
            "next_global_state": (16, 6),
            "next_critic_total_tat": (16, 1),
            "next_previous_applied_action": (16, 1),
            "done": (16, 1),
            "controlled_rail_id": (16,),
        }
        for name, shape in expected_shapes.items():
            value = getattr(batch, name)
            self.assertEqual(tuple(value.shape), shape)
            self.assertTrue(value.is_contiguous())
            self.assertTrue(torch.isfinite(value).all())
        for name in (
            "center_rail_index",
            "incoming_rail_indices",
            "outgoing_rail_indices",
        ):
            self.assertEqual(getattr(batch, name).dtype, torch.int64)
        self.assertTrue(torch.all(batch.done == 1))

        physical_index_by_id = {
            int(rail_id): index
            for index, rail_id in enumerate(self.topology.all_rail_ids)
        }
        for index, key in enumerate(batch.sample_keys):
            row = key.controlled_row
            rail_id = int(self.topology.controlled_rail_ids[row])
            physical_row = int(
                self.topology.controlled_row_to_physical_index[row]
            )
            self.assertEqual(int(batch.controlled_rail_id[index]), rail_id)
            self.assertEqual(
                int(batch.center_rail_index[index]), physical_row
            )
            np.testing.assert_array_equal(
                batch.incoming_rail_indices[index].numpy(),
                np.asarray([
                    physical_index_by_id[int(neighbor_id)]
                    for neighbor_id in self.topology.incoming_neighbor_ids[row]
                ], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                batch.outgoing_rail_indices[index].numpy(),
                np.asarray([
                    physical_index_by_id[int(neighbor_id)]
                    for neighbor_id in self.topology.outgoing_neighbor_ids[row]
                ], dtype=np.int64),
            )
            self.assertAlmostEqual(
                float(batch.center_local[index, 0]), float(physical_row)
            )
            np.testing.assert_array_equal(
                batch.center_local[index].numpy(),
                snapshot.physical_local_state[physical_row],
            )
            incoming_rows = batch.incoming_rail_indices[index].numpy()
            outgoing_rows = batch.outgoing_rail_indices[index].numpy()
            np.testing.assert_array_equal(
                batch.incoming_local[index].numpy(),
                snapshot.physical_local_state[incoming_rows],
            )
            np.testing.assert_array_equal(
                batch.outgoing_local[index].numpy(),
                snapshot.physical_local_state[outgoing_rows],
            )
            np.testing.assert_array_equal(
                batch.next_center_local[index].numpy(),
                snapshot.next_physical_local_state[physical_row],
            )
            self.assertLessEqual(
                abs(
                    float(batch.policy_action[index, 0])
                    - row / CONTROLLED_COUNT
                ),
                ACTION_FIXED_POINT_MAX_ABS_ERROR
                + float(np.finfo(np.float32).eps),
            )
            self.assertAlmostEqual(float(batch.reward[index, 0]), float(row))
            self.assertEqual(
                float(batch.critic_total_tat[index, 0]),
                float(snapshot.critic_total_tat[0]),
            )
            self.assertEqual(
                float(batch.next_critic_total_tat[index, 0]),
                float(snapshot.next_critic_total_tat[0]),
            )
            self.assertNotEqual(
                float(batch.critic_total_tat[index, 0]),
                float(batch.next_critic_total_tat[index, 0]),
            )

        self.assertEqual(replay._physical_uint8.dtype, np.uint8)
        self.assertEqual(replay._physical_uint16.dtype, np.uint16)
        self.assertEqual(replay._physical_static.dtype, np.float32)

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
                    boundary_index = int(np.flatnonzero(
                        self.topology.all_rail_ids == BOUNDARY_IDS[0]
                    )[0])
                    self.assertEqual(
                        float(batch.incoming_local[i, 0, 0]),
                        float(BOUNDARY_IDS[0]),
                    )
                    self.assertEqual(
                        int(batch.incoming_rail_indices[i, 0]),
                        boundary_index,
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
            self.builder.critic_normalizer.count,
            self.builder.critic_normalizer.update_calls,
        )
        replay.sample(32)
        after = (
            self.builder.local_normalizer.count,
            self.builder.local_normalizer.update_calls,
            self.builder.global_normalizer.count,
            self.builder.global_normalizer.update_calls,
            self.builder.critic_normalizer.count,
            self.builder.critic_normalizer.update_calls,
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
        packed = make_snapshot(self.topology, 0)
        completed = replace(
            completed,
            state=replace(
                completed.state,
                physical_local_raw=packed.physical_local_state,
                global_raw=packed.global_state,
                critic_total_tat_raw=packed.critic_total_tat,
            ),
            next_state=replace(
                completed.next_state,
                physical_local_raw=packed.next_physical_local_state,
                global_raw=packed.next_global_state,
                critic_total_tat_raw=packed.next_critic_total_tat,
            ),
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
        for field in ("critic_total_tat", "next_critic_total_tat"):
            with self.subTest(field=field):
                with self.assertRaises(ContextualReplayError):
                    replay.push(
                        replace(
                            base,
                            **{field: np.asarray([np.nan], np.float32)},
                        )
                    )

        mismatched_global = base.global_state.copy()
        mismatched_global[0] += 1.0
        with self.assertRaisesRegex(
            ContextualReplayError,
            "actor-global TotalTat must equal direct critic TotalTat",
        ):
            replay.push(replace(base, global_state=mismatched_global))

    def test_packed_integer_features_reject_fractional_and_overflow(self):
        base = make_snapshot(self.topology, 0)
        cases = (
            (4, 1.5, "uint8 physical replay features must be exactly integral"),
            (4, 256.0, "uint8 physical replay feature overflow"),
            (5, 1.5, "uint16 physical replay features must be exactly integral"),
            (5, 65_536.0, "uint16 physical replay feature overflow"),
        )
        for feature, value, message in cases:
            with self.subTest(feature=feature, value=value):
                physical = base.physical_local_state.copy()
                physical[0, feature] = value
                replay = ContextualStepReplayBuffer(
                    self.topology, self.builder, capacity_env_steps=2
                )
                with self.assertRaisesRegex(ContextualReplayError, message):
                    replay.push(replace(base, physical_local_state=physical))

    def test_static_physical_feature_drift_is_rejected(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        replay.push(make_snapshot(self.topology, 0))
        second = make_snapshot(self.topology, 1)
        physical = second.physical_local_state.copy()
        physical[0, 0] += 0.5
        with self.assertRaisesRegex(
            ContextualReplayError,
            "static physical replay features changed after initialization",
        ):
            replay.push(replace(second, physical_local_state=physical))

    def test_contiguous_previous_action_mismatch_is_rejected(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        replay.push(make_snapshot(self.topology, 0))
        second = make_snapshot(self.topology, 1)
        previous = second.previous_applied_action.copy()
        previous[0, 0] += 0.1
        with self.assertRaisesRegex(
            ContextualReplayError,
            "contiguous transition previous_applied_action differs",
        ):
            replay.push(
                replace(second, previous_applied_action=previous)
            )

    def test_duplicate_live_transition_is_rejected_before_state_packing(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=2
        )
        snapshot = make_snapshot(self.topology, 0)
        replay.push(snapshot)
        state_cursor = replay._state_cursor
        transition_cursor = replay._transition_cursor
        key_to_state = replay._key_to_state.copy()
        key_to_transition = replay._key_to_transition.copy()

        # A packed-column violation would fail later in _pack_physical.  The
        # duplicate live transition key must be rejected before any state is
        # packed, inserted, or overwritten.
        duplicate_physical = snapshot.physical_local_state.copy()
        duplicate_physical[0, 4] = 1.5
        with self.assertRaisesRegex(
            ContextualReplayError, "duplicate live transition"
        ):
            replay.push(
                replace(
                    snapshot,
                    physical_local_state=duplicate_physical,
                )
            )

        self.assertEqual(replay.push_count, 1)
        self.assertEqual(replay.size_env_steps, 1)
        self.assertEqual(replay._state_cursor, state_cursor)
        self.assertEqual(replay._transition_cursor, transition_cursor)
        self.assertEqual(replay._key_to_state, key_to_state)
        self.assertEqual(replay._key_to_transition, key_to_transition)

    def test_overwrite_prunes_episode_without_live_transitions(self):
        replay = ContextualStepReplayBuffer(
            self.topology, self.builder, capacity_env_steps=1
        )
        replay.push(make_snapshot(self.topology, 0, episode=0, done=True))
        self.assertEqual(replay._episode_first_step, {0: 0})

        replay.push(make_snapshot(self.topology, 0, episode=1, done=True))

        self.assertEqual(replay.size_env_steps, 1)
        self.assertEqual(replay._episode_first_step, {1: 0})
        batch = replay.sample(16)
        self.assertTrue(torch.all(batch.episode_id == 1))

    def test_int16_actions_round_trip_within_fixed_point_error(self):
        replay = ContextualStepReplayBuffer(
            self.topology,
            self.builder,
            capacity_env_steps=2,
            sampling_mode=REPLAY_SAMPLING_SNAPSHOT,
        )
        snapshot = make_snapshot(self.topology, 0)
        previous = np.linspace(
            -1.0, 1.0, CONTROLLED_COUNT, dtype=np.float32
        )[:, None]
        policy = previous[::-1].copy()
        applied = np.sin(
            np.linspace(-np.pi / 2, np.pi / 2, CONTROLLED_COUNT)
        ).astype(np.float32)[:, None]
        snapshot = replace(
            snapshot,
            previous_applied_action=previous,
            policy_action=policy,
            applied_action=applied,
            next_previous_applied_action=applied.copy(),
        )
        replay.push(snapshot)
        batch = replay.sample(1)
        tolerance = (
            ACTION_FIXED_POINT_MAX_ABS_ERROR
            + float(np.finfo(np.float32).eps)
        )
        for actual, expected in (
            (batch.previous_applied_action[:, 0].numpy(), previous[:, 0]),
            (batch.policy_action[:, 0].numpy(), policy[:, 0]),
            (batch.applied_action[:, 0].numpy(), applied[:, 0]),
            (batch.next_previous_applied_action[:, 0].numpy(), applied[:, 0]),
        ):
            self.assertLessEqual(
                float(np.max(np.abs(actual - expected))), tolerance
            )
        np.testing.assert_array_equal(
            batch.next_previous_applied_action.numpy(),
            batch.applied_action.numpy(),
        )
        self.assertEqual(replay._previous_applied_action.dtype, np.int16)
        self.assertEqual(replay._policy_action.dtype, np.int16)
        self.assertEqual(replay._applied_action.dtype, np.int16)

    def test_replay_is_locked_to_reward_q(self):
        with self.assertRaisesRegex(ValueError, "only reward_version in \('Q', 'N'\)"):
            ContextualStepReplayBuffer(
                self.topology,
                self.builder,
                capacity_env_steps=2,
                reward_version="O",
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
        self.assertEqual(
            tuple(batch.center_local.shape), (expected_rows, 14)
        )
        self.assertEqual(
            tuple(batch.incoming_local.shape), (expected_rows, 15, 14)
        )
        self.assertEqual(
            tuple(batch.center_rail_index.shape), (expected_rows,)
        )
        self.assertEqual(
            tuple(batch.incoming_rail_indices.shape), (expected_rows, 15)
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
