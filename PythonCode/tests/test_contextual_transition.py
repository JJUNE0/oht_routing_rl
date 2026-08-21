import unittest

import numpy as np

from contextual_observation import ContextualObservationBatch
from contextual_reward import ContextualRewardBuilder, ContextualRewardConfig
from contextual_transition import (
    ContextualTransitionAligner,
    ContextualTransitionError,
)
from test_contextual_observation import CONTROLLED_COUNT, PHYSICAL_COUNT, make_topology
from test_contextual_reward import reward_client


def observation(topology, suffix=0):
    return ContextualObservationBatch(
        center_local=np.full((CONTROLLED_COUNT, 8), suffix, np.float32),
        incoming_local=np.zeros((CONTROLLED_COUNT, 10, 8), np.float32),
        outgoing_local=np.zeros((CONTROLLED_COUNT, 10, 8), np.float32),
        incoming_relation=np.zeros((CONTROLLED_COUNT, 10, 2), np.float32),
        outgoing_relation=np.zeros((CONTROLLED_COUNT, 10, 2), np.float32),
        global_state=np.zeros(6, np.float32),
        previous_applied_action=np.zeros((CONTROLLED_COUNT, 1), np.float32),
        controlled_rail_ids=topology.controlled_rail_ids.copy(),
        topology_hash=topology.topology_hash,
        mapping_hash=topology.mapping_hash,
        physical_local_raw=np.full(
            (PHYSICAL_COUNT, 8), suffix, np.float32
        ),
        global_raw=np.full(6, suffix, np.float32),
    )


class ContextualTransitionTests(unittest.TestCase):
    def setUp(self):
        self.topology = make_topology()
        self.builder = ContextualRewardBuilder(
            self.topology,
            ContextualRewardConfig(rail_reward_mode="fixed_tat_reference"),
        )
        self.items = []
        self.aligner = ContextualTransitionAligner(
            self.topology, self.builder, self.items.append
        )
        self.zeros = np.zeros((CONTROLLED_COUNT, 1), np.float32)
        self.cost = np.ones(PHYSICAL_COUNT, np.float32)

    def advance(self, step, *, done=False, episode=0):
        obs = observation(self.topology, step)
        completed = self.aligner.advance(
            observation=obs, pclient=reward_client(),
            controlled_action=self.zeros, applied_action=self.zeros,
            baseline_cost=self.cost, final_cost=self.cost,
            env_step=step, episode_id=episode, done=done,
        )
        return obs, completed

    def test_first_tick_then_exact_alignment_and_identity_reuse(self):
        first, completed = self.advance(0)
        self.assertIsNone(completed)
        self.assertIs(self.aligner.pending.observation, first)
        second, completed = self.advance(1)
        self.assertEqual(completed.env_step, 0)
        self.assertEqual(completed.next_env_step, 1)
        self.assertIs(completed.next_state, second)
        self.assertIs(self.aligner.pending.observation, second)
        self.assertEqual(len(self.items), 1)
        self.assertFalse(completed.action.flags.writeable)

    def test_100_ticks_make_99_transitions(self):
        for step in range(100):
            self.advance(step)
        self.assertEqual(self.aligner.completed_count, 99)
        self.assertEqual(len(self.items), 99)

    def test_reset_prevents_cross_episode_transition(self):
        self.advance(0)
        self.aligner.reset(1)
        _, completed = self.advance(0, episode=1)
        self.assertIsNone(completed)
        self.assertEqual(self.aligner.completed_count, 0)
        self.assertIsNone(self.aligner.previous_applied_action)

    def test_100_ticks_with_midpoint_reset_make_98_transitions(self):
        for step in range(50):
            self.advance(step)
        self.aligner.reset(1)
        for step in range(50):
            self.advance(step, episode=1)
        self.assertEqual(self.aligner.completed_count, 98)

    def test_terminal_transition_done_and_no_new_pending(self):
        self.advance(0)
        _, completed = self.advance(1, done=True)
        self.assertTrue(completed.done)
        self.assertIsNone(self.aligner.pending)

    def test_hash_and_step_gap_fail_fast(self):
        self.advance(0)
        bad = observation(self.topology, 1)
        object.__setattr__(bad, "mapping_hash", "wrong")
        with self.assertRaises(ContextualTransitionError):
            self.aligner.complete_previous(
                observation=bad, pclient=reward_client(),
                env_step=1, episode_id=0,
            )
        with self.assertRaises(ContextualTransitionError):
            self.aligner.complete_previous(
                observation=observation(self.topology, 2),
                pclient=reward_client(), env_step=2, episode_id=0,
            )


if __name__ == "__main__":
    unittest.main()
