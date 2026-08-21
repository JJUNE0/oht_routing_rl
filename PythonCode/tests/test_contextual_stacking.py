import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualObservationHistory,
    ContextualReplayError,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
    ContextualActor,
    ContextualTwinCritic,
    interleave_state_action,
)
from oht_routing.algorithms.rl.contextual_td7.replay_buffer import (
    ACTION_FIXED_POINT_MAX_ABS_ERROR,
)
from test_contextual_learner import SMALL_NETWORK
from test_contextual_observation import CONTROLLED_COUNT, make_topology
from test_contextual_replay import (
    FakeObservationBuilder,
    make_snapshot,
)
from test_contextual_training_runtime import training_runtime


def stacked_replay(*, capacity=12, count=12, stacks=3, interval=2):
    topology = make_topology()
    replay = ContextualStepReplayBuffer(
        topology,
        FakeObservationBuilder(topology),
        capacity_env_steps=capacity,
        seed=19,
        num_stacks=stacks,
        stack_interval=interval,
    )
    rows = np.arange(CONTROLLED_COUNT, dtype=np.float32)
    row_fraction = (rows + 1.0) / CONTROLLED_COUNT

    def applied_at(step):
        return -0.75 + 0.08 * float(step) + 0.02 * row_fraction

    for step in range(count):
        snapshot = make_snapshot(topology, step)
        policy = (
            -0.65 + 0.07 * float(step) + 0.03 * row_fraction
        )[:, None]
        applied = applied_at(step)[:, None]
        previous_applied = applied_at(max(step - 1, 0))[:, None]
        replay.push(replace(
            snapshot,
            previous_applied_action=previous_applied,
            policy_action=policy,
            applied_action=applied,
            next_previous_applied_action=applied,
        ))
    return replay


class ContextualStackingTests(unittest.TestCase):
    def test_config_validation_and_default_one_frame_contract(self):
        default = ContextualNetworkConfig()
        self.assertEqual(default.num_stacks, 1)
        self.assertEqual(default.stack_interval, 1)
        self.assertEqual(default.stacked_context_dim, default.context_dim)
        self.assertEqual(default.stacked_action_dim, default.action_dim)
        self.assertEqual(
            default.actor_input_dim, default.context_dim + default.action_dim
        )
        for kwargs in (
            {"num_stacks": 0},
            {"stack_interval": 0},
            {"num_stacks": True},
            {"stack_interval": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ContextualNetworkConfig(**kwargs)

    def test_online_history_uses_interval_and_left_pads_first_frame(self):
        history = ContextualObservationHistory(3, 2)
        frames = [SimpleNamespace(marker=step) for step in range(4)]
        for step, frame in enumerate(frames):
            history.append(frame, env_step=step, episode_id=7)
        self.assertEqual(
            [frame.marker for frame in history.frames()], [3, 1, 0]
        )
        history.clear()
        history.append(frames[0], env_step=5, episode_id=8)
        self.assertEqual(
            [frame.marker for frame in history.frames()], [0, 0, 0]
        )
        with self.assertRaisesRegex(ValueError, "episode boundary"):
            history.append(frames[1], env_step=6, episode_id=9)

    def test_replay_materializes_current_next_and_action_history(self):
        replay = stacked_replay()
        batch = replay.sample(128)
        self.assertEqual(batch.center_local.shape, (128, 3, 16))
        self.assertEqual(batch.incoming_local.shape, (128, 3, 15, 16))
        self.assertEqual(batch.outgoing_local.shape, (128, 3, 15, 16))
        self.assertEqual(batch.center_rail_index.shape, (128, 3))
        self.assertEqual(batch.incoming_rail_indices.shape, (128, 3, 15))
        self.assertEqual(batch.outgoing_rail_indices.shape, (128, 3, 15))
        self.assertEqual(batch.global_state.shape, (128, 3, 17))
        self.assertEqual(batch.center_rail_index.dtype, torch.long)
        self.assertEqual(batch.incoming_rail_indices.dtype, torch.long)
        self.assertEqual(batch.outgoing_rail_indices.dtype, torch.long)
        self.assertEqual(
            batch.previous_applied_action.shape, (128, 3, 1)
        )
        self.assertEqual(batch.applied_action.shape, (128, 3, 1))
        self.assertEqual(
            batch.next_previous_applied_action.shape, (128, 3, 1)
        )
        self.assertEqual(batch.next_applied_action.shape, (128, 3, 1))

        first_slot = replay._key_to_transition[(0, 0)][0]
        _, _, first_actions, first_next_actions = replay._history_slot_arrays(
            np.asarray([first_slot], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            first_actions[0], np.full(3, first_slot, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            first_next_actions[0, 1:],
            np.full(2, first_slot, dtype=np.int64),
        )

        for index, key in enumerate(batch.sample_keys):
            step = int(batch.env_step[index])
            row = key.controlled_row
            physical_row = int(replay._center_rows[row])
            np.testing.assert_array_equal(
                batch.center_rail_index[index].numpy(),
                np.full(3, physical_row, dtype=np.int64),
            )
            np.testing.assert_array_equal(
                batch.incoming_rail_indices[index].numpy(),
                np.broadcast_to(replay._incoming_rows[row], (3, 15)),
            )
            np.testing.assert_array_equal(
                batch.outgoing_rail_indices[index].numpy(),
                np.broadcast_to(replay._outgoing_rows[row], (3, 15)),
            )
            base = float(replay.topology.all_rail_ids[physical_row])
            expected_state_steps = [max(step - offset, 0) for offset in (0, 2, 4)]
            expected_next_steps = [
                max(step + 1 - offset, 0) for offset in (0, 2, 4)
            ]
            np.testing.assert_allclose(
                batch.center_local[index, :, 6].numpy(),
                7.0 * base + np.asarray(expected_state_steps),
            )
            np.testing.assert_allclose(
                batch.next_center_local[index, :, 6].numpy(),
                7.0 * base + np.asarray(expected_next_steps),
            )
            row_fraction = (row + 1) / CONTROLLED_COUNT

            def expected_applied(action_step):
                return (
                    -0.75
                    + 0.08 * float(action_step)
                    + 0.02 * row_fraction
                )

            expected_actions = [
                expected_applied(max(step - offset, 0))
                for offset in (0, 2, 4)
            ]
            expected_next_actions = [
                0.0,
                expected_applied(max(step - 1, 0)),
                expected_applied(max(step - 3, 0)),
            ]
            action_tolerance = (
                ACTION_FIXED_POINT_MAX_ABS_ERROR
                + float(np.finfo(np.float32).eps)
            )
            np.testing.assert_allclose(
                batch.applied_action[index, :, 0].numpy(),
                expected_actions,
                rtol=0,
                atol=action_tolerance,
            )
            expected_previous_actions = [
                expected_applied(max(step - offset - 1, 0))
                for offset in (0, 2, 4)
            ]
            np.testing.assert_allclose(
                batch.previous_applied_action[index, :, 0].numpy(),
                expected_previous_actions,
                rtol=0,
                atol=action_tolerance,
            )
            np.testing.assert_allclose(
                batch.next_previous_applied_action[index, :, 0].numpy(),
                expected_actions,
                rtol=0,
                atol=action_tolerance,
            )
            np.testing.assert_allclose(
                batch.next_applied_action[index, :, 0].numpy(),
                expected_next_actions,
                rtol=0,
                atol=action_tolerance,
            )
        torch.testing.assert_close(
            batch.next_previous_applied_action,
            batch.applied_action,
            rtol=0,
            atol=0,
        )

    def test_ring_overwrite_excludes_transitions_with_missing_history(self):
        replay = stacked_replay(
            capacity=4, count=5, stacks=3, interval=1
        )
        valid_steps = set(
            replay._env_step[replay._valid_transition_slots()].tolist()
        )
        self.assertEqual(valid_steps, {3, 4})
        self.assertEqual(replay.size_env_steps, 2)
        sampled = replay.sample(64)
        self.assertTrue(set(sampled.env_step.tolist()) <= {3, 4})

        padded = stacked_replay(
            capacity=4, count=8, stacks=2, interval=10
        )
        self.assertEqual(padded.size_env_steps, 0)
        with self.assertRaisesRegex(ContextualReplayError, "empty replay"):
            padded.sample(1)

    def test_actor_critic_dimensions_and_frame_interleave(self):
        config = replace(
            SMALL_NETWORK, num_stacks=3, stack_interval=2
        )
        state = torch.arange(
            2 * config.stacked_context_dim, dtype=torch.float32
        ).reshape(2, -1)
        action = torch.arange(
            2 * config.stacked_action_dim, dtype=torch.float32
        ).reshape(2, -1)
        previous_action = -action
        interleaved = interleave_state_action(
            state,
            action,
            num_stacks=3,
            context_dim=config.context_dim,
            action_dim=config.action_dim,
        )
        first = interleaved[0].reshape(3, config.context_dim + 1)
        torch.testing.assert_close(
            first[:, :-1], state[0].reshape(3, config.context_dim)
        )
        torch.testing.assert_close(first[:, -1], action[0])

        actor = ContextualActor(config)
        critic = ContextualTwinCritic(config)
        actor_output = actor(state, previous_action=previous_action)
        critic_output = critic(
            state, action, previous_action=previous_action
        )
        self.assertEqual(actor_output.action.shape, (2, 1))
        self.assertEqual(critic_output.q1.shape, (2, 1))
        self.assertEqual(critic.q1.network[0].in_features, config.critic_input_dim)

    def test_stacked_sale_learner_update_smoke(self):
        replay = stacked_replay(capacity=10, count=10)
        network = replace(
            SMALL_NETWORK, num_stacks=3, stack_interval=2
        )
        config = ContextualLearnerConfig(
            batch_size=8,
            target_noise=0.0,
            target_update_interval=4,
            policy_update_delay=2,
            minimum_replay_env_steps=1,
            minimum_action_enabled_env_steps=1,
            require_normalizer_frozen=False,
            sale_enabled=True,
            lap_enabled=False,
            sale_embedding_dim=16,
            sale_feature_dim=16,
        )
        learner = ContextualTD7Learner(
            replay, network_config=network, config=config, seed=23
        )
        first = learner.update()
        second = learner.update()
        self.assertFalse(first.actor_updated)
        self.assertTrue(second.actor_updated)
        self.assertTrue(all(
            np.isfinite(value) for value in second.diagnostics.values()
        ))

    def test_runtime_actor_uses_and_resets_stacked_history(self):
        runtime, pclient = training_runtime(
            num_stacks=3,
            stack_interval=2,
            warmup_steps=0,
            normalizer_freeze_steps=1,
        )
        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)
        self.assertEqual(runtime.observation_history.size, 3)
        self.assertEqual(runtime.last_diagnostics["stack/num_stacks"], 3.0)
        self.assertEqual(runtime.last_diagnostics["stack/interval"], 2.0)
        self.assertEqual(runtime.last_controlled_action.shape, (CONTROLLED_COUNT,))
        runtime.Reset(pclient)
        self.assertEqual(runtime.observation_history.size, 0)


if __name__ == "__main__":
    unittest.main()
