import copy
import unittest
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from cocel_rl.algorithms.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
)
from cocel_rl.algorithms.contextual_td7.sale import (
    SALEOnline,
    avg_l1_norm,
)
from test_contextual_learner import SMALL_NETWORK
from test_contextual_observation import make_topology
from test_contextual_replay import FakeObservationBuilder, make_snapshot


def sale_replay(lap=False):
    topology = make_topology()
    builder = FakeObservationBuilder(topology)
    replay = ContextualStepReplayBuffer(
        topology, builder, capacity_env_steps=8, seed=4,
        lap_enabled=lap,
    )
    rows = np.arange(4996, dtype=np.float32)
    for step in range(8):
        snapshot = make_snapshot(topology, step)
        policy = np.sin(rows + step)[:, None]
        replay.push(replace(
            snapshot, policy_action=policy,
            applied_action=0.05 * policy,
            reward=np.tanh(rows / 1000 + step),
        ))
    return replay


def sale_config(lap=False, interval=4):
    return ContextualLearnerConfig(
        batch_size=16, action_scale=0.05,
        sale_enabled=True, lap_enabled=lap,
        sale_embedding_dim=16, sale_feature_dim=16,
        target_noise=0, target_update_interval=interval,
        minimum_replay_env_steps=1,
        minimum_action_enabled_env_steps=1,
        require_normalizer_frozen=False,
    )


class ContextualSALETests(unittest.TestCase):
    def test_avg_l1_and_sale_shapes_zero_finite(self):
        module = SALEOnline(SMALL_NETWORK, embedding_dim=16)
        batch = sale_replay().sample(4)
        observation = (
            batch.center_local, batch.incoming_local, batch.outgoing_local,
            batch.incoming_relation, batch.outgoing_relation,
            batch.global_state,
        )
        zs = module.state(observation)
        zsa = module.state_action(zs, batch.applied_action)
        self.assertEqual(zs.shape, (4, 16))
        self.assertEqual(zsa.shape, (4, 16))
        self.assertTrue(torch.isfinite(zs).all())
        self.assertTrue(torch.isfinite(avg_l1_norm(torch.zeros(4, 16))).all())
        torch.testing.assert_close(
            zs.abs().mean(-1), torch.ones(4), atol=1e-5, rtol=1e-5
        )

    def test_sale_loss_contract_and_applied_action_input(self):
        learner = ContextualTD7Learner(
            sale_replay(), network_config=SMALL_NETWORK,
            config=sale_config(), seed=8,
        )
        batch = learner.replay.sample(8)
        obs = (
            batch.center_local, batch.incoming_local, batch.outgoing_local,
            batch.incoming_relation, batch.outgoing_relation,
            batch.global_state,
        )
        nxt = (
            batch.next_center_local, batch.next_incoming_local,
            batch.next_outgoing_local, batch.next_incoming_relation,
            batch.next_outgoing_relation, batch.next_global_state,
        )
        zs = learner.sale_online.state(obs)
        pred = learner.sale_online.state_action(zs, batch.applied_action)
        with torch.no_grad():
            target = learner.sale_online.state(nxt)
        expected = F.mse_loss(pred, target.detach())
        changed = learner.sale_online.state_action(zs, batch.policy_action)
        self.assertFalse(torch.equal(pred, changed))
        result = learner.update(batch)
        self.assertAlmostEqual(
            result.diagnostics["sale/loss"], float(expected), places=5
        )

    def test_only_sale_online_changes_from_sale_loss(self):
        learner = ContextualTD7Learner(
            sale_replay(), network_config=SMALL_NETWORK,
            config=sale_config(), seed=2,
        )
        fixed_before = copy.deepcopy(learner.sale_fixed.state_dict())
        target_before = copy.deepcopy(learner.sale_target_fixed.state_dict())
        learner.update()
        self.assertTrue(any(
            not torch.equal(value, fixed_before[name])
            for name, value in learner.sale_online.state_dict().items()
        ))
        for name, value in learner.sale_fixed.state_dict().items():
            torch.testing.assert_close(value, fixed_before[name])
        for name, value in learner.sale_target_fixed.state_dict().items():
            torch.testing.assert_close(value, target_before[name])
        self.assertEqual(learner.last_diagnostics["sale/fixed_grad_norm"], 0)
        self.assertEqual(
            learner.last_diagnostics["sale/target_fixed_grad_norm"], 0
        )

    def test_fixed_rollover_uses_previous_generation(self):
        learner = ContextualTD7Learner(
            sale_replay(), network_config=SMALL_NETWORK,
            config=sale_config(interval=2), seed=3,
        )
        old_fixed = copy.deepcopy(learner.sale_fixed.state_dict())
        learner.update()
        learner.update()
        for name, value in learner.sale_target_fixed.state_dict().items():
            torch.testing.assert_close(value, old_fixed[name])
        for name, value in learner.sale_fixed.state_dict().items():
            torch.testing.assert_close(
                value, learner.sale_online.state_dict()[name]
            )
        self.assertEqual(learner.sale_fixed_generation, 1)

    def test_optimizer_ownership_is_disjoint(self):
        learner = ContextualTD7Learner(
            sale_replay(), network_config=SMALL_NETWORK,
            config=sale_config(),
        )
        optimizers = (
            learner.encoder_optimizer, learner.actor_optimizer,
            learner.critic_optimizer, learner.sale_optimizer,
        )
        sets = [{
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        } for optimizer in optimizers]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertFalse(sets[i] & sets[j])

    def test_actor_and_critic_paths_do_not_backpropagate_into_sale(self):
        learner = ContextualTD7Learner(
            sale_replay(), network_config=SMALL_NETWORK,
            config=sale_config(), seed=12,
        )
        batch = learner.replay.sample(4)
        observation = (
            batch.center_local, batch.incoming_local, batch.outgoing_local,
            batch.incoming_relation, batch.outgoing_relation,
            batch.global_state,
        )
        with torch.no_grad():
            task_state = learner.encoder(*observation).state
            zs = learner.sale_fixed.state(observation)
            zsa = learner.sale_fixed.state_action(zs, batch.applied_action)
        actor_output = learner.actor(task_state.detach(), zs)
        critic_output = learner.critic(
            task_state, batch.applied_action, zs, zsa
        )
        (actor_output.action.mean() + critic_output.q1.mean()).backward()
        self.assertTrue(all(
            parameter.grad is None
            for parameter in learner.sale_online.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in learner.sale_fixed.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in learner.sale_target_fixed.parameters()
        ))
        actor_input = learner.actor.network[0].in_features
        critic_input = learner.critic.q1[0].in_features
        self.assertEqual(actor_input, 32)
        self.assertEqual(critic_input, 48)


if __name__ == "__main__":
    unittest.main()
