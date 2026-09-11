import copy
import io
import inspect
import unittest

import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualActor,
    ContextualNetworkConfig,
    ContextualNetworkError,
    ContextualEnsembleCritic,
    DirectionalContextEncoder,
    actor_diagnostics,
    critic_diagnostics,
)


def make_inputs(batch=4, *, requires_grad=False, scale=1.0, device="cpu"):
    config = ContextualNetworkConfig()
    float_shapes = (
        (batch, config.local_physical_dim),
        (batch, config.neighbor_count, config.local_physical_dim),
        (batch, config.neighbor_count, config.local_physical_dim),
        (batch, config.neighbor_count, config.relation_dim),
        (batch, config.neighbor_count, config.relation_dim),
        (batch, config.global_dim),
    )
    floats = tuple(
        (torch.randn(shape, device=device) * scale).requires_grad_(requires_grad)
        for shape in float_shapes
    )
    center_index = torch.arange(batch, device=device, dtype=torch.long)
    neighbor_offset = torch.arange(
        1, config.neighbor_count + 1, device=device, dtype=torch.long
    )
    incoming_indices = (
        center_index[:, None] + neighbor_offset[None]
    ) % config.num_rails
    outgoing_indices = (
        center_index[:, None] + 2 * neighbor_offset[None]
    ) % config.num_rails
    return (
        floats[0],
        floats[1],
        floats[2],
        center_index,
        incoming_indices,
        outgoing_indices,
        floats[3],
        floats[4],
        floats[5],
    )


FLOAT_INPUT_INDICES = (0, 1, 2, 6, 7, 8)


def make_critic_total_tat(batch, config=None, *, value=0.0, device="cpu"):
    config = config or ContextualNetworkConfig()
    return torch.full(
        (batch, config.critic_extra_dim),
        float(value),
        dtype=torch.float32,
        device=device,
    )


class ContextualNetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.config = ContextualNetworkConfig()
        self.encoder = DirectionalContextEncoder(self.config)
        self.actor = ContextualActor(self.config)
        self.critic = ContextualEnsembleCritic(self.config)

    def test_actor_and_ensemble_critic_shapes_ranges_and_gradients(self):
        state = torch.randn(11, 128, requires_grad=True)
        previous_action = torch.randn(11, 1).tanh().requires_grad_(True)
        actor_output = self.actor(
            state, previous_action=previous_action
        )
        self.assertEqual(actor_output.action.shape, (11, 1))
        self.assertEqual(actor_output.pre_tanh.shape, (11, 1))
        self.assertTrue((actor_output.action >= -1.0).all())
        self.assertTrue((actor_output.action <= 1.0).all())
        action = actor_output.action.detach().requires_grad_(True)
        critic_output = self.critic(
            state,
            action,
            critic_total_tat=make_critic_total_tat(11, self.config),
            previous_action=previous_action,
        )
        self.assertEqual(
            critic_output.q.shape, (11, self.config.num_critics)
        )
        self.assertEqual(critic_output.num_critics, self.config.num_critics)
        self.assertEqual(critic_output.head(0).shape, (11, 1))
        torch.testing.assert_close(
            critic_output.mean, critic_output.q.mean(dim=1, keepdim=True)
        )
        critic_output.q.sum(dim=1).mean().backward()
        self.assertIsNotNone(action.grad)
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(float(action.grad.abs().sum()), 0.0)
        self.assertIsNotNone(previous_action.grad)
        self.assertGreater(float(previous_action.grad.abs().sum()), 0.0)

    def test_previous_action_changes_both_actor_and_critic_outputs(self):
        state = torch.randn(7, self.config.stacked_context_dim)
        current_action = torch.full(
            (7, self.config.stacked_action_dim), 0.25
        )
        low = torch.full_like(current_action, -0.75)
        high = torch.full_like(current_action, 0.75)

        actor_low = self.actor(state, previous_action=low).action
        actor_high = self.actor(state, previous_action=high).action
        critic_low = self.critic(
            state,
            current_action,
            critic_total_tat=make_critic_total_tat(7, self.config),
            previous_action=low,
        )
        critic_high = self.critic(
            state,
            current_action,
            critic_total_tat=make_critic_total_tat(7, self.config),
            previous_action=high,
        )

        self.assertGreater(
            float((actor_low - actor_high).detach().abs().sum()), 0.0
        )
        self.assertGreater(
            float((critic_low.q - critic_high.q).detach().abs().sum()), 0.0
        )
        self.assertEqual(
            self.actor.network[0].in_features, self.config.actor_input_dim
        )
        self.assertEqual(
            self.critic.q_nets[0].network[0].in_features,
            self.config.critic_input_dim,
        )

    def test_total_tat_changes_actor_context_and_ensemble_q(self):
        batch = 7
        low_inputs = list(make_inputs(batch))
        high_inputs = [value.detach().clone() for value in low_inputs]
        low_inputs[-1][:, 0] = -1.25
        high_inputs[-1][:, 0] = 2.5
        self.encoder.eval()
        self.actor.eval()
        with torch.inference_mode():
            low_state = self.encoder(*low_inputs).state
            high_state = self.encoder(*high_inputs).state
            actor_before = self.actor(low_state).action
            actor_after = self.actor(high_state).action
        self.assertGreater(
            float((low_state - high_state).abs().sum()), 0.0
        )
        self.assertGreater(
            float((actor_before - actor_after).abs().sum()), 0.0
        )

        state = torch.randn(batch, self.config.stacked_context_dim)
        previous_action = torch.randn(
            batch, self.config.stacked_action_dim
        ).tanh()
        low_tat = make_critic_total_tat(
            batch, self.config, value=-1.25
        )
        high_tat = make_critic_total_tat(
            batch, self.config, value=2.5
        )
        captured = []
        hook = self.critic.q_nets[0].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            low = self.critic(
                state,
                actor_before,
                critic_total_tat=low_tat,
                previous_action=previous_action,
            )
            high = self.critic(
                state,
                actor_before,
                critic_total_tat=high_tat,
                previous_action=previous_action,
            )
        finally:
            hook.remove()

        self.assertEqual(len(captured), 2)
        torch.testing.assert_close(
            captured[0][:, -self.config.stacked_critic_extra_dim :],
            low_tat,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            captured[1][:, -self.config.stacked_critic_extra_dim :],
            high_tat,
            rtol=0,
            atol=0,
        )
        for index in range(self.config.num_critics):
            self.assertGreater(
                float(
                    (low.head(index) - high.head(index)).detach().abs().sum()
                ),
                0.0,
            )

    def test_actor_loss_reaches_every_observation_group(self):
        inputs = make_inputs(5, requires_grad=True)
        encoding = self.encoder(*inputs)
        loss = -self.actor(encoding.state).action.mean()
        loss.backward()
        for index in FLOAT_INPUT_INDICES:
            value = inputs[index]
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.0)
        self.assertGreater(float(inputs[-1].grad[:, 0].abs().sum()), 0.0)
        embedding_grad = self.encoder.rail_embedding.weight.grad
        self.assertIsNotNone(embedding_grad)
        self.assertTrue(torch.isfinite(embedding_grad).all())
        self.assertGreater(float(embedding_grad.abs().sum()), 0.0)

    def test_rail_identity_changes_encoding_with_identical_physical_state(self):
        inputs = list(make_inputs(3, scale=0.0))
        first = self.encoder(*inputs).state
        changed = list(inputs)
        changed[3] = (changed[3] + 101) % self.config.num_rails
        second = self.encoder(*changed).state
        self.assertFalse(torch.allclose(first, second))

    def test_rail_indices_require_long_dtype_and_valid_range(self):
        wrong_dtype = list(make_inputs(2))
        wrong_dtype[3] = wrong_dtype[3].to(torch.float32)
        with self.assertRaisesRegex(TypeError, "torch.long"):
            self.encoder(*wrong_dtype)

        out_of_range = list(make_inputs(2))
        out_of_range[4] = out_of_range[4].clone()
        out_of_range[4][0, 0] = self.config.num_rails
        with self.assertRaisesRegex(ValueError, "outside"):
            self.encoder(*out_of_range)

    def test_all_parameter_gradients_are_finite(self):
        inputs = make_inputs(4)
        state = self.encoder(*inputs).state
        actor_output = self.actor(state)
        critic_output = self.critic(
            state,
            actor_output.action,
            critic_total_tat=make_critic_total_tat(4, self.config),
        )
        loss = (
            actor_output.action.square().mean()
            + critic_output.q.square().mean()
        )
        loss.backward()
        parameters = list(self.encoder.parameters())
        parameters += list(self.actor.parameters())
        parameters += list(self.critic.parameters())
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))
        self.assertTrue(
            all(torch.isfinite(parameter.grad).all() for parameter in parameters)
        )

    def test_ensemble_critics_do_not_share_parameters(self):
        head_ids = [
            {id(parameter) for parameter in q_net.parameters()}
            for q_net in self.critic.q_nets
        ]
        self.assertEqual(len(head_ids), self.config.num_critics)
        for index, ids in enumerate(head_ids):
            for other in head_ids[index + 1:]:
                self.assertTrue(ids.isdisjoint(other))

    def test_ensemble_size_follows_config(self):
        for num_critics in (2, 3, 7):
            with self.subTest(num_critics=num_critics):
                config = ContextualNetworkConfig(num_critics=num_critics)
                critic = ContextualEnsembleCritic(config)
                self.assertEqual(critic.num_critics, num_critics)
                output = critic(
                    torch.randn(4, config.stacked_context_dim),
                    torch.randn(4, config.stacked_action_dim).tanh(),
                    critic_total_tat=make_critic_total_tat(4, config),
                )
                self.assertEqual(output.q.shape, (4, num_critics))

    def test_ensemble_size_below_two_is_rejected(self):
        for num_critics in (0, 1):
            with self.subTest(num_critics=num_critics):
                with self.assertRaisesRegex(ValueError, "num_critics"):
                    ContextualNetworkConfig(num_critics=num_critics)

    def test_sale_ensemble_critics_are_independently_initialized(self):
        torch.manual_seed(101)
        critic = ContextualEnsembleCritic(
            self.config, sale_embedding_dim=16, sale_feature_dim=16
        )
        heads = [dict(q_net.named_parameters()) for q_net in critic.q_nets]
        schema = heads[0].keys()
        for head in heads[1:]:
            self.assertEqual(head.keys(), schema)
        for index, left in enumerate(heads):
            for right in heads[index + 1:]:
                self.assertTrue(
                    {id(value) for value in left.values()}.isdisjoint(
                        {id(value) for value in right.values()}
                    )
                )
                self.assertTrue(all(
                    a.untyped_storage().data_ptr()
                    != b.untyped_storage().data_ptr()
                    for a, b in zip(left.values(), right.values())
                ))
                self.assertGreater(
                    max(
                        float((left[name] - right[name]).detach().abs().max())
                        for name in schema
                    ),
                    0.0,
                )

        batch = 7
        state = torch.randn(batch, self.config.context_dim)
        action = torch.randn(batch, self.config.action_dim).tanh()
        sale_state = torch.randn(batch, 16)
        sale_state_action = torch.randn(batch, 16)
        output = critic(
            state,
            action,
            sale_state,
            sale_state_action,
            critic_total_tat=make_critic_total_tat(batch, self.config),
        )
        self.assertTrue(torch.isfinite(output.q).all())
        self.assertGreater(
            float(
                output.q.detach().std(dim=1, unbiased=True).mean()
            ),
            0.0,
        )

    def test_models_are_shared_across_batch_not_rail_specific(self):
        self.assertFalse(
            any("rail" in name.lower() for name, _ in self.actor.named_parameters())
        )
        self.assertFalse(
            any("rail" in name.lower() for name, _ in self.critic.named_parameters())
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in self.actor.parameters()),
            sum(parameter.numel() for parameter in ContextualActor().parameters()),
        )

    def test_source_contains_no_region_mask_or_aggregation_path(self):
        source = inspect.getsource(DirectionalContextEncoder).lower()
        source += inspect.getsource(ContextualEnsembleCritic).lower()
        self.assertNotIn("padding", source)
        self.assertNotIn("region", source)
        self.assertNotIn("masked_mean", source)

    def test_zero_extreme_and_normal_forward_are_finite(self):
        for scale in (0.0, 1.0, 1e6):
            with self.subTest(scale=scale):
                inputs = make_inputs(3, scale=scale)
                encoding = self.encoder(*inputs)
                actor_output = self.actor(encoding.state)
                critic_output = self.critic(
                    encoding.state,
                    actor_output.action,
                    critic_total_tat=make_critic_total_tat(
                        3, self.config, value=scale
                    ),
                )
                for tensor in (
                    encoding.state,
                    actor_output.pre_tanh,
                    actor_output.action,
                    critic_output.q,
                ):
                    self.assertTrue(torch.isfinite(tensor).all())

    def test_nan_inf_inputs_fail_fast_without_sanitization(self):
        for value in (float("nan"), float("inf")):
            inputs = list(make_inputs(2))
            inputs[0][0, 0] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ContextualNetworkError, "NaN or Inf"):
                    self.encoder(*inputs)

    def test_dropout_zero_forward_is_deterministic(self):
        inputs = make_inputs(4)
        config = ContextualNetworkConfig(use_attention=True)
        encoder = DirectionalContextEncoder(config)
        first = encoder(*inputs, return_attention=True)
        second = encoder(*inputs, return_attention=True)
        torch.testing.assert_close(first.state, second.state)
        torch.testing.assert_close(
            first.incoming_attention, second.incoming_attention
        )

    def test_same_seed_and_state_dict_produce_identical_output(self):
        inputs = make_inputs(4)
        clone = DirectionalContextEncoder(self.config)
        clone.load_state_dict(copy.deepcopy(self.encoder.state_dict()))
        first = self.encoder(*inputs)
        second = clone(*inputs)
        torch.testing.assert_close(first.state, second.state)

    def test_save_load_round_trip_is_identical(self):
        inputs = make_inputs(4)
        state = self.encoder(*inputs).state
        actor_output = self.actor(state)
        critic_total_tat = make_critic_total_tat(4, self.config, value=0.75)
        critic_output = self.critic(
            state,
            actor_output.action,
            critic_total_tat=critic_total_tat,
        )
        stream = io.BytesIO()
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            stream,
        )
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        encoder = DirectionalContextEncoder(self.config)
        actor = ContextualActor(self.config)
        critic = ContextualEnsembleCritic(self.config)
        encoder.load_state_dict(saved["encoder"])
        actor.load_state_dict(saved["actor"])
        critic.load_state_dict(saved["critic"])
        loaded_state = encoder(*inputs).state
        loaded_actor = actor(loaded_state)
        loaded_critic = critic(
            loaded_state,
            loaded_actor.action,
            critic_total_tat=critic_total_tat,
        )
        torch.testing.assert_close(state, loaded_state)
        torch.testing.assert_close(actor_output.action, loaded_actor.action)
        torch.testing.assert_close(critic_output.q, loaded_critic.q)

    def test_diagnostics_have_required_finite_values(self):
        state = torch.randn(8, 128)
        actor_output = self.actor(state)
        critic_output = self.critic(
            state,
            actor_output.action,
            critic_total_tat=make_critic_total_tat(8, self.config),
        )
        diagnostics = {}
        diagnostics.update(actor_diagnostics(actor_output))
        diagnostics.update(critic_diagnostics(critic_output))
        expected = {
            "action/pre_tanh_mean",
            "action/pre_tanh_std",
            "action/mean",
            "action/std",
            "action/saturation_ratio",
            "critic/q1_mean",
            "critic/q2_mean",
            "critic/q_range",
        }
        self.assertTrue(expected.issubset(diagnostics))
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())
        )

    def test_inputs_are_not_modified(self):
        inputs = make_inputs(3)
        before = tuple(value.clone() for value in inputs)
        state = self.encoder(*inputs).state
        state_before = state.clone()
        action = self.actor(state).action
        action_before = action.clone()
        critic_total_tat = make_critic_total_tat(3, self.config)
        critic_total_tat_before = critic_total_tat.clone()
        self.critic(
            state, action, critic_total_tat=critic_total_tat
        )
        for original, current in zip(before, inputs):
            torch.testing.assert_close(original, current)
        torch.testing.assert_close(state_before, state)
        torch.testing.assert_close(action_before, action)
        torch.testing.assert_close(
            critic_total_tat_before, critic_total_tat
        )

    def test_forward_backward_smoke_100_iterations(self):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=1e-5,
        )
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            state = self.encoder(*make_inputs(2)).state
            actor_output = self.actor(state)
            critic_output = self.critic(
                state,
                actor_output.action,
                critic_total_tat=make_critic_total_tat(2, self.config),
            )
            loss = (
                actor_output.pre_tanh.square().mean()
                + critic_output.q.square().mean()
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        self.assertTrue(
            all(
                torch.isfinite(parameter).all()
                for module in (self.encoder, self.actor, self.critic)
                for parameter in module.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
