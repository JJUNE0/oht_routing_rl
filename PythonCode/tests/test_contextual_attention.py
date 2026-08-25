import inspect
import unittest

import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualNetworkConfig,
    DirectionalContextEncoder,
    encoding_diagnostics,
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


class ContextualAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = ContextualNetworkConfig()
        self.encoder = DirectionalContextEncoder(self.config).eval()

    def test_shapes_for_one_and_arbitrary_batch(self):
        for batch in (1, 7):
            with self.subTest(batch=batch):
                output = self.encoder(*make_inputs(batch), return_attention=True)
                self.assertEqual(output.state.shape, (batch, 128))
                self.assertEqual(output.center_embedding.shape, (batch, 64))
                self.assertEqual(output.global_embedding.shape, (batch, 32))
                self.assertEqual(output.incoming_context.shape, (batch, 64))
                self.assertEqual(output.outgoing_context.shape, (batch, 64))
                self.assertEqual(
                    output.incoming_attention.shape, (batch, 4, 1, 15)
                )
                self.assertEqual(
                    output.outgoing_attention.shape, (batch, 4, 1, 15)
                )

    def test_production_batch_4996_is_supported(self):
        with torch.inference_mode():
            output = self.encoder(*make_inputs(4996))
        self.assertEqual(output.state.shape, (4996, 128))
        self.assertIsNone(output.incoming_attention)
        self.assertIsNone(output.outgoing_attention)

    def test_directional_changes_and_direct_isolation(self):
        inputs = list(make_inputs(3))
        base = self.encoder(*inputs)
        changed_incoming = list(inputs)
        changed_incoming[1] = changed_incoming[1] + 2.0
        incoming_result = self.encoder(*changed_incoming)
        self.assertFalse(
            torch.allclose(base.incoming_context, incoming_result.incoming_context)
        )
        torch.testing.assert_close(
            base.outgoing_context, incoming_result.outgoing_context
        )

        changed_outgoing = list(inputs)
        changed_outgoing[2] = changed_outgoing[2] - 2.0
        outgoing_result = self.encoder(*changed_outgoing)
        self.assertFalse(
            torch.allclose(base.outgoing_context, outgoing_result.outgoing_context)
        )
        torch.testing.assert_close(
            base.incoming_context, outgoing_result.incoming_context
        )

    def test_direction_modules_have_distinct_parameters(self):
        incoming = list(self.encoder.incoming_attention.parameters())
        outgoing = list(self.encoder.outgoing_attention.parameters())
        self.assertTrue(incoming)
        self.assertEqual(len(incoming), len(outgoing))
        self.assertTrue(all(left is not right for left, right in zip(incoming, outgoing)))
        self.assertTrue(
            set(map(id, incoming)).isdisjoint(set(map(id, outgoing)))
        )

    def test_attention_weights_sum_to_one_and_entropy_is_finite(self):
        output = self.encoder(*make_inputs(5), return_attention=True)
        for weights in (
            output.incoming_attention,
            output.outgoing_attention,
        ):
            torch.testing.assert_close(
                weights.sum(dim=-1),
                torch.ones_like(weights.sum(dim=-1)),
                rtol=1e-5,
                atol=1e-6,
            )
        diagnostics = encoding_diagnostics(output)
        expected = {
            "attention/incoming_entropy",
            "attention/outgoing_entropy",
            "attention/incoming_max_weight",
            "attention/outgoing_max_weight",
            "attention/incoming_context_norm",
            "attention/outgoing_context_norm",
            "embedding/center_norm",
            "embedding/global_norm",
            "embedding/fused_norm",
        }
        self.assertTrue(expected.issubset(diagnostics))
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())
        )

    def test_paired_neighbor_relation_permutation_is_invariant(self):
        inputs = list(make_inputs(4))
        base = self.encoder(*inputs)
        permutation = torch.tensor(
            [12, 2, 14, 1, 5, 0, 8, 11, 6, 3, 13, 4, 10, 9, 7]
        )
        permuted = list(inputs)
        permuted[1] = permuted[1][:, permutation]
        permuted[4] = permuted[4][:, permutation]
        permuted[6] = permuted[6][:, permutation]
        result = self.encoder(*permuted)
        torch.testing.assert_close(
            base.incoming_context, result.incoming_context, rtol=1e-5, atol=1e-6
        )

    def test_local_only_permutation_can_change_context(self):
        inputs = list(make_inputs(4))
        base = self.encoder(*inputs)
        permutation = torch.tensor(
            [12, 2, 14, 1, 5, 0, 8, 11, 6, 3, 13, 4, 10, 9, 7]
        )
        permuted = list(inputs)
        permuted[1] = permuted[1][:, permutation]
        result = self.encoder(*permuted)
        self.assertFalse(
            torch.allclose(base.incoming_context, result.incoming_context)
        )

    def test_direction_exchange_is_not_symmetric(self):
        inputs = list(make_inputs(4))
        base = self.encoder(*inputs)
        swapped = self.encoder(
            inputs[0],
            inputs[2],
            inputs[1],
            inputs[3],
            inputs[5],
            inputs[4],
            inputs[7],
            inputs[6],
            inputs[8],
        )
        self.assertFalse(
            torch.allclose(base.incoming_context, swapped.outgoing_context)
        )

    def test_attention_parameters_and_all_input_groups_receive_gradients(self):
        inputs = make_inputs(3, requires_grad=True)
        output = self.encoder(*inputs)
        output.state.square().mean().backward()
        for index in FLOAT_INPUT_INDICES:
            value = inputs[index]
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.0)
        embedding_grad = self.encoder.rail_embedding.weight.grad
        self.assertIsNotNone(embedding_grad)
        self.assertTrue(torch.isfinite(embedding_grad).all())
        self.assertGreater(float(embedding_grad.abs().sum()), 0.0)
        for module in (
            self.encoder.incoming_attention,
            self.encoder.outgoing_attention,
        ):
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(grad is not None for grad in grads))
            self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))

    def test_no_cross_batch_state_or_gradient_leakage(self):
        inputs = make_inputs(2, requires_grad=True)
        batch_output = self.encoder(*inputs).state[0]
        single_inputs = tuple(value[0:1].detach() for value in inputs)
        single_output = self.encoder(*single_inputs).state[0]
        torch.testing.assert_close(batch_output, single_output, rtol=1e-5, atol=1e-6)
        batch_output.sum().backward()
        for index in FLOAT_INPUT_INDICES:
            value = inputs[index]
            self.assertEqual(float(value.grad[1].abs().sum()), 0.0)

    def test_no_padding_mask_argument(self):
        signature = inspect.signature(self.encoder.forward)
        self.assertNotIn("mask", signature.parameters)
        self.assertNotIn("padding_mask", signature.parameters)
        self.assertFalse(signature.parameters["return_attention"].default)

    def test_global_encoder_is_separate_from_neighbor_token_encoder(self):
        first_center_linear = self.encoder.center_encoder.network[0]
        first_neighbor_linear = self.encoder.neighbor_encoder.network[0]
        first_global_linear = self.encoder.global_encoder.network[0]
        self.assertEqual(first_center_linear.in_features, 22)
        self.assertEqual(first_neighbor_linear.in_features, 24)
        self.assertEqual(first_global_linear.in_features, 5)
        self.assertIsNot(first_neighbor_linear, first_global_linear)


if __name__ == "__main__":
    unittest.main()
