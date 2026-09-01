import unittest

import numpy as np

from oht_routing.mdp.action import (
    EXP_RESIDUAL,
    FREE_FLOW_RESIDUAL,
    REGION_B_RL,
    ContextualActionError,
    apply_controlled_action,
    assemble_full_action,
    validate_topology_partition,
)
from test_contextual_observation import (
    BOUNDARY_IDS,
    CONTROLLED_COUNT,
    PHYSICAL_COUNT,
    make_topology,
)


class ContextualActionTests(unittest.TestCase):
    def setUp(self):
        self.topology = make_topology()
        self.baseline = np.linspace(1.0, 20.0, PHYSICAL_COUNT)
        self.action = np.linspace(-1.0, 1.0, CONTROLLED_COUNT)
        self.boundary_rows = np.flatnonzero(
            self.topology.physical_index_to_controlled_row == -1
        )

    def test_controlled_action_maps_to_explicit_physical_rows(self):
        full = assemble_full_action(self.topology, self.action)
        np.testing.assert_array_equal(
            full[self.topology.controlled_row_to_physical_index],
            self.action.astype(np.float32),
        )
        np.testing.assert_array_equal(full[self.boundary_rows], 0.0)
        for row in (0, 100, 3000):
            physical = self.topology.controlled_row_to_physical_index[row]
            self.assertEqual(
                self.topology.all_rail_ids[physical],
                self.topology.controlled_rail_ids[row],
            )

    def test_controlled_boundary_exactly_partition_physical_topology(self):
        validate_topology_partition(self.topology)
        controlled = set(self.topology.controlled_rail_ids.tolist())
        boundary = set(self.topology.boundary_rail_ids.tolist())
        physical = set(self.topology.all_rail_ids.tolist())
        self.assertFalse(controlled & boundary)
        self.assertEqual(controlled | boundary, physical)
        self.assertEqual(boundary, set(BOUNDARY_IDS))

    def test_disabled_and_zero_scale_are_exact_baseline_parity(self):
        disabled = apply_controlled_action(
            self.baseline,
            self.action,
            self.topology,
            action_enabled=False,
            action_scale=1.0,
            action_mode=EXP_RESIDUAL,
        )
        zero_scale = apply_controlled_action(
            self.baseline,
            self.action,
            self.topology,
            action_enabled=True,
            action_scale=0.0,
            action_mode=EXP_RESIDUAL,
        )
        np.testing.assert_array_equal(disabled.final_cost, self.baseline)
        np.testing.assert_array_equal(zero_scale.final_cost, self.baseline)

    def test_positive_increases_and_negative_decreases_controlled_cost(self):
        action = np.zeros(CONTROLLED_COUNT)
        action[0] = 0.5
        action[1] = -0.5
        result = apply_controlled_action(
            self.baseline,
            action,
            self.topology,
            action_enabled=True,
            action_scale=0.2,
            action_mode=EXP_RESIDUAL,
        )
        rows = self.topology.controlled_row_to_physical_index
        self.assertGreater(result.final_cost[rows[0]], self.baseline[rows[0]])
        self.assertLess(result.final_cost[rows[1]], self.baseline[rows[1]])

    def test_boundary_is_exact_baseline_for_arbitrary_actor_action(self):
        result = apply_controlled_action(
            self.baseline,
            self.action,
            self.topology,
            action_enabled=True,
            action_scale=0.5,
            action_mode=EXP_RESIDUAL,
        )
        np.testing.assert_array_equal(
            result.final_cost[self.boundary_rows],
            self.baseline[self.boundary_rows],
        )
        self.assertEqual(
            result.diagnostics["boundary/cost_baseline_abs_error_max"], 0.0
        )
        self.assertEqual(result.diagnostics["boundary/action_abs_max"], 0.0)

    def test_nan_inf_action_and_baseline_fail_fast(self):
        for value in (np.nan, np.inf):
            action = self.action.copy()
            action[0] = value
            with self.subTest(kind="action", value=value):
                with self.assertRaisesRegex(ContextualActionError, "NaN or Inf"):
                    apply_controlled_action(
                        self.baseline,
                        action,
                        self.topology,
                        action_enabled=True,
                        action_scale=0.1,
                        action_mode=EXP_RESIDUAL,
                    )
            baseline = self.baseline.copy()
            baseline[0] = value
            with self.subTest(kind="baseline", value=value):
                with self.assertRaisesRegex(ContextualActionError, "NaN or Inf"):
                    apply_controlled_action(
                        baseline,
                        self.action,
                        self.topology,
                        action_enabled=True,
                        action_scale=0.1,
                        action_mode=EXP_RESIDUAL,
                    )

    def test_out_of_range_cost_fails_instead_of_clipping(self):
        baseline = self.baseline.copy()
        baseline[0] = 20_000_000.0
        with self.assertRaisesRegex(ContextualActionError, "outside simulator range"):
            apply_controlled_action(
                baseline,
                self.action,
                self.topology,
                action_enabled=False,
                action_scale=0.0,
                action_mode=EXP_RESIDUAL,
            )

    def test_inputs_are_not_modified(self):
        baseline = self.baseline.copy()
        action = self.action.copy()
        baseline_before = baseline.copy()
        action_before = action.copy()
        apply_controlled_action(
            baseline,
            action,
            self.topology,
            action_enabled=True,
            action_scale=0.1,
            action_mode=EXP_RESIDUAL,
        )
        np.testing.assert_array_equal(baseline, baseline_before)
        np.testing.assert_array_equal(action, action_before)

    def test_region_b_rl_matches_legacy_mapping_at_full_scale(self):
        base = np.linspace(1.0, 2.0, PHYSICAL_COUNT)
        congestion = np.linspace(2.0, 4.0, PHYSICAL_COUNT)
        baseline = base + 0.5 * congestion
        action = np.zeros(CONTROLLED_COUNT)
        action[:3] = (-1.0, 0.0, 1.0)
        result = apply_controlled_action(
            baseline,
            action,
            self.topology,
            action_enabled=True,
            action_scale=1.0,
            action_mode=REGION_B_RL,
            base_cost=base,
            congestion_cost=congestion,
        )
        rows = self.topology.controlled_row_to_physical_index
        self.assertAlmostEqual(result.final_cost[rows[0]], base[rows[0]])
        self.assertAlmostEqual(result.final_cost[rows[1]], baseline[rows[1]])
        self.assertAlmostEqual(
            result.final_cost[rows[2]],
            base[rows[2]] + congestion[rows[2]],
        )
        np.testing.assert_allclose(
            result.applied_controlled_action[:3], (-1.0, 0.0, 1.0)
        )

    def test_region_curriculum_scale_and_zero_congestion_contract(self):
        base = np.linspace(1.0, 2.0, PHYSICAL_COUNT)
        congestion = np.ones(PHYSICAL_COUNT)
        rows = self.topology.controlled_row_to_physical_index
        congestion[rows[1]] = 0.0
        baseline = base + 0.5 * congestion
        action = np.zeros(CONTROLLED_COUNT)
        action[:2] = (1.0, -1.0)
        result = apply_controlled_action(
            baseline,
            action,
            self.topology,
            action_enabled=True,
            action_scale=0.1,
            action_mode=REGION_B_RL,
            base_cost=base,
            congestion_cost=congestion,
        )
        self.assertAlmostEqual(result.diagnostics["b_rl/min"], 0.45)
        self.assertAlmostEqual(result.diagnostics["b_rl/max"], 0.55)
        self.assertAlmostEqual(
            result.final_cost[rows[0]], base[rows[0]] + 0.55
        )
        self.assertEqual(result.final_cost[rows[1]], base[rows[1]])

    def test_free_flow_residual_matches_requested_values_and_ignores_scale(self):
        base = np.full(PHYSICAL_COUNT, 2.0)
        congestion = np.full(PHYSICAL_COUNT, 0.3)
        baseline = base + 0.5 * congestion
        action = np.zeros(CONTROLLED_COUNT)
        action[:3] = (-1.0, 0.0, 1.0)

        scaled = apply_controlled_action(
            baseline,
            action,
            self.topology,
            action_enabled=True,
            action_scale=0.1,
            action_mode=FREE_FLOW_RESIDUAL,
            base_cost=base,
            congestion_cost=congestion,
            rl_cost_lambda=0.5,
        )
        unscaled = apply_controlled_action(
            baseline,
            action,
            self.topology,
            action_enabled=True,
            action_scale=1.0,
            action_mode=FREE_FLOW_RESIDUAL,
            base_cost=base,
            congestion_cost=congestion,
            rl_cost_lambda=0.5,
        )
        rows = self.topology.controlled_row_to_physical_index
        np.testing.assert_allclose(
            scaled.final_cost[rows[:3]], (1.15, 2.15, 3.15), rtol=0.0, atol=1e-12
        )
        np.testing.assert_array_equal(scaled.final_cost, unscaled.final_cost)
        np.testing.assert_array_equal(
            scaled.applied_controlled_action, action.astype(np.float32)
        )

    def test_free_flow_residual_zero_action_is_exact_baseline(self):
        base = np.linspace(0.5, 2.0, PHYSICAL_COUNT)
        congestion = np.linspace(0.0, 0.6, PHYSICAL_COUNT)
        baseline = base + 0.5 * congestion
        result = apply_controlled_action(
            baseline,
            np.zeros(CONTROLLED_COUNT),
            self.topology,
            action_enabled=True,
            action_scale=0.05,
            action_mode=FREE_FLOW_RESIDUAL,
            base_cost=base,
            congestion_cost=congestion,
            rl_cost_lambda=0.5,
        )
        np.testing.assert_array_equal(result.final_cost, baseline)

    def test_free_flow_residual_changes_cost_when_congestion_is_zero(self):
        base = np.full(PHYSICAL_COUNT, 2.0)
        congestion = np.zeros(PHYSICAL_COUNT)
        action = np.zeros(CONTROLLED_COUNT)
        action[:3] = (-1.0, 0.0, 1.0)
        result = apply_controlled_action(
            base,
            action,
            self.topology,
            action_enabled=True,
            action_scale=0.1,
            action_mode=FREE_FLOW_RESIDUAL,
            base_cost=base,
            congestion_cost=congestion,
            rl_cost_lambda=0.5,
        )
        rows = self.topology.controlled_row_to_physical_index
        np.testing.assert_allclose(
            result.final_cost[rows[:3]], (1.0, 2.0, 3.0), rtol=0.0, atol=1e-12
        )

    def test_free_flow_residual_diagnostics_match_cost_decomposition(self):
        base = np.linspace(1.0, 2.0, PHYSICAL_COUNT)
        congestion = np.linspace(0.0, 0.6, PHYSICAL_COUNT)
        baseline = base + 0.5 * congestion
        action = np.linspace(-1.0, 1.0, CONTROLLED_COUNT)
        result = apply_controlled_action(
            baseline,
            action,
            self.topology,
            action_enabled=True,
            action_scale=0.1,
            action_mode=FREE_FLOW_RESIDUAL,
            base_cost=base,
            congestion_cost=congestion,
            rl_cost_lambda=0.5,
        )
        diagnostics = result.diagnostics
        expected_keys = {
            "rl/action_mean",
            "rl/action_std",
            "rl/action_abs_mean",
            "rail_cost/t_ff_mean",
            "rail_cost/congestion_mean",
            "rail_cost/residual_mean",
            "rail_cost/residual_abs_mean",
            "rail_cost/final_cost_mean",
        }
        self.assertTrue(expected_keys <= diagnostics.keys())
        self.assertAlmostEqual(diagnostics["rl/action_mean"], action.mean())
        self.assertAlmostEqual(diagnostics["rl/action_std"], action.std())
        self.assertAlmostEqual(
            diagnostics["rl/action_abs_mean"], np.abs(action).mean()
        )
        self.assertAlmostEqual(
            diagnostics["rail_cost/t_ff_mean"]
            + diagnostics["rail_cost/congestion_mean"]
            + diagnostics["rail_cost/residual_mean"],
            diagnostics["rail_cost/final_cost_mean"],
        )


if __name__ == "__main__":
    unittest.main()
