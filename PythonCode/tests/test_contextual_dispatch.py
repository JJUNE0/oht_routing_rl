import unittest
from types import SimpleNamespace

import numpy as np

from ClientAlgorithm_contextual import (
    ClientAlgorithm,
    ContextualRuntimeConfig,
)
from contextual_dispatch import (
    DISPATCH_COST,
    DISPATCH_FIRST_MATCH,
)


def make_job(job_id=100, from_node=9):
    return SimpleNamespace(
        ID=job_id,
        State=1,
        OHTId=0,
        FromNode=from_node,
        CarrierTypes=[1],
        RunningAreaTyes=[1],
        RouteList=[],
    )


def make_oht(route, *, state=0, carrier_types=None, area=1):
    return SimpleNamespace(
        State=state,
        JobID=0,
        DispatchedCommand=0,
        CarrierTypes=[1] if carrier_types is None else carrier_types,
        RunningAreaType=area,
        RouteList=route,
    )


def make_runtime(dispatch_mode):
    runtime = object.__new__(ClientAlgorithm)
    runtime.config = ContextualRuntimeConfig(
        device="cpu", dispatch_mode=dispatch_mode
    )
    runtime.total_steps = 0
    runtime.latest_dispatch_live_cost_by_rail = {}
    runtime.latest_dispatch_cost_tick = None
    runtime.dispatch_cost_ready = False
    runtime.last_diagnostics = {}
    runtime.parameterDw = {}
    runtime.parameterPassTimes = {}
    runtime.parameterC = {}
    runtime._reset_dispatch_diagnostics()
    return runtime


def assignment_oht_ids(assignments):
    return {
        job_id: assignment["oht_id"]
        for job_id, assignment in assignments.items()
    }


class ContextualDispatchTests(unittest.TestCase):
    def test_first_match_preserves_existing_iteration_order(self):
        runtime = make_runtime(DISPATCH_FIRST_MATCH)
        pclient = SimpleNamespace(OHT_DIC={
            10: make_oht([1, 2, 9]),
            20: make_oht([3, 9]),
        })
        runtime.latest_dispatch_live_cost_by_rail = {
            1: 10.0, 2: 10.0, 3: 1.0, 9: 1.0,
        }
        runtime.dispatch_cost_ready = True

        assigned = runtime.Assign(pclient, [make_job()])

        self.assertEqual(assignment_oht_ids(assigned), {100: 10})
        self.assertEqual(
            runtime.last_diagnostics["dispatch/mode_first_match"], 1.0
        )

    def test_pickup_index_preserves_first_occurrence_and_used_oht(self):
        runtime = make_runtime(DISPATCH_FIRST_MATCH)
        pclient = SimpleNamespace(OHT_DIC={
            10: make_oht([9, 1, 9]),
            20: make_oht([9]),
        })

        assigned = runtime.Assign(
            pclient,
            [make_job(job_id=100), make_job(job_id=200)],
        )

        self.assertEqual(
            assignment_oht_ids(assigned),
            {100: 10, 200: 20},
        )
        self.assertEqual(
            runtime.last_diagnostics["dispatch/selected_pickup_hops_mean"],
            1.0,
        )

    def test_cost_selects_lowest_pickup_prefix_sum(self):
        runtime = make_runtime(DISPATCH_COST)
        pclient = SimpleNamespace(OHT_DIC={
            10: make_oht([1, 2, 9, 50]),
            20: make_oht([3, 9, 60]),
        })
        runtime.latest_dispatch_live_cost_by_rail = {
            1: 10.0,
            2: 10.0,
            3: 1.0,
            9: 1.0,
            # Costs after the first pickup occurrence must not affect score.
            50: 0.0,
            60: 1000.0,
        }
        runtime.dispatch_cost_ready = True

        assigned = runtime.Assign(pclient, [make_job()])

        self.assertEqual(assignment_oht_ids(assigned), {100: 20})
        self.assertEqual(
            runtime.last_diagnostics[
                "dispatch/selection_changed_from_first_match_ratio"
            ],
            1.0,
        )
        self.assertEqual(
            runtime.last_diagnostics["dispatch/selected_path_cost_mean"],
            2.0,
        )
        self.assertEqual(
            runtime.last_diagnostics["dispatch/best_second_margin_mean"],
            19.0,
        )
        self.assertEqual(
            runtime.last_diagnostics["dispatch/first_match_path_cost_mean"],
            21.0,
        )
        self.assertEqual(
            runtime.last_diagnostics["dispatch/cost_saving_vs_first_mean"],
            19.0,
        )
        self.assertAlmostEqual(
            runtime.last_diagnostics[
                "dispatch/cost_saving_vs_first_ratio_mean"
            ],
            19.0 / 21.0,
        )
        self.assertEqual(
            runtime.last_diagnostics[
                "dispatch/candidate_cost_spread_mean"
            ],
            19.0,
        )
        self.assertEqual(
            runtime.last_diagnostics["dispatch/multi_candidate_ratio"],
            1.0,
        )
        self.assertEqual(
            runtime.last_diagnostics[
                "dispatch/strict_cost_improvement_ratio"
            ],
            1.0,
        )
        self.assertEqual(
            runtime.last_diagnostics[
                "dispatch/cost_snapshot_ready_ratio"
            ],
            1.0,
        )

    def test_cost_ties_use_hops_then_oht_id(self):
        runtime = make_runtime(DISPATCH_COST)
        pclient = SimpleNamespace(OHT_DIC={
            30: make_oht([1, 9]),
            20: make_oht([9]),
            10: make_oht([9]),
        })
        runtime.latest_dispatch_live_cost_by_rail = {1: 0.0, 9: 1.0}
        runtime.dispatch_cost_ready = True

        assigned = runtime.Assign(pclient, [make_job()])

        self.assertEqual(assignment_oht_ids(assigned), {100: 10})
        self.assertEqual(
            runtime.last_diagnostics["dispatch/changed_from_first_ratio"],
            1.0,
        )
        self.assertEqual(
            runtime.last_diagnostics[
                "dispatch/strict_cost_improvement_ratio"
            ],
            0.0,
        )
        self.assertEqual(
            runtime.last_diagnostics["dispatch/cost_saving_vs_first_mean"],
            0.0,
        )
        self.assertEqual(
            runtime.last_diagnostics[
                "dispatch/candidate_cost_spread_mean"
            ],
            0.0,
        )

    def test_cost_before_first_snapshot_falls_back_to_first_match(self):
        runtime = make_runtime(DISPATCH_COST)
        pclient = SimpleNamespace(OHT_DIC={
            10: make_oht([1, 9]),
            20: make_oht([9]),
        })

        assigned = runtime.Assign(pclient, [make_job()])

        self.assertEqual(assignment_oht_ids(assigned), {100: 10})
        self.assertEqual(
            runtime.last_diagnostics[
                "dispatch/cost_snapshot_fallback_count"
            ],
            1.0,
        )
        self.assertEqual(
            runtime.last_diagnostics[
                "dispatch/cost_snapshot_fallback_ratio"
            ],
            1.0,
        )
        self.assertEqual(
            runtime.last_diagnostics[
                "dispatch/cost_snapshot_ready_ratio"
            ],
            0.0,
        )
        self.assertEqual(
            runtime.last_diagnostics["dispatch/cost_snapshot_age_steps"],
            -1.0,
        )

    def test_ready_snapshot_rejects_missing_or_nonfinite_cost(self):
        for invalid_cost in (None, np.nan, np.inf, -1.0):
            with self.subTest(invalid_cost=invalid_cost):
                runtime = make_runtime(DISPATCH_COST)
                runtime.latest_dispatch_live_cost_by_rail = {9: 1.0}
                if invalid_cost is not None:
                    runtime.latest_dispatch_live_cost_by_rail[1] = invalid_cost
                runtime.dispatch_cost_ready = True
                pclient = SimpleNamespace(
                    OHT_DIC={10: make_oht([1, 9])}
                )

                with self.assertRaisesRegex(
                    ValueError, "mode=cost, rail_id=1"
                ):
                    runtime.Assign(pclient, [make_job()])

    def test_existing_eligibility_and_no_assignment_are_preserved(self):
        for mode in (DISPATCH_FIRST_MATCH, DISPATCH_COST):
            with self.subTest(mode=mode):
                runtime = make_runtime(mode)
                pclient = SimpleNamespace(OHT_DIC={
                    10: make_oht([9], state=3),
                    20: make_oht([9], carrier_types=[2]),
                    30: make_oht([8]),
                })
                self.assertEqual(
                    dict(runtime.Assign(pclient, [make_job()])),
                    {},
                )
                self.assertEqual(
                    runtime.last_diagnostics[
                        "dispatch/zero_candidate_ratio_per_eligible_job"
                    ],
                    1.0,
                )
                self.assertEqual(
                    runtime.last_diagnostics[
                        "dispatch/selected_ratio_per_eligible_job"
                    ],
                    0.0,
                )

    def test_existing_assignments_and_busy_ohts_are_never_overridden(self):
        for mode in (DISPATCH_FIRST_MATCH, DISPATCH_COST):
            with self.subTest(mode=mode):
                runtime = make_runtime(mode)
                already_waiting = make_job(job_id=100)
                already_waiting.State = 3
                already_waiting.OHTId = 10
                reservated = make_job(job_id=200)
                reservated.State = 2
                reservated.OHTId = 20
                queued = make_job(job_id=300)
                pclient = SimpleNamespace(OHT_DIC={
                    10: make_oht([9]),
                    20: make_oht([9]),
                    30: make_oht([9], state=2),
                    40: make_oht([9]),
                    50: make_oht([9]),
                })
                pclient.OHT_DIC[40].JobID = 999
                pclient.OHT_DIC[50].DispatchedCommand = 888

                assigned = runtime.Assign(
                    pclient,
                    [already_waiting, reservated, queued],
                )

                self.assertEqual(dict(assigned), {})

    def test_queued_job_with_oht_and_unassigned_nonqueued_job_are_skipped(self):
        runtime = make_runtime(DISPATCH_FIRST_MATCH)
        queued_with_oht = make_job(job_id=100)
        queued_with_oht.OHTId = 99
        unassigned_reservated = make_job(job_id=200)
        unassigned_reservated.State = 2
        pclient = SimpleNamespace(OHT_DIC={
            10: make_oht([9]),
        })

        assigned = runtime.Assign(
            pclient,
            [queued_with_oht, unassigned_reservated],
        )

        self.assertEqual(assigned, {})

    def test_each_busy_oht_marker_excludes_the_candidate(self):
        busy_ohts = {
            "stage": make_oht([9], state=1),
            "move_to_load": make_oht([9], state=2),
            "job_owner": make_oht([9]),
            "dispatch_owner": make_oht([9]),
        }
        busy_ohts["job_owner"].JobID = 123
        busy_ohts["dispatch_owner"].DispatchedCommand = 456

        for reason, oht in busy_ohts.items():
            with self.subTest(reason=reason):
                runtime = make_runtime(DISPATCH_FIRST_MATCH)
                pclient = SimpleNamespace(OHT_DIC={10: oht})

                self.assertEqual(runtime.Assign(pclient, [make_job()]), {})

    def test_selected_pickup_path_is_returned_in_assignment_payload(self):
        runtime = make_runtime(DISPATCH_FIRST_MATCH)
        job = make_job()
        pclient = SimpleNamespace(OHT_DIC={
            10: make_oht([1, 2, 9, 99]),
        })

        assigned = runtime.Assign(pclient, [job])

        self.assertEqual(
            assigned,
            {100: {"oht_id": 10, "pickup_path": [1, 2, 9]}},
        )
        self.assertEqual(job.RouteList, [])

    def test_cost_snapshot_is_frozen_and_reset_clears_it(self):
        runtime = make_runtime(DISPATCH_COST)
        runtime.topology = SimpleNamespace(
            all_rail_ids=np.asarray([1, 2], dtype=np.int64)
        )
        runtime.total_steps = 7
        costs = np.asarray([3.0, 4.0])
        runtime._capture_dispatch_cost_snapshot(costs)
        costs[:] = 99.0

        self.assertEqual(
            runtime.latest_dispatch_live_cost_by_rail,
            {1: 3.0, 2: 4.0},
        )
        self.assertEqual(runtime.latest_dispatch_cost_tick, 7)
        self.assertEqual(runtime._dispatch_cost_snapshot_step, 7)
        runtime.total_steps = 9
        self.assertEqual(
            runtime._dispatch_diagnostics()[
                "dispatch/cost_snapshot_age_steps"
            ],
            2.0,
        )

        runtime.episode_id = 0
        runtime.episode_steps = 1
        runtime.last_observation = object()
        runtime.last_controlled_action = object()
        runtime.last_policy_action = object()
        runtime.last_exploratory_action = object()
        runtime._last_sim_time = 1.0
        runtime._stale_sim_time_ticks = 1
        runtime._burnin_last_applied_action = object()
        runtime._burnin_previous_applied_action = object()
        runtime.transition_aligner = None
        runtime.reward_builder = None
        runtime.Reset(SimpleNamespace())

        self.assertFalse(runtime.dispatch_cost_ready)
        self.assertEqual(runtime.latest_dispatch_live_cost_by_rail, {})
        self.assertIsNone(runtime.latest_dispatch_cost_tick)
        self.assertIsNone(runtime._dispatch_cost_snapshot_step)


if __name__ == "__main__":
    unittest.main()
