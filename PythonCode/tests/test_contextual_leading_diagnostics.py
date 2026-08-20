import unittest
from types import SimpleNamespace

import numpy as np

from oht_routing.telemetry.reward_diagnostic import (
    LeadingIndicatorTracker,
    RunningPearson,
)


def route_metrics(value=1.5, available=True):
    result = {
        "lead/route_ratio/available": float(available),
        "lead/route_ratio/mean": value if available else 0.0,
        "lead/route_ratio/max": value if available else 0.0,
        "lead/route_ratio/ratio_gt_2": float(value > 2.0 and available),
        "lead/route_ratio/negative_reward_cycle_ratio": float(
            value > 2.0 and available
        ),
    }
    for percentile in (50, 75, 90, 95, 99):
        result[f"lead/route_ratio/p{percentile}"] = (
            value if available else 0.0
        )
    return result


def pclient(step=0, *, job_ids=()):
    states = (0, 2, 3, 4, 5)
    return SimpleNamespace(
        WaitingCommandCount=2 * step,
        QueuedCommandCount=step,
        CompletedCommandCount=1,
        TransferCommandCount=step % 23,
        TotalTat=float(step),
        TotalOhtOperationRate=0.75 + 0.0001 * step,
        OHT_DIC={
            index: SimpleNamespace(State=state)
            for index, state in enumerate(states)
        },
        JOB_DIC={
            job_id: SimpleNamespace(ID=job_id) for job_id in job_ids
        },
    )


class LeadingIndicatorTrackerTests(unittest.TestCase):
    def test_rolling_horizons_are_exact_bounded_and_reset(self):
        tracker = LeadingIndicatorTracker()
        diagnostics = None
        for step in range(1_101):
            diagnostics = tracker.update(
                pclient(step),
                predicted_oht=np.asarray([step, step + 1], dtype=np.float64),
                route_ratio_diagnostics=route_metrics(),
            )
        self.assertEqual(diagnostics["lead/backlog/delta_100"], 300.0)
        self.assertEqual(diagnostics["lead/backlog/delta_300"], 900.0)
        self.assertEqual(diagnostics["lead/backlog/delta_500"], 1_500.0)
        self.assertEqual(diagnostics["lead/backlog/delta_1000"], 3_000.0)
        self.assertEqual(tracker.history_size, 1_001)
        self.assertEqual(diagnostics["lead/predicted_oht/p90"], 1_100.9)
        self.assertEqual(diagnostics["lead/flow/completion_count_300"], 300.0)

        tracker.reset_episode()
        reset = tracker.update(
            pclient(5),
            predicted_oht=np.asarray([1.0]),
            route_ratio_diagnostics=route_metrics(available=False),
        )
        self.assertEqual(tracker.history_size, 1)
        self.assertEqual(reset["lead/backlog/delta_100"], 0.0)
        self.assertEqual(reset["leadlag/sample_count_500"], 0.0)
        self.assertEqual(reset["lead/route_ratio/available"], 0.0)

    def test_arrival_bootstrap_new_ids_and_reuse_fail_safe(self):
        tracker = LeadingIndicatorTracker()
        bootstrap = tracker.update(
            pclient(job_ids=(10, 11)),
            predicted_oht=np.asarray([1.0]),
            route_ratio_diagnostics=route_metrics(),
        )
        self.assertEqual(bootstrap["lead/flow/new_job_count"], 0.0)
        new = tracker.update(
            pclient(1, job_ids=(10, 11, 12)),
            predicted_oht=np.asarray([1.0]),
            route_ratio_diagnostics=route_metrics(),
        )
        self.assertEqual(new["lead/flow/new_job_count"], 1.0)
        tracker.update(
            pclient(2, job_ids=(10, 12)),
            predicted_oht=np.asarray([1.0]),
            route_ratio_diagnostics=route_metrics(),
        )
        reused = tracker.update(
            pclient(3, job_ids=(10, 11, 12)),
            predicted_oht=np.asarray([1.0]),
            route_ratio_diagnostics=route_metrics(),
        )
        self.assertEqual(reused["lead/flow/job_id_reuse_count"], 1.0)
        self.assertEqual(reused["lead/flow/arrival_available"], 0.0)

    def test_delayed_synthetic_indicator_has_positive_500_step_correlation(self):
        tracker = LeadingIndicatorTracker()
        tat = [0.0] * 1_100
        indicators = [1.0 + (step % 17) for step in range(1_100)]
        diagnostics = None
        for step in range(1_100):
            if step >= 500:
                tat[step] = tat[step - 500] + indicators[step - 500]
            client = pclient(step)
            client.TotalTat = tat[step]
            diagnostics = tracker.update(
                client,
                predicted_oht=np.asarray([indicators[step]]),
                route_ratio_diagnostics=route_metrics(),
            )
        self.assertGreater(
            diagnostics["leadlag/predicted_p90_vs_future_tat_500"],
            0.99,
        )
        self.assertEqual(diagnostics["leadlag/sample_count_500"], 600.0)

        tracker.reset_episode()
        reset = tracker.update(
            pclient(),
            predicted_oht=np.asarray([1.0]),
            route_ratio_diagnostics=route_metrics(),
        )
        self.assertEqual(reset["leadlag/sample_count_500"], 0.0)

    def test_constant_pearson_is_safe(self):
        correlation = RunningPearson()
        for value in range(10):
            correlation.update(1.0, value)
        self.assertEqual(correlation.correlation, 0.0)


if __name__ == "__main__":
    unittest.main()
