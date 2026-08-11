"""TAT is the acceptance metric, so its logging contract is regression-tested.

Before these metrics existed, a run's per-episode final TAT had to be
reconstructed by parsing the raw .wandb file, and the run summary carried only
`run/status` - the final episode and the best result were never recorded.
"""
import unittest

from contextual_action import B_RL_NEUTRAL
from contextual_wandb import (
    EPISODE_METRIC_KEYS,
    TAT_BASELINE,
    TAT_TARGET,
    WANDB_METRIC_KEYS,
    ContextualWandbLogger,
    tat_derived_metrics,
    tat_improvement_pct,
)
from test_contextual_training_runtime import training_runtime


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.defined = {}
        self.finished = False

    def define_metric(self, name, **kwargs):
        self.defined[name] = kwargs

    def log(self, payload, step=None):
        self.logged.append((step, dict(payload)))

    def finish(self):
        self.finished = True


def logger_with_fake_run():
    logger = ContextualWandbLogger.__new__(ContextualWandbLogger)
    logger.enabled = True
    logger._finished = False
    logger._episode_finals = []
    logger._best_episode = None
    logger.run = FakeRun()
    logger._define_metrics()
    return logger


class TatDerivedMetricTests(unittest.TestCase):
    def test_baseline_and_target_match_the_project_criterion(self):
        self.assertAlmostEqual(TAT_BASELINE, 174.4236, places=4)
        self.assertAlmostEqual(TAT_TARGET, TAT_BASELINE * 0.95, places=6)

    def test_improvement_percent_sign_and_scale(self):
        self.assertAlmostEqual(tat_improvement_pct(TAT_BASELINE), 0.0)
        self.assertAlmostEqual(tat_improvement_pct(TAT_TARGET), 5.0, places=6)
        # A TAT above baseline is a regression and must read negative.
        self.assertLess(tat_improvement_pct(TAT_BASELINE + 1.0), 0.0)

    def test_derived_metrics_are_exported(self):
        metrics = tat_derived_metrics(
            170.4, b_rl_mean=0.9, rail_tat_share=0.06
        )
        self.assertAlmostEqual(metrics["tat/level"], 170.4)
        self.assertGreater(metrics["tat/improvement_pct"], 0.0)
        self.assertAlmostEqual(
            metrics["tat/gap_to_target"], 170.4 - TAT_TARGET, places=6
        )
        self.assertAlmostEqual(
            metrics["b_rl/level_deviation"], abs(0.9 - B_RL_NEUTRAL), places=9
        )
        self.assertAlmostEqual(metrics["credit/rail_tat_share"], 0.06)
        for key in metrics:
            self.assertIn(key, WANDB_METRIC_KEYS)

    def test_zero_level_before_first_completion_emits_no_improvement(self):
        metrics = tat_derived_metrics(0.0)
        self.assertNotIn("tat/improvement_pct", metrics)
        self.assertNotIn("tat/gap_to_target", metrics)


class EpisodeLoggingTests(unittest.TestCase):
    def test_best_episode_is_the_minimum_not_the_last(self):
        logger = logger_with_fake_run()
        for index, tat in enumerate([174.3, 170.4, 174.9, 175.6], start=1):
            logger.log_episode(index, {"final_tat": tat, "steps": 45_000}, index)
        payload = logger.run.logged[-1][1]
        self.assertEqual(payload["best/episode_index"], 2)
        self.assertAlmostEqual(payload["best/episode_tat"], 170.4)
        self.assertAlmostEqual(payload["episode/final_tat"], 175.6)
        self.assertEqual(payload["trend/episodes_since_best"], 2.0)

    def test_regressing_run_reports_a_positive_trend(self):
        logger = logger_with_fake_run()
        for index, tat in enumerate([170.0, 172.0, 174.0, 176.0], start=1):
            logger.log_episode(index, {"final_tat": tat}, index)
        self.assertAlmostEqual(logger.run.logged[-1][1]["trend/tat_per_episode"], 2.0)

    def test_summary_records_the_verdict_without_opening_a_chart(self):
        logger = logger_with_fake_run()
        for index, tat in enumerate([174.3, 170.4, 175.6], start=1):
            logger.log_episode(index, {"final_tat": tat}, index)
        logger._finish("completed")
        summary = logger.run.summary
        self.assertEqual(summary["run/status"], "completed")
        self.assertAlmostEqual(summary["summary/best_episode_tat"], 170.4)
        self.assertEqual(summary["summary/best_episode_index"], 2)
        self.assertAlmostEqual(
            summary["summary/best_improvement_pct"],
            tat_improvement_pct(170.4),
        )
        self.assertEqual(summary["summary/episodes_completed"], 3)
        self.assertEqual(summary["summary/episodes_since_best"], 1)
        self.assertEqual(summary["summary/target_reached"], 0.0)
        self.assertEqual(summary["summary/regressing"], 1.0)
        self.assertTrue(logger.run.finished)

    def test_target_reached_is_flagged(self):
        logger = logger_with_fake_run()
        logger.log_episode(1, {"final_tat": TAT_TARGET - 0.5}, 1)
        logger._finish("completed")
        self.assertEqual(logger.run.summary["summary/target_reached"], 1.0)
        self.assertEqual(logger.run.logged[-1][1]["episode/reached_target"], 1.0)

    def test_episode_metrics_are_declared_against_the_episode_axis(self):
        logger = logger_with_fake_run()
        for key in EPISODE_METRIC_KEYS:
            self.assertEqual(
                logger.run.defined[key].get("step_metric"), "episode/index"
            )
        self.assertEqual(
            logger.run.defined["episode/final_tat"].get("summary"), "min"
        )
        self.assertEqual(
            logger.run.defined["tat/improvement_pct"].get("summary"), "max"
        )
        # env/tat is a within-episode cumulative mean; its last value is not a
        # run-level result, so no auto-summary should be produced for it.
        self.assertEqual(logger.run.defined["env/tat"].get("summary"), "none")


class RuntimeEpisodeAccumulationTests(unittest.TestCase):
    def test_runtime_accumulates_and_flushes_episode_aggregates(self):
        runtime, pclient = training_runtime(warmup_steps=0)
        runtime.Algorithm(pclient)
        runtime.Algorithm(pclient)
        self.assertIn("tat/level", runtime.last_diagnostics)
        self.assertIn("final_tat", runtime._episode_accum)

        captured = []
        runtime.wandb_logger.log_episode = (
            lambda index, stats, step: captured.append((index, stats, step))
        )
        finished_episode = runtime.episode_id
        runtime.Reset(pclient)

        self.assertEqual(len(captured), 1)
        index, stats, _step = captured[0]
        self.assertEqual(index, finished_episode)
        self.assertIn("final_tat", stats)
        self.assertIn("mean_tat", stats)
        self.assertEqual(runtime._episode_accum, {})
        self.assertEqual(runtime.episode_id, finished_episode + 1)

    def test_flush_is_idempotent_so_run_end_never_double_logs(self):
        runtime, pclient = training_runtime(warmup_steps=0)
        runtime.Algorithm(pclient)
        captured = []
        runtime.wandb_logger.log_episode = (
            lambda index, stats, step: captured.append(index)
        )
        runtime.flush_episode_metrics()
        runtime.flush_episode_metrics()
        self.assertEqual(len(captured), 1)


if __name__ == "__main__":
    unittest.main()
