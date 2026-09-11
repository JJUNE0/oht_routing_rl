from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulator.evaluate import evaluate


class EvaluationWindowTests(unittest.TestCase):
    def make_db(self, path, factor=1):
        with closing(sqlite3.connect(path)) as con:
            con.execute("CREATE TABLE COMMAND_LOG (OHT_NAME TEXT, ACTIVATED_TIME TEXT, ASSIGNED_TIME TEXT, COMPLETED_TIME TEXT)")
            for activated, minutes in [("2026-02-16 07:00:01", 2),
                                       ("2026-02-16 09:00:01", 3),
                                       ("2026-02-16 18:59:00", 5)]:
                completed = datetime.fromisoformat(activated) + timedelta(minutes=minutes * factor)
                con.execute("INSERT INTO COMMAND_LOG VALUES (?,?,?,?)", ("oht1", activated, activated, str(completed)))
            con.commit()

    def test_custom_window_requires_its_own_baseline_and_keeps_completion_tail(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            current, baseline = Path(directory)/"current.db", Path(directory)/"baseline.db"
            self.make_db(current)
            self.make_db(baseline, factor=2)
            args = SimpleNamespace(save_dir=directory, db_filename="current.db",
                                   start_time="2026-02-16 07:00:00", end_time="2026-02-16 19:00:00")
            raw = evaluate(args)
            self.assertEqual(raw["command_count"], 3)
            self.assertAlmostEqual(raw["tat_s"], 200)
            self.assertIsNone(raw["baseline_tat_s"])
            self.assertIsNone(raw["tat_improvement_rate"])
            args.baseline_db = baseline
            compared = evaluate(args)
            self.assertAlmostEqual(compared["baseline_tat_s"], 400)
            self.assertAlmostEqual(compared["tat_improvement_rate"], 0.5)

    def test_legacy_window_keeps_legacy_constants(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            self.make_db(Path(directory)/"current.db")
            score = evaluate(SimpleNamespace(save_dir=directory, db_filename="current.db"))
            self.assertEqual(score["command_count"], 2)
            self.assertEqual(score["tat_s"], 240)
            self.assertAlmostEqual(score["baseline_tat_s"], 174.4236)

    def test_partial_rollout_has_no_improvement_claim(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            self.make_db(Path(directory)/"current.db")
            with closing(sqlite3.connect(Path(directory)/"current.db")) as con:
                con.execute("DELETE FROM COMMAND_LOG WHERE ACTIVATED_TIME > '2026-02-16 18:00:00'")
                con.commit()
            score = evaluate(SimpleNamespace(save_dir=directory, db_filename="current.db"))
            self.assertFalse(score["reaches_window_end"])
            self.assertIsNone(score["tat_improvement_rate"])


if __name__ == "__main__":
    unittest.main()
