from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from simulator.final_window import score_final_window, score_meeting_window


class FinalWindowTests(unittest.TestCase):
    def test_activation_boundaries_completion_tail_and_timestamp_tat(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.db"
            with closing(sqlite3.connect(path)) as con:
                con.execute("CREATE TABLE COMMAND_LOG (ACTIVATED_TIME TEXT, COMPLETED_TIME TEXT, TOTAL_TIME REAL)")
                con.executemany("INSERT INTO COMMAND_LOG VALUES (?,?,?)", [
                    ("2026-02-16 09:00:00", "2026-02-16 09:30:00", 1800),
                    ("2026-02-16 09:00:01", "2026-02-16 09:02:01", 120.7),
                    ("2026-02-16 18:59:00", "2026-02-16 19:02:00", 180.8),
                    ("2026-02-16 19:00:00", "2026-02-16 19:20:00", 1200),
                ])
                con.commit()
            score = score_final_window(path, full_rollout=True)
            self.assertEqual(score["eval_command_count"], 2)
            self.assertEqual(score["eval_tat_s"], 150.0)
            self.assertAlmostEqual(score["eval_baseline_tat_s"], 174.4236)
            self.assertTrue(score["eval_tat_target_met"])
            meeting = score_meeting_window(path, full_rollout=True)
            self.assertEqual(meeting["eval_command_count"], 3)
            self.assertEqual(meeting["eval_tat_s"], 700.0)
            self.assertTrue(meeting["eval_window_complete"])
            self.assertIsNone(meeting["eval_baseline_tat_s"])
            self.assertIsNone(meeting["eval_improvement_pct"])
            self.assertIsNone(meeting["eval_tat_target_met"])
            # The same partial artifact must not be presented as a final success.
            partial = score_final_window(path, full_rollout=False)
            self.assertFalse(partial["eval_window_complete"])
            self.assertIsNone(partial["eval_tat_target_met"])

    def test_stage1_has_no_samples_in_the_final_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.db"
            with closing(sqlite3.connect(path)) as con:
                con.execute("CREATE TABLE COMMAND_LOG (ACTIVATED_TIME TEXT, COMPLETED_TIME TEXT)")
                con.execute("INSERT INTO COMMAND_LOG VALUES ('2026-02-16 07:01:00', '2026-02-16 07:04:00')")
                con.commit()
            score = score_final_window(path, full_rollout=True)
            self.assertEqual(score["eval_completed_samples"], 0)
            self.assertIsNone(score["eval_tat_s"])
            self.assertIsNone(score["eval_improvement_pct"])
            self.assertIsNone(score["eval_tat_target_met"])


if __name__ == "__main__":
    unittest.main()
