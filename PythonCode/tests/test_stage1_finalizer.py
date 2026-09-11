from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from finalize_ud7_stage1 import audit_training, sha256, verify_inference, evaluation_root_for, ROOT


class Stage1FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.run, self.checkpoints = root / "run", root / "checkpoints"
        self.run.mkdir()
        (self.checkpoints / "episodes").mkdir(parents=True)
        (self.checkpoints / "best_tat").mkdir()
        source = root / "input.db"
        source.write_bytes(b"fixed source fixture")
        self.state = {"training_run": str(self.run), "checkpoint_root": str(self.checkpoints),
                      "planned_episodes": 8, "input_path": str(source), "input_sha256": sha256(source)}
        self.manifest = {"status": "complete", "mode": "stage1", "episodes": 8,
                         "completed_episodes": 8, "end_time": 2000,
                         "inputs": [str(source)], "name": "test_{episode:04d}"}
        self.write(self.run / "run.json", self.manifest)
        self.rows, self.records = [], []
        for episode in range(1, 9):
            tat = 150.0 if episode >= 7 else 180.0
            eligible = episode >= 6
            end = {"tat_s": tat, "sim_time": 1999.0, "episode_steps": 2000,
                   "full_horizon": True, "early_termination": False,
                   "trained_episode": eligible, "normalizers_ready": episode >= 5,
                   "eligible_for_best": eligible}
            checkpoint = self.checkpoints / "episodes" / f"episode_{episode:06d}_step_{episode*2000:09d}.pt"
            checkpoint.write_bytes(f"checkpoint {episode}".encode())
            record = {"episode_id": episode, "runtime_env_step": episode * 2000, "stage": 1,
                      "episode_end": end, "normalizers_frozen": episode >= 5, "training_failed": False,
                      "source_checkpoint": str(checkpoint), "sha256": sha256(checkpoint)}
            self.write(checkpoint.with_suffix(".json"), record)
            self.records.append(record)
            self.rows.append({"episode_id": episode, "step": episode * 2000, "episode_steps": 2000,
                              "mode": "training", "stage": 1, "tat_s": tat, "sim_time": 1999.0,
                              "early_termination": False, "full_horizon": True, "training_failed": False})
            with closing(sqlite3.connect(self.run / f"test_{episode:04d}.db")) as database:
                database.execute("CREATE TABLE result (id INTEGER)")
                database.commit()
            (self.run / f"test_{episode:04d}.log").write_text("[headless] completed sim_time=2000", encoding="utf-8")
        self.write_rows()
        self.set_best(7)

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_rows(self):
        (self.run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")

    def set_best(self, episode):
        record = self.records[episode - 1]
        self.write(self.checkpoints / "best_tat/selection.json", record)
        shutil.copyfile(record["source_checkpoint"], self.checkpoints / "best_tat/checkpoint.pt")

    def test_audit_checks_every_episode_and_earliest_lowest_tie(self):
        best, records = audit_training(self.state)
        self.assertEqual(len(records), 8)
        self.assertEqual(best["episode_id"], 7)
        self.assertEqual(best["episode_end"]["tat_s"], 150.0)

    def test_evaluation_has_a_short_unique_root_outside_nested_training_output(self):
        first = evaluation_root_for(self.run)
        self.assertEqual(first, evaluation_root_for(self.run))
        self.assertNotEqual(first, evaluation_root_for(self.run / "another_run"))
        self.assertEqual(first.parent, ROOT / "results")
        self.assertFalse(first.is_relative_to(self.run))
        self.assertLess(len(str(first / "run_0910_2344_v10.3.1_b_rl_0.0-1.0_Q_headless/runtime/episode_0001/Pinokio.Headless.exe.config")), 260)

    def test_live_or_partial_batch_is_not_completion(self):
        self.manifest["status"] = "running"
        self.write(self.run / "run.json", self.manifest)
        with self.assertRaisesRegex(RuntimeError, "not completed"):
            audit_training(self.state)

    def test_nonwinning_checkpoint_corruption_is_caught(self):
        Path(self.records[0]["source_checkpoint"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "checkpoint identity"):
            audit_training(self.state)

    def test_wrong_best_or_later_tie_is_rejected(self):
        for episode in (6, 8):
            self.set_best(episode)
            with self.subTest(episode=episode), self.assertRaisesRegex(RuntimeError, "not the lowest"):
                audit_training(self.state)

    def test_missing_episode_is_rejected(self):
        Path(self.records[3]["source_checkpoint"]).with_suffix(".json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or extra"):
            audit_training(self.state)

    def test_warmup_cannot_be_promoted(self):
        record = self.records[4]
        record["episode_end"]["eligible_for_best"] = True
        self.write(Path(record["source_checkpoint"]).with_suffix(".json"), record)
        with self.assertRaisesRegex(RuntimeError, "eligibility"):
            audit_training(self.state)

    def test_input_identity_and_native_errors_are_checked(self):
        source = Path(self.state["input_path"])
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "input identity"):
            audit_training(self.state)
        source.write_bytes(b"fixed source fixture")
        (self.run / "test_0002.log").write_text("[ERROR] native failure\n[headless] completed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Simulator reported an error"):
            audit_training(self.state)

    def test_summary_tat_must_match_checkpoint_provenance(self):
        self.rows[6]["tat_s"] = 100
        self.write_rows()
        with self.assertRaisesRegex(RuntimeError, "metrics and checkpoint disagree"):
            audit_training(self.state)

    def prepare_inference(self):
        pinned = self.checkpoints / "best_tat/checkpoint.pt"
        manifest = {"status": "complete", "mode": "inference", "episodes": 1,
                    "completed_episodes": 1, "end_time": 2000, "checkpoint": str(pinned),
                    "inputs": [self.state["input_path"]]}
        self.write(self.run / "run.json", manifest)
        self.rows = [{**self.rows[0], "mode": "actor_inference", "stage": None, "end_time": 2000}]
        self.write_rows()
        (self.run / "python.log").write_text("[checkpoint-loaded]", encoding="utf-8")
        shutil.copyfile(self.run / "test_0001.db", self.run / "best_stage1_inference.db")
        shutil.copyfile(self.run / "test_0001.log", self.run / "best_stage1_inference.log")
        return pinned, manifest

    def test_inference_requires_complete_horizon_and_same_inputs(self):
        pinned, manifest = self.prepare_inference()
        self.assertEqual(verify_inference(self.run, pinned, self.state["input_path"])["tat_s"], 180.0)
        self.rows[0]["stage"] = 2
        self.write_rows()
        with self.assertRaisesRegex(RuntimeError, "valid full-horizon"):
            verify_inference(self.run, pinned, self.state["input_path"])
        self.rows[0]["stage"] = None
        self.rows[0]["early_termination"] = True
        self.write_rows()
        with self.assertRaisesRegex(RuntimeError, "valid full-horizon"):
            verify_inference(self.run, pinned, self.state["input_path"])
        self.rows[0]["early_termination"] = False
        self.write_rows()
        for key, value in (("checkpoint", str(pinned.with_suffix(".other"))), ("inputs", [str(self.run / "other.db")])):
            with self.subTest(key=key):
                self.write(self.run / "run.json", {**manifest, key: value})
                with self.assertRaisesRegex(RuntimeError, "incomplete or used another"):
                    verify_inference(self.run, pinned, self.state["input_path"])

    def test_inference_requires_checkpoint_load_and_native_result(self):
        pinned, _ = self.prepare_inference()
        (self.run / "python.log").write_text("server started", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "No confirmation"):
            verify_inference(self.run, pinned, self.state["input_path"])
        (self.run / "python.log").write_text("[checkpoint-loaded]", encoding="utf-8")
        (self.run / "best_stage1_inference.log").write_text("[ERROR] failed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Simulator reported an error"):
            verify_inference(self.run, pinned, self.state["input_path"])


if __name__ == "__main__":
    unittest.main()
