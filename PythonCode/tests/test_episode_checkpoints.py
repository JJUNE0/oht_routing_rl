import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oht_routing.runtime.episode_checkpoints import episode_selection, save_episode
from test_contextual_training_runtime import training_runtime
from test_contextual_checkpoint import components
from oht_routing.algorithms.rl.contextual_td7.checkpoint import (
    save_contextual_checkpoint, load_frozen_contextual_policy,
)
from oht_routing.mdp.reward.config import REWARD_VERSION
import torch


class EpisodeCheckpointTests(unittest.TestCase):
    def selection(self, **overrides):
        args = dict(diagnostics={"env/tat": 165, "env/sim_time": 1999},
                    episode_steps=2000, total_steps=14000, warmup_steps=10000,
                    sim_end_time=2000, normalizers_ready=True, learner_updates=100)
        args.update(overrides)
        return episode_selection(**args)

    def test_best_requires_full_trained_valid_episode(self):
        self.assertTrue(self.selection()["eligible_for_best"])
        cases = [
            dict(total_steps=10000), dict(learner_updates=0),
            dict(normalizers_ready=False),
            dict(diagnostics={"env/tat": 100, "env/sim_time": 999}),
            dict(diagnostics={"env/tat": 0, "env/sim_time": 1999}),
            dict(diagnostics={"env/tat": float("nan"), "env/sim_time": 1999}),
            dict(diagnostics={"env/tat": 100, "env/sim_time": 1999, "termination/done": 1}),
        ]
        for case in cases:
            with self.subTest(case=case):
                selection = self.selection(**case)
                self.assertFalse(selection["eligible_for_best"])
                json.dumps(selection, allow_nan=False)

    def test_all_episodes_saved_but_only_lower_eligible_tat_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def writer(path, metadata):
                calls.append(metadata["episode_id"])
                path.write_bytes(str(metadata["episode_id"]).encode())
            def record(episode, tat, sim_time=1999):
                metadata = dict(episode_id=episode, runtime_env_step=episode * 2000,
                    episode_end=self.selection(diagnostics={"env/tat": tat, "env/sim_time": sim_time}))
                return save_episode(root, metadata, writer)
            record(1, 170)
            record(2, 160)
            record(2, 150)  # v=1 followed by reset cannot overwrite the snapshot.
            record(3, 165)
            record(4, 100, sim_time=500)
            record(5, 160)  # Ties preserve the first best.
            self.assertEqual(calls, [1, 2, 3, 4, 5])
            self.assertEqual(len(list((root / "episodes").glob("*.pt"))), 5)
            self.assertEqual((root / "best_tat/checkpoint.pt").read_bytes(), b"2")
            self.assertEqual(json.loads((root / "best_tat/selection.json").read_text())["episode_id"], 2)
            record(6, 159)  # Selection persists across calls; no in-memory best needed.
            self.assertEqual((root / "best_tat/checkpoint.pt").read_bytes(), b"6")

    def test_failed_save_does_not_publish_best(self):
        with tempfile.TemporaryDirectory() as directory:
            def fail(*args):
                raise OSError("disk full")
            with self.assertRaises(OSError):
                save_episode(directory, dict(episode_id=1, runtime_env_step=2000,
                    episode_end=self.selection()), fail)
            self.assertFalse((Path(directory) / "best_tat/selection.json").exists())

    def test_terminal_then_reset_saves_old_episode_once(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, pclient = training_runtime(stage=1, checkpoint_root=directory)
            runtime.Algorithm(pclient)
            runtime.episode_id = 7
            runtime.episode_steps = 2000
            runtime.total_steps = 14000
            runtime.last_diagnostics.update({"env/tat": 165, "env/sim_time": 1999})
            def writer(path, *args, **kwargs):
                Path(path).write_bytes(b"episode-policy")
            with patch("oht_routing.runtime.client.save_contextual_checkpoint", side_effect=writer) as save:
                runtime.on_terminal()
                runtime.Reset(pclient)
            self.assertEqual(save.call_count, 1)
            meta = save.call_args.kwargs["runtime_metadata"]
            self.assertEqual(meta["episode_id"], 7)
            self.assertEqual(meta["episode_end"]["tat_s"], 165)
            self.assertEqual(runtime.episode_id, 8)

    def test_reset_without_terminal_still_saves_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, pclient = training_runtime(stage=1, checkpoint_root=directory)
            runtime.Algorithm(pclient)
            with patch("oht_routing.runtime.client.save_contextual_checkpoint",
                       side_effect=lambda path, *a, **kw: Path(path).write_bytes(b"policy")) as save:
                runtime.Reset(pclient)
            self.assertEqual(save.call_count, 1)
            self.assertFalse(save.call_args.kwargs["runtime_metadata"]["episode_end"]["eligible_for_best"])

    def test_episode_best_is_loadable_as_frozen_stage2_prefix(self):
        source, builder, reward = components(sale=True, lap=True)
        source.set_applied_action_scale(1.0)
        with tempfile.TemporaryDirectory() as directory:
            metadata = dict(stage=1, episode_id=7, runtime_env_step=14000,
                            checkpoint_kind="episode_end", episode_end=self.selection())
            def writer(path, metadata):
                save_contextual_checkpoint(path, source, observation_builder=builder,
                    reward_builder=reward, runtime_metadata=metadata)
            save_episode(directory, metadata, writer)
            target, target_builder, _ = components(seed=99, sale=True, lap=True, populate_replay=False)
            policy = load_frozen_contextual_policy(
                Path(directory) / "best_tat/checkpoint.pt", target,
                observation_builder=target_builder, expected_reward_version=REWARD_VERSION,
                initialize_fresh_learner_policy=True,
            )
            self.assertEqual(policy.runtime_metadata["episode_end"]["tat_s"], 165)
            self.assertEqual(policy.applied_action_scale, 1.0)
            self.assertFalse(any(p.requires_grad for p in policy.actor.parameters()))
            for name, tensor in source.actor.state_dict().items():
                torch.testing.assert_close(tensor, policy.actor.state_dict()[name])
                torch.testing.assert_close(tensor, target.actor.state_dict()[name])


if __name__ == "__main__":
    unittest.main()
