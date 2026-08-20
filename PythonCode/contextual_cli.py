"""Command-line interface for contextual TD7 v2."""

import argparse
from pathlib import Path

from contextual_action import ACTION_MODES, REGION_B_RL
from contextual_dispatch import DISPATCH_FIRST_MATCH, DISPATCH_MODES
from contextual_reward_version_cfg import (
    REWARD_VERSION,
    REWARD_VERSIONS,
    canonical_reward_version,
)
from cocel_rl.algorithms.contextual_td7 import (
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contextual baseline, inference, and Phase 7 training runtime"
    )
    parser.add_argument(
        "--mode",
        choices=("baseline_only", "actor_inference", "training"),
        default="baseline_only",
    )
    parser.add_argument(
        "--reward-version",
        type=canonical_reward_version,
        choices=REWARD_VERSIONS,
        default=REWARD_VERSION,
        help=(
            "Compatibility flag for the single locked Reward N contract."
        ),
    )
    parser.add_argument("--action-enabled", action="store_true")
    parser.add_argument(
        "--action-mode", choices=ACTION_MODES, default=REGION_B_RL
    )
    parser.add_argument(
        "--dispatch-mode",
        choices=DISPATCH_MODES,
        default=DISPATCH_FIRST_MATCH,
        help=(
            "OHT selection for command 6: preserve iteration-order "
            "first-match, or minimize cumulative live command-0 rail cost."
        ),
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.05,
        help="Fixed applied-action scale for exp_residual mode only.",
    )
    parser.add_argument(
        "--num-stacks",
        type=int,
        default=1,
        help="Total observation frames including the current frame.",
    )
    parser.add_argument(
        "--stack-interval",
        type=int,
        default=1,
        help="Environment-step spacing between stacked frames.",
    )
    parser.add_argument("--curriculum-end-step", type=int, default=20_000)
    parser.add_argument("--curriculum-scale-start", type=float, default=1.0)
    parser.add_argument("--curriculum-scale-end", type=float, default=1.0)
    parser.add_argument(
        "--curriculum-shape",
        choices=("geometric", "linear"),
        default="geometric",
    )
    parser.add_argument(
        "--exploration-noise-std",
        type=float,
        default=0.10,
        help="Exploration noise std at global environment step 0.",
    )
    parser.add_argument(
        "--exploration-noise-final-std",
        type=float,
        default=0.02,
        help="Final exploration noise std after annealing.",
    )
    parser.add_argument(
        "--exploration-noise-anneal-steps",
        type=int,
        default=100_000,
        help="Post-warmup env steps over which noise reaches its final std.",
    )
    parser.add_argument("--exploration-noise-clip", type=float, default=0.20)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument(
        "--terminate-on-warmup-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "End the collection episode at the warm-up transition boundary "
            "so policy control starts from a clean next episode."
        ),
    )
    parser.add_argument("--episode-burnin-steps", type=int, default=0)
    parser.add_argument("--normalizer-freeze-steps", type=int, default=10_000)
    parser.add_argument(
        "--load-state-normalizer",
        type=Path,
        default=None,
        help=(
            "Load only a populated, frozen observation/state normalizer "
            "snapshot. Exact observation feature order, dimensions, topology, "
            "mapping, epsilon, and clip compatibility is required; a valid "
            "load makes effective action warm-up zero without restoring the "
            "actor, critic, reward state, replay, or checkpoint."
        ),
    )
    parser.add_argument(
        "--save-state-normalizer",
        type=Path,
        default=None,
        help=(
            "Atomically save the observation/state normalizer once both local "
            "and global statistics are populated and frozen."
        ),
    )
    parser.add_argument(
        "--reward-diagnostic-dir",
        default=None,
        help="Opt-in directory for contextual reward step/cycle JSONL diagnostics.",
    )
    parser.add_argument(
        "--reward-diagnostic-windows",
        default="0:1000,10000:11000,20000:21000",
        help="Start-inclusive:end-exclusive global-step windows.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-capacity-env-steps", type=int, default=100_000)
    replay_sampling = parser.add_mutually_exclusive_group()
    replay_sampling.add_argument(
        "--replay-buffer-rail",
        dest="replay_sampling_mode",
        action="store_const",
        const=REPLAY_SAMPLING_RAIL,
        help=(
            "Sample batch-size independent (environment step, controlled "
            "rail) transitions. This is the default and original behavior."
        ),
    )
    replay_sampling.add_argument(
        "--replay-buffer-snapshot",
        dest="replay_sampling_mode",
        action="store_const",
        const=REPLAY_SAMPLING_SNAPSHOT,
        help=(
            "Sample batch-size distinct environment steps and include every "
            "controlled rail from each selected snapshot. Requires --no-lap."
        ),
    )
    replay_sampling.add_argument(
        "--replay-buffer-random-rail",
        "--random-rail-mode",
        dest="replay_sampling_mode",
        action="store_const",
        const=REPLAY_SAMPLING_RANDOM_RAIL,
        help=(
            "Uniformly sample batch-size unique centered-rail transitions "
            "directly from the complete (environment step, controlled rail) "
            "replay pool. Requires --no-lap."
        ),
    )
    parser.set_defaults(replay_sampling_mode=REPLAY_SAMPLING_RAIL)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--minimum-replay-env-steps", type=int, default=100)
    parser.add_argument(
        "--minimum-action-enabled-env-steps", type=int, default=100
    )
    parser.add_argument("--updates-per-env-step", type=int, default=1)
    parser.add_argument("--learn-every-env-steps", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--sale", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--lap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--critic-loss-mode", choices=("auto", "huber", "mse"), default="auto"
    )
    parser.add_argument("--wandb-log-interval", type=int, default=10)
    parser.add_argument("--console-log-interval", type=int, default=100)
    parser.add_argument("--early-stop-queued-threshold", type=float, default=500.0)
    parser.add_argument("--max-stale-sim-time-ticks", type=int, default=5)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--resume-inference-until-replay-full",
        action="store_true",
        help=(
            "After checkpoint resume, keep the restored actor/critic frozen "
            "and collect actor-plus-exploration transitions until the new "
            "replay reaches replay-capacity-env-steps; enable learner updates "
            "on the following tick."
        ),
    )
    parser.add_argument(
        "--resume-deterministic-first-episode",
        action="store_true",
        help=(
            "After checkpoint resume, run the first resumed episode with the "
            "restored deterministic actor, zero exploration noise, and no "
            "learner updates. Keep its transitions in replay and resume the "
            "saved exploration schedule plus learner updates from the next "
            "episode once the ordinary minimum replay gate is satisfied."
        ),
    )
    parser.add_argument(
        "--rail-tat-diagnostic",
        default=None,
        help=(
            "Opt in to the legacy rail-TAT completion JSONL at this path."
        ),
    )
    parser.add_argument(
        "--rail-tat-diagnostic-max-step",
        type=int,
        default=1_000,
        help=(
            "Write rail-TAT JSONL only while global_step is below this "
            "exclusive cutoff. Zero disables event logging."
        ),
    )
    parser.add_argument(
        "--sim-end-time",
        type=int,
        default=45_000,
        help="Override PClient's simulator episode end time after every init/reset.",
    )
    parser.add_argument(
        "--smoke-report",
        type=Path,
        default=None,
        help="Continuously write aggregated live smoke metrics as JSON.",
    )
    parser.add_argument(
        "--smoke-report-interval",
        type=int,
        default=100,
        help="Write the smoke summary every N active ticks.",
    )
    return parser.parse_args()
