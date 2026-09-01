"""Explicit command-line overrides for the contextual runtime config."""

import argparse
from pathlib import Path

from oht_dispatching.config import DISPATCH_MODES
from oht_routing.mdp.action import ACTION_MODES
from oht_routing.mdp.reward.config import (
    REWARD_VERSIONS,
    canonical_reward_version,
)
from oht_routing.algorithms.rl.contextual_td7 import (
    REPLAY_SAMPLING_RANDOM_RAIL,
    REPLAY_SAMPLING_RAIL,
    REPLAY_SAMPLING_SNAPSHOT,
)
from oht_routing.runtime.stages import VALID_STAGES


def _parse_stage(value: str) -> int:
    try:
        stage = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"--stage must be one of {VALID_STAGES}"
        ) from error
    if stage not in VALID_STAGES:
        raise argparse.ArgumentTypeError(
            f"--stage must be one of {VALID_STAGES}"
        )
    return stage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contextual baseline, inference, and Phase 7 training runtime",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mode",
        choices=("baseline_only", "actor_inference", "training"),
    )
    parser.add_argument(
        "--reward-version",
        type=canonical_reward_version,
        choices=REWARD_VERSIONS,
        help=(
            "Compatibility flag for the single locked Reward P contract."
        ),
    )
    parser.add_argument("--action-enabled", action="store_true")
    parser.add_argument(
        "--action-mode", choices=ACTION_MODES
    )
    parser.add_argument(
        "--dispatch-mode",
        choices=DISPATCH_MODES,
        help=(
            "OHT selection for command 6: preserve iteration-order "
            "first-match, or minimize cumulative live command-0 rail cost."
        ),
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        help="Fixed applied-action scale for exp_residual mode only.",
    )
    parser.add_argument(
        "--rl-cost-lambda",
        type=float,
        help=(
            "Free-flow residual coefficient; the default is 0.5 and "
            "action_scale is not applied to it."
        ),
    )
    parser.add_argument(
        "--num-stacks",
        type=int,
        help="Total observation frames including the current frame.",
    )
    parser.add_argument(
        "--stack-interval",
        type=int,
        help="Environment-step spacing between stacked frames.",
    )
    parser.add_argument("--curriculum-end-step", type=int)
    parser.add_argument("--curriculum-scale-start", type=float)
    parser.add_argument("--curriculum-scale-end", type=float)
    parser.add_argument(
        "--curriculum-shape",
        choices=("geometric", "linear"),
    )
    parser.add_argument(
        "--exploration-noise-std",
        type=float,
        help=(
            "Exploration noise std at global environment step 0. Fresh "
            "Stage 2 defaults both noise endpoints to 0.05 when neither "
            "endpoint is supplied."
        ),
    )
    parser.add_argument(
        "--exploration-noise-final-std",
        type=float,
        help=(
            "Final exploration noise std after annealing. Fresh Stage 2 "
            "defaults both endpoints to 0.05 when neither is supplied."
        ),
    )
    parser.add_argument(
        "--exploration-noise-anneal-steps",
        type=int,
        help="Post-warmup env steps over which noise reaches its final std.",
    )
    parser.add_argument("--exploration-noise-clip", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument(
        "--terminate-on-warmup-complete",
        action=argparse.BooleanOptionalAction,
        help=(
            "End the collection episode at the warm-up transition boundary "
            "so policy control starts from a clean next episode."
        ),
    )
    parser.add_argument("--normalizer-freeze-steps", type=int)
    parser.add_argument(
        "--load-state-normalizer",
        dest="load_state_normalizer_path",
        type=Path,
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
        dest="save_state_normalizer_path",
        type=Path,
        help=(
            "Atomically save the observation/state normalizer once both local "
            "and global statistics are populated and frozen."
        ),
    )
    parser.add_argument(
        "--reward-diagnostic-dir",
        help="Opt-in directory for contextual reward step/cycle JSONL diagnostics.",
    )
    parser.add_argument(
        "--reward-diagnostic-windows",
        help="Start-inclusive:end-exclusive global-step windows.",
    )
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--num-sim",
        type=int,
        help=(
            "Number of simulator connections managed by this launch. "
            "Values greater than one require an explicit port list."
        ),
    )
    parser.add_argument(
        "--port",
        "--ports",
        dest="sim_ports",
        type=int,
        nargs="+",
        metavar="PORT",
        help=(
            "One simulator listen port per configured simulator. The single-"
            "sim default continues to use wpconfig.json when omitted."
        ),
    )
    parser.add_argument(
        "--replay-capacity-env-steps",
        type=int,
        help=(
            "Number of environment-step transitions retained by the packed "
            "CPU replay. The default is defined in ContextualRuntimeConfig."
        ),
    )
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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--minimum-replay-env-steps", type=int)
    parser.add_argument(
        "--minimum-action-enabled-env-steps", type=int
    )
    parser.add_argument("--updates-per-env-step", type=int)
    parser.add_argument("--learn-every-env-steps", type=int)
    parser.add_argument(
        "--wandb",
        dest="wandb_enabled",
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable W&B logging. By default it is enabled for training and "
            "actor_inference, and disabled for baseline_only."
        ),
    )
    parser.add_argument(
        "--save_data",
        "--save-data",
        dest="save_data_enabled",
        action="store_true",
        help=(
            "Save full rail, OHT, job, action, and environment snapshots for "
            "the first 2,000 actor_inference ticks."
        ),
    )
    parser.add_argument(
        "--sale",
        dest="sale_enabled",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--lap",
        dest="lap_enabled",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help=(
            "Use directional cross-attention. The default encoder keeps rail "
            "embeddings but flattens the 15 encoded neighbors per direction."
        ),
    )
    parser.add_argument(
        "--critic-loss-mode", choices=("auto", "huber", "mse")
    )
    parser.add_argument("--wandb-log-interval", type=int)
    parser.add_argument("--console-log-interval", type=int)
    parser.add_argument("--early-stop-queued-threshold", type=float)
    parser.add_argument("--max-stale-sim-time-ticks", type=int)
    parser.add_argument("--checkpoint-root")
    parser.add_argument(
        "--periodic-checkpoint-interval",
        type=int,
        help=(
            "Save an immutable periodic checkpoint every N environment "
            "steps. Stage 2 uses learner-active stage2_env_steps; other "
            "runs use global environment steps. An explicit value remains "
            "launch-controlled when resuming from a checkpoint."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint", dest="resume_checkpoint_path"
    )
    parser.add_argument(
        "--load-stage1-policy",
        "--load_stage1_policy",
        dest="load_stage1_policy_path",
        type=Path,
        help=(
            "Stage 2 only: load the frozen Stage 1 encoder, actor, SALE "
            "state encoder, observation normalizers, and saved applied-action "
            "scale. A fresh Stage 2 learner is initialized once from that "
            "policy path; checkpoint resume never overwrites the resumed "
            "policy. The critic, optimizers, replay, counters, reward state, "
            "applied-action scale, and RNG state remain Stage 2-owned."
        ),
    )
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
        "--resume-warmstart-steps",
        type=int,
        help=(
            "After checkpoint resume in training mode, run N deterministic "
            "throwaway environment ticks with no replay insertion, "
            "exploration, or learner updates, then request an episode reset. "
            "Ordinary resumed training begins in the next episode."
        ),
    )
    parser.add_argument(
        "--rail-tat-diagnostic",
        dest="rail_tat_diagnostic_path",
        help=(
            "Opt in to the legacy rail-TAT completion JSONL at this path."
        ),
    )
    parser.add_argument(
        "--rail-tat-diagnostic-max-step",
        type=int,
        help=(
            "Write rail-TAT JSONL only while global_step is below this "
            "exclusive cutoff. Zero disables event logging."
        ),
    )
    episode_length = parser.add_mutually_exclusive_group()
    episode_length.add_argument(
        "--stage",
        type=_parse_stage,
        metavar="{1,2}",
        help=(
            "Select the fixed simulator contract: Stage 1 ends at 2000; "
            "Stage 2 ends at 45000 and requires --load-stage1-policy."
        ),
    )
    episode_length.add_argument(
        "--sim-end-time",
        type=int,
        help="Override PClient's simulator episode end time after every init/reset.",
    )
    return parser.parse_args()
