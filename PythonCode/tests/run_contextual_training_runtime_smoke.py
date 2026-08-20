"""Phase 7 fake 1,500-tick runtime smoke; no socket, W&B, or simulator."""

import json
import time

import numpy as np

from test_contextual_training_runtime import (
    FakeLearner,
    TrainingObservationBuilder,
)
from test_contextual_observation import make_topology
from test_contextual_runtime import make_runtime_pclient
from ClientAlgorithm_contextual import ClientAlgorithm, ContextualRuntimeConfig


EXP_META = {
    "cost_structure": "b_rl",
    "action_range": "0.0-1.0",
    "reward_version": "E",
    "centering": False,
    "note": "epburnin",
    "description": (
        "Fake 1,500-tick runtime smoke with a three-step deterministic "
        "episode burn-in; no simulator and W&B disabled."
    ),
}


def run():
    config = ContextualRuntimeConfig(
        mode="training",
        action_enabled=True,
        action_scale=0.05,
        exploration_noise_std=0.10,
        exploration_noise_clip=0.20,
        warmup_steps=100,
        episode_burnin_steps=3,
        normalizer_freeze_steps=100,
        replay_capacity_env_steps=1_000,
        batch_size=1_024,
        minimum_replay_env_steps=50,
        minimum_action_enabled_env_steps=50,
        updates_per_env_step=1,
        learn_every_env_steps=1,
        latest_checkpoint_interval=10_000,
        periodic_checkpoint_interval=10_000,
        wandb_enabled=False,
        device="cpu",
        seed=2612,
        sale_enabled=False,
        lap_enabled=False,
    )
    runtime = ClientAlgorithm(config)
    runtime.topology = make_topology()
    runtime.observation_builder = TrainingObservationBuilder(
        runtime.topology, freeze_steps=100
    )
    pclient = make_runtime_pclient()
    runtime._ensure_initialized(pclient)
    runtime.learner = FakeLearner()
    runtime.checkpoint_loaded = True

    # Actor/network correctness is covered separately; avoid 1,500 large CPU
    # attention forwards in this runtime state-machine smoke.
    zero_policy = np.zeros(4996, np.float32)
    runtime._actor_inference = lambda observation, **kwargs: (
        zero_policy.copy(),
        {
            "runtime/tensor_conversion_ms": 0.0,
            "runtime/host_to_device_ms": 0.0,
            "runtime/encoder_actor_ms": 0.0,
            "runtime/device_to_host_ms": 0.0,
        },
    )
    boundary = np.flatnonzero(
        runtime.topology.physical_index_to_controlled_row == -1
    )
    baseline_error = 0.0
    update_start_tick = None
    updates_by_tick = []
    timings = []
    started = time.perf_counter()
    for tick in range(1_500):
        before = runtime.learner.learner_update_count
        result = runtime.Algorithm(pclient)
        after = runtime.learner.learner_update_count
        if tick < 3:
            assert runtime.last_diagnostics["burnin/active"] == 1.0
            assert runtime.last_diagnostics["burnin/action_source"] == 1.0
            assert runtime.last_diagnostics["action/exploration_noise_std"] == 0.0
            assert runtime.replay_buffer.push_count == 0
            assert after == before == 0
            np.testing.assert_array_equal(
                runtime.last_controlled_action, zero_policy
            )
        elif tick == 3:
            assert runtime.last_diagnostics["burnin/active"] == 0.0
            assert runtime.replay_buffer.push_count == 0
            assert runtime.transition_aligner.pending is not None
        elif tick == 4:
            assert runtime.replay_buffer.push_count == 1
        if after > before and update_start_tick is None:
            update_start_tick = tick
        updates_by_tick.append(after - before)
        baseline = np.asarray([
            pclient.RAILLINE_DIC[int(rail_id)].DistancePerVelocity
            for rail_id in runtime.topology.all_rail_ids
        ])
        baseline_error = max(
            baseline_error,
            float(np.max(np.abs(
                result.final_cost[boundary] - baseline[boundary]
            ))),
        )
        timings.append(runtime.last_diagnostics["runtime/total_ms"])
        if not all(
            np.isfinite(value)
            for value in runtime.last_diagnostics.values()
            if isinstance(value, (int, float))
        ):
            raise FloatingPointError(f"non-finite diagnostics at tick {tick}")

    print(json.dumps({
        "EXP_META": EXP_META,
        "ticks": 1500,
        "warmup_updates": 0,
        "update_start_tick_zero_based": update_start_tick,
        "learner_updates": runtime.learner.learner_update_count,
        "max_updates_per_tick": max(updates_by_tick),
        "replay_env_steps": runtime.replay_buffer.size_env_steps,
        "replay_push_count": runtime.replay_buffer.push_count,
        "action_enabled_env_steps": runtime.action_enabled_env_steps,
        "boundary_baseline_max_error": baseline_error,
        "observation_calls": runtime.observation_builder.calls,
        "wandb_run": runtime.wandb_logger.run is not None,
        "all_finite": True,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_ms": {
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
            "p99": float(np.percentile(timings, 99)),
        },
        "last_diagnostics": runtime.last_diagnostics,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
