"""Four-variant SALE/LAP offline smoke and benchmark; no simulator access."""

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cocel_rl.algorithms.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
)
from test_contextual_observation import make_topology
from test_contextual_replay import FakeObservationBuilder, make_snapshot


EXP_META = {
    "cost_structure": "b_rl",
    "action_range": "0.05-scaled",
    "reward_version": "A",
    "centering": False,
    "note": "sale_lap_offline",
    "description": (
        "Four contextual TD7 SALE/LAP variants, 500 synthetic offline "
        "updates each plus batch latency/memory benchmark; no simulator."
    ),
}

VARIANTS = (
    (True, True),
    (False, True),
    (True, False),
    (False, False),
)


def make_replay(lap_enabled):
    topology = make_topology()
    replay = ContextualStepReplayBuffer(
        topology, FakeObservationBuilder(topology),
        capacity_env_steps=100, seed=2612, lap_enabled=lap_enabled,
    )
    rows = np.arange(len(topology.controlled_rail_ids), dtype=np.float32)
    for step in range(100):
        snapshot = make_snapshot(topology, step, done=(step == 99))
        policy = np.sin(rows * 0.013 + step * 0.07)[:, None]
        replay.push(replace(
            snapshot,
            policy_action=policy,
            applied_action=0.05 * policy,
            reward=np.tanh(rows * 0.001 + step * 0.01),
        ))
    return replay


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_variant(sale, lap, device, updates, benchmark_iterations):
    replay = make_replay(lap)
    config = ContextualLearnerConfig(
        batch_size=256, action_scale=0.05,
        sale_enabled=sale, lap_enabled=lap,
        minimum_replay_env_steps=1,
        minimum_action_enabled_env_steps=1,
        require_normalizer_frozen=False,
    )
    learner = ContextualTD7Learner(
        replay, network_config=ContextualNetworkConfig(),
        config=config, device=device, seed=2612,
    )
    losses = []
    started = time.perf_counter()
    for _ in range(updates):
        result = learner.update()
        losses.append(result.diagnostics["learner/critic_loss"])
    synchronize(device)
    smoke_seconds = time.perf_counter() - started
    if not np.isfinite(losses).all():
        raise FloatingPointError(f"{config.algorithm_variant}: non-finite loss")

    benchmarks = {}
    for batch_size in (256, 512, 1024):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        for _ in range(benchmark_iterations):
            sample_started = time.perf_counter()
            batch = replay.sample(batch_size, device=device)
            synchronize(device)
            sample_ms = (time.perf_counter() - sample_started) * 1000.0
            output = learner.update(batch)
            synchronize(device)
            rows.append({
                "replay_sampling_ms": sample_ms,
                **{
                    key: output.diagnostics[key]
                    for key in (
                        "timing/sale_forward_ms",
                        "timing/sale_backward_ms",
                        "timing/encoder_forward_ms",
                        "timing/critic_update_ms",
                        "timing/actor_update_ms",
                        "timing/total_learner_update_ms",
                    )
                },
            })
        keys = rows[0]
        benchmarks[str(batch_size)] = {
            key: {
                "median": float(np.median([row[key] for row in rows])),
                "p95": float(np.percentile([row[key] for row in rows], 95)),
            }
            for key in keys
        }
        benchmarks[str(batch_size)]["peak_gpu_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda" else 0.0
        )
    result = {
        "algorithm_variant": config.algorithm_variant,
        "sale_enabled": sale,
        "lap_enabled": lap,
        "critic_loss_mode": config.resolved_critic_loss_mode,
        "updates": updates,
        "benchmark_updates": benchmark_iterations * 3,
        "total_learner_updates": learner.learner_update_count,
        "smoke_seconds": smoke_seconds,
        "first_critic_loss": losses[0],
        "last_critic_loss": losses[-1],
        "actor_updates": learner.actor_update_count,
        "target_updates": learner.target_update_count,
        "sale_updates": learner.sale_update_count,
        "current_target_q_min": learner.current_target_q_min,
        "current_target_q_max": learner.current_target_q_max,
        "fixed_target_q_min": learner.fixed_target_q_min,
        "fixed_target_q_max": learner.fixed_target_q_max,
        "finite": True,
        "benchmark": benchmarks,
    }
    del learner, replay
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--benchmark-iterations", type=int, default=10)
    parser.add_argument(
        "--output", type=Path,
        default=Path("contextual_sale_lap_offline_benchmark.json"),
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "EXP_META": EXP_META,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else "CPU"
        ),
        "variants": [],
        "live_simulator_started": False,
    }
    for sale, lap in VARIANTS:
        result = run_variant(
            sale, lap, device, args.updates, args.benchmark_iterations
        )
        report["variants"].append(result)
        print(
            f"[offline] {result['algorithm_variant']}: "
            f"{result['updates']} updates finite"
        )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[offline] report={args.output.resolve()}")


if __name__ == "__main__":
    main()
