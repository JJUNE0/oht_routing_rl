"""Manual Phase 6 offline smoke/benchmark. Never connects to the simulator."""

import json
import time
from dataclasses import replace

import numpy as np
import torch

from oht_routing.algorithms.rl.contextual_td7 import (
    ContextualLearnerConfig,
    ContextualNetworkConfig,
    ContextualStepReplayBuffer,
    ContextualTD7Learner,
)
from test_contextual_observation import make_topology
from test_contextual_replay import FakeObservationBuilder, make_snapshot


EXP_META = {
    "cost_structure": "b_rl",
    "action_range": "0.1-scaled",
    "reward_version": "A",
    "centering": False,
    "note": "offline500",
    "description": (
        "Phase 6 synthetic 100-step snapshot replay with 500 offline "
        "contextual TD7 learner updates; no simulator and no W&B."
    ),
}


def replay_100():
    topology = make_topology()
    builder = FakeObservationBuilder(topology)
    replay = ContextualStepReplayBuffer(
        topology, builder, capacity_env_steps=100, seed=2612
    )
    rows = np.arange(len(topology.controlled_rail_ids), dtype=np.float32)
    for step in range(100):
        snapshot = make_snapshot(topology, step, done=(step == 99))
        policy = np.sin(rows * 0.013 + step * 0.07)[:, None]
        applied = 0.1 * policy
        previous_applied = 0.1 * np.sin(
            rows * 0.013 + max(step - 1, 0) * 0.07
        )[:, None]
        replay.push(replace(
            snapshot,
            previous_applied_action=previous_applied,
            policy_action=policy,
            applied_action=applied,
            next_previous_applied_action=applied,
            reward=np.tanh(rows * 0.001 + step * 0.01),
        ))
    return replay


def parameter_vector(module):
    return torch.cat([p.detach().flatten().cpu() for p in module.parameters()])


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    replay = replay_100()
    config = ContextualLearnerConfig(
        batch_size=256,
        action_scale=0.1,
        minimum_replay_env_steps=100,
        minimum_action_enabled_env_steps=100,
        require_normalizer_frozen=False,
        sale_enabled=False,
        lap_enabled=False,
    )
    learner = ContextualTD7Learner(
        replay, network_config=ContextualNetworkConfig(),
        config=config, device=device, seed=2612,
    )
    initial = {
        "encoder": parameter_vector(learner.encoder),
        "actor": parameter_vector(learner.actor),
        "critic": parameter_vector(learner.critic),
    }
    losses = []
    started = time.perf_counter()
    for _ in range(500):
        result = learner.update()
        losses.append(result.diagnostics["learner/critic_loss"])
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    changes = {
        name: float(torch.linalg.vector_norm(
            parameter_vector(getattr(learner, name)) - before
        ))
        for name, before in initial.items()
    }

    performance = {}
    for batch_size in (256, 512, 1024):
        timings = []
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        for _ in range(20):
            batch_started = time.perf_counter()
            output = learner.update(replay.sample(batch_size, device=device))
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append({
                **output.diagnostics,
                "wall_ms": (time.perf_counter() - batch_started) * 1000,
            })
        performance[str(batch_size)] = {
            key: float(np.median([row[key] for row in timings]))
            for key in (
                "replay/sample_materialize_ms",
                "replay/host_to_device_ms",
                "timing/encoder_forward_ms",
                "timing/critic_update_ms",
                "timing/actor_update_ms",
                "timing/total_learner_update_ms",
                "wall_ms",
            )
        }
        performance[str(batch_size)]["peak_cuda_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        )
    print(json.dumps({
        "EXP_META": EXP_META,
        "device": device,
        "updates": 500,
        "elapsed_seconds": elapsed,
        "loss_finite": bool(np.isfinite(losses).all()),
        "loss_min": float(np.min(losses)),
        "loss_max": float(np.max(losses)),
        "last_diagnostics": learner.last_diagnostics,
        "parameter_change_l2": changes,
        "performance_median": performance,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
