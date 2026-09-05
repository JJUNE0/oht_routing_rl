"""Reward Q identity and locked parameters for contextual TD7.

Rewards O and P remain recorded below as historical profiles. Reward Q keeps
Reward P's unbounded one-sided penalty on the simulator's cumulative
``pclient.TotalTat`` level and every global coefficient, and changes only how
per-rail credit is distributed: the local channel moves its budget off the
predicted-traffic forecast onto delay and density signals, and the rail-cycle
outcome term becomes a signed discriminator around the measured route-ratio
median. The current runtime executes Reward Q only.
"""

from __future__ import annotations

from dataclasses import dataclass


REWARD_VERSION = "Q"
TAT_PENALTY_START = 160.0
TAT_WINDOW_SECONDS = 300.0

TAT_SIGNAL_RECENT_COMPLETED_300S = "recent_completed_tat_300s_mean"
TAT_SIGNAL_CUMULATIVE_TOTAL = "cumulative_total_tat"

# Historical label used while the rail-cycle term was neutral at ratio 2.0.
RAIL_REWARD_FREE_FLOW_NEUTRAL_2 = "free_flow_neutral_2"
# Reward Q neutral point. 1.70 is the measured p50 of route_time /
# route_free_flow_time over the 9,008 completed cycles in
# results/reward_diagnostics/v4_reward_o_activescale_seed0, so the term splits
# roughly 51/49 positive/negative instead of paying out on 79.9% of cycles.
RAIL_REWARD_FREE_FLOW_NEUTRAL_1_7 = "free_flow_neutral_1_7"
RAIL_FREE_FLOW_NEUTRAL_RATIO = 1.70


@dataclass(frozen=True)
class RewardContract:
    version: str
    tat_signal_mode: str
    tat_signal_description: str
    tat_window_seconds: float
    tat_termination_enabled: bool
    tat_termination_grace_steps: int
    tat_termination_threshold: float
    tat_termination_patience: int
    tat_termination_inclusive: bool
    terminal_tat_penalty: float


REWARD_O_CONTRACT = RewardContract(
    version="O",
    tat_signal_mode=TAT_SIGNAL_RECENT_COMPLETED_300S,
    tat_signal_description="centered_recent_completed_tat_300s_mean",
    tat_window_seconds=TAT_WINDOW_SECONDS,
    tat_termination_enabled=True,
    tat_termination_grace_steps=10_000,
    tat_termination_threshold=200.0,
    tat_termination_patience=300,
    tat_termination_inclusive=True,
    terminal_tat_penalty=-20.0,
)

REWARD_P_CONTRACT = RewardContract(
    version="P",
    tat_signal_mode=TAT_SIGNAL_CUMULATIVE_TOTAL,
    tat_signal_description="one_sided_cumulative_total_tat_penalty",
    # Reward P has no TAT window. The builder still maintains the historical
    # tracker for diagnostics, but it cannot influence Reward P.
    tat_window_seconds=0.0,
    tat_termination_enabled=True,
    tat_termination_grace_steps=10_000,
    tat_termination_threshold=200.0,
    tat_termination_patience=300,
    tat_termination_inclusive=True,
    terminal_tat_penalty=-20.0,
)

REWARD_Q_CONTRACT = RewardContract(
    version="Q",
    tat_signal_mode=TAT_SIGNAL_CUMULATIVE_TOTAL,
    tat_signal_description="one_sided_cumulative_total_tat_penalty",
    # Reward Q inherits Reward P's TAT contract unchanged; only the per-rail
    # credit distribution differs.
    tat_window_seconds=0.0,
    tat_termination_enabled=True,
    tat_termination_grace_steps=10_000,
    tat_termination_threshold=200.0,
    tat_termination_patience=300,
    tat_termination_inclusive=True,
    terminal_tat_penalty=-20.0,
)

REWARD_N_CONTRACT = RewardContract(
    version="N",
    tat_signal_mode=TAT_SIGNAL_CUMULATIVE_TOTAL,
    tat_signal_description="one_sided_cumulative_total_tat_penalty",
    tat_window_seconds=0.0,
    tat_termination_enabled=True,
    tat_termination_grace_steps=10_000,
    tat_termination_threshold=200.0,
    tat_termination_patience=300,
    tat_termination_inclusive=True,
    terminal_tat_penalty=-20.0,
)

# Executable profiles, newest first. Q is the default; N is the restored
# 2026-08-20 run 1y9sx4a5 coefficient set, kept selectable as a fallback.
REWARD_VERSIONS = ("Q", "N")


def canonical_reward_version(version: str) -> str:
    canonical = str(version).strip().upper().replace("-", "_")
    if canonical not in REWARD_VERSIONS:
        raise ValueError(
            "contextual TD7 supports only reward_version in "
            f"{REWARD_VERSIONS}"
        )
    return canonical


def reward_contract(version: str = REWARD_VERSION) -> RewardContract:
    canonical = canonical_reward_version(version)
    return {
        "Q": REWARD_Q_CONTRACT,
        "N": REWARD_N_CONTRACT,
    }[canonical]


def reward_profile(version: str = REWARD_VERSION) -> dict:
    """Locked coefficient set for an executable reward version."""
    canonical = canonical_reward_version(version)
    return dict({
        "Q": REWARD_Q_PROFILE,
        "N": REWARD_N_PROFILE,
    }[canonical])


REWARD_O_PROFILE = {
    "global_alpha": 0.5,
    "local_alpha": 0.5,
    "rail_tat_weight": 30.0,
    "rail_free_flow_neutral_ratio": 2.0,
    "smooth_b_rl_weight": 0.25,
    "smooth_exp_residual_weight": 0.5,
    "tat_weight": 4.3,
    "tat_window_seconds": TAT_WINDOW_SECONDS,
    "op_weight": 0.0,
    "backlog_weight": 0.0007,
    "backlog_growth_enabled": True,
    "backlog_growth_horizon": 300,
    "backlog_growth_scale": 30.0,
    "backlog_growth_weight": 0.17,
    "idle_reserve_target": 200.0,
    "idle_reserve_scale": 50.0,
    "idle_reserve_weight": 0.09,
    "local_oht_weight": 0.0,
    "local_predicted_oht_weight": 0.05,
    "local_stop_weight": 0.12,
    "local_idle_weight": 0.0,
    "local_capacity_weight": 0.0,
    "use_tat": True,
    "use_op": False,
    "use_backlog": True,
    "tat_reference": 165.0,
    "op_reference": 0.80,
    "local_reward_scale": 2.0,
    "rail_tat_clip": 1.0,
}

# Keep a distinct object so future P changes cannot mutate historical O data.
REWARD_P_PROFILE = dict(REWARD_O_PROFILE)

# Reward Q. Global terms are copied from P unchanged; the differences are the
# per-rail credit weights. The coefficients below were sized against measured
# signal magnitudes on
# results/environment_capture/capture_20260821_081812_v2.2.0_actor_inference
# so that delay/congestion terms carry roughly 18% of the reward budget
# instead of 3.3%, and the predicted-traffic forecast drops from 21.4% to
# under 5%.
REWARD_Q_PROFILE = dict(REWARD_P_PROFILE)
REWARD_Q_PROFILE.update({
    # Trimmed only enough to land the Stage 2 budget at ~30% TAT; raising the
    # delay terms does the rest of the dilution.
    "tat_weight": 4.0,
    # The forecast is accurate but measures demand, not delay, and it is
    # already an observation feature. It keeps a small tie-breaking weight.
    "local_predicted_oht_weight": 0.01,
    # The only dense local term whose sign tracks actual delay.
    "local_stop_weight": 0.30,
    # OHT count per metre of rail. Unlike raw occupancy this is not a proxy
    # for rail length (measured cross-rail corr with length -0.04 vs +0.60).
    "local_density_weight": 5.5,
    "rail_free_flow_neutral_ratio": RAIL_FREE_FLOW_NEUTRAL_RATIO,
    "rail_tat_weight": 660.0,
    # Raised with the weight so the clip still binds at the same route-time
    # share (~11.5%). Holding it at 1.0 would truncate exactly the rails an
    # OHT dwelt longest on.
    "rail_tat_clip": 22.0,
})

# Reward N. The coefficient set that ran W&B run 1y9sx4a5 on 2026-08-20,
# restored verbatim from that run's saved config and kept selectable with
# `--reward-version N`. Only the coefficients are restored: the observation,
# action mapping, Stage 1/2 contract, replay, and network are the current
# ones, so this is not a reproduction of that run.
#
# The TAT clamp is part of this profile. Commit 2ad214e, the code 1y9sx4a5
# actually ran, computed `max(0.0, cur_tat - TAT_PENALTY_START)` with
# `tat_one_sided=True`; the clamp was only dropped later, in v6.0.0.
REWARD_N_PROFILE = dict(REWARD_Q_PROFILE)
REWARD_N_PROFILE.update({
    "tat_weight": 11.0,
    "op_weight": 4.0,
    "use_op": True,
    "backlog_weight": 0.0004,
    "backlog_growth_weight": 0.16,
    "idle_reserve_weight": 0.2,
    "local_oht_weight": 0.3,
    "local_predicted_oht_weight": 0.075,
    "local_stop_weight": 0.3,
    "local_capacity_weight": 0.1,
    # Reward N predates the density term.
    "local_density_weight": 0.0,
    "rail_tat_weight": 30.0,
    "rail_tat_clip": 1.0,
    "rail_free_flow_neutral_ratio": 2.0,
})

__all__ = (
    "RAIL_FREE_FLOW_NEUTRAL_RATIO",
    "RAIL_REWARD_FREE_FLOW_NEUTRAL_1_7",
    "RAIL_REWARD_FREE_FLOW_NEUTRAL_2",
    "REWARD_N_CONTRACT",
    "REWARD_N_PROFILE",
    "REWARD_O_CONTRACT",
    "REWARD_O_PROFILE",
    "REWARD_P_CONTRACT",
    "REWARD_P_PROFILE",
    "REWARD_Q_CONTRACT",
    "REWARD_Q_PROFILE",
    "reward_profile",
    "REWARD_VERSION",
    "REWARD_VERSIONS",
    "RewardContract",
    "TAT_PENALTY_START",
    "TAT_SIGNAL_CUMULATIVE_TOTAL",
    "TAT_SIGNAL_RECENT_COMPLETED_300S",
    "TAT_WINDOW_SECONDS",
    "canonical_reward_version",
    "reward_contract",
)
