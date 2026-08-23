"""Reward O identity and locked parameters for contextual TD7 v4.

Historical reward profiles remain available from the
major-scoped experiment logs.  The v4 runtime has one executable reward
contract: Reward O.
"""

from __future__ import annotations

from dataclasses import dataclass


REWARD_VERSION = "O"
TAT_PENALTY_START = 160.0
TAT_WINDOW_SECONDS = 300.0

TAT_SIGNAL_RECENT_COMPLETED_300S = "recent_completed_tat_300s_mean"

RAIL_REWARD_FREE_FLOW_NEUTRAL_2 = "free_flow_neutral_2"


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
    tat_signal_description="one_sided_recent_completed_tat_300s_mean",
    tat_window_seconds=TAT_WINDOW_SECONDS,
    tat_termination_enabled=True,
    tat_termination_grace_steps=10_000,
    tat_termination_threshold=200.0,
    tat_termination_patience=300,
    tat_termination_inclusive=True,
    terminal_tat_penalty=-20.0,
)

REWARD_VERSIONS = (REWARD_VERSION,)


def canonical_reward_version(version: str) -> str:
    canonical = str(version).strip().upper().replace("-", "_")
    if canonical != REWARD_VERSION:
        raise ValueError("contextual TD7 v4 supports only reward_version='O'")
    return REWARD_VERSION


def reward_contract(version: str = REWARD_VERSION) -> RewardContract:
    canonical_reward_version(version)
    return REWARD_O_CONTRACT


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

__all__ = (
    "RAIL_REWARD_FREE_FLOW_NEUTRAL_2",
    "REWARD_O_CONTRACT",
    "REWARD_O_PROFILE",
    "REWARD_VERSION",
    "REWARD_VERSIONS",
    "RewardContract",
    "TAT_PENALTY_START",
    "TAT_SIGNAL_RECENT_COMPLETED_300S",
    "TAT_WINDOW_SECONDS",
    "canonical_reward_version",
    "reward_contract",
)
