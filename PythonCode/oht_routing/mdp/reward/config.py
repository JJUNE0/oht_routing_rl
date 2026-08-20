"""Reward N identity and locked parameters for contextual TD7 v2.

Historical reward profiles remain available from the
``contextual-region-brl-v1`` branch and ``EXPERIMENTS.md``.  The v2 runtime
has one executable reward contract: Reward N.
"""

from __future__ import annotations

from dataclasses import dataclass


REWARD_VERSION = "N"
TAT_PENALTY_START = 160.0

TAT_SIGNAL_TOTAL_TAT_LEVEL = "total_tat_level"

RAIL_REWARD_FREE_FLOW_NEUTRAL_2 = "free_flow_neutral_2"


@dataclass(frozen=True)
class RewardContract:
    version: str
    tat_signal_mode: str
    tat_signal_description: str
    tat_termination_enabled: bool
    tat_termination_grace_steps: int
    tat_termination_threshold: float
    tat_termination_patience: int
    tat_termination_inclusive: bool
    terminal_tat_penalty: float


REWARD_N_CONTRACT = RewardContract(
    version="N",
    tat_signal_mode=TAT_SIGNAL_TOTAL_TAT_LEVEL,
    tat_signal_description="one_sided_total_tat_level",
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
        raise ValueError("contextual TD7 v2 supports only reward_version='N'")
    return REWARD_VERSION


def reward_contract(version: str = REWARD_VERSION) -> RewardContract:
    canonical_reward_version(version)
    return REWARD_N_CONTRACT


REWARD_N_PROFILE = {
    "global_alpha": 0.5,
    "local_alpha": 0.5,
    "rail_tat_weight": 30.0,
    "rail_free_flow_neutral_ratio": 2.0,
    "smooth_b_rl_weight": 0.25,
    "smooth_exp_residual_weight": 0.5,
    "tat_weight": 11.0,
    "op_weight": 4.0,
    "backlog_weight": 0.0004,
    "backlog_growth_enabled": True,
    "backlog_growth_horizon": 300,
    "backlog_growth_scale": 30.0,
    "backlog_growth_weight": 0.16,
    "idle_reserve_target": 200.0,
    "idle_reserve_scale": 50.0,
    "idle_reserve_weight": 0.20,
    "local_oht_weight": 0.3,
    "local_predicted_oht_weight": 0.075,
    "local_stop_weight": 0.3,
    "local_idle_weight": 0.0,
    "local_capacity_weight": 0.1,
    "use_tat": True,
    "use_op": True,
    "use_backlog": True,
    "tat_reference": 165.0,
    "op_reference": 0.80,
    "local_reward_scale": 2.0,
    "rail_tat_clip": 1.0,
}

__all__ = (
    "RAIL_REWARD_FREE_FLOW_NEUTRAL_2",
    "REWARD_N_CONTRACT",
    "REWARD_N_PROFILE",
    "REWARD_VERSION",
    "REWARD_VERSIONS",
    "RewardContract",
    "TAT_PENALTY_START",
    "TAT_SIGNAL_TOTAL_TAT_LEVEL",
    "canonical_reward_version",
    "reward_contract",
)
