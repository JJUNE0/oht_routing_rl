"""Authoritative historical reward-version contracts and parameters.

Keep reward-version identity, aliases, formulas, normalizer settings, and
termination policy in this module.  Runtime reward calculation belongs in
``contextual_reward.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


REWARD_VERSION = "U"
TAT_PENALTY_START = 160.0

TAT_SIGNAL_COMPLETION_EVENT = "completion_event"
TAT_SIGNAL_TOTAL_TAT_LEVEL = "total_tat_level"
TAT_SIGNAL_MARGINAL_TAT_EMA = "marginal_tat_ema"
TAT_SIGNAL_MODES = (
    TAT_SIGNAL_COMPLETION_EVENT,
    TAT_SIGNAL_TOTAL_TAT_LEVEL,
    TAT_SIGNAL_MARGINAL_TAT_EMA,
)

RAIL_REWARD_FIXED_TAT_REFERENCE = "fixed_tat_reference"
RAIL_REWARD_BASELINE_RATIO = "baseline_ratio"
RAIL_REWARD_FREE_FLOW_NEUTRAL_2 = "free_flow_neutral_2"
RAIL_REWARD_MODES = (
    RAIL_REWARD_FIXED_TAT_REFERENCE,
    RAIL_REWARD_BASELINE_RATIO,
    RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
)


@dataclass(frozen=True)
class RewardContract:
    version: str
    historical_version: str
    contract_version: str
    tat_version: str
    normalization_version: str
    tat_signal_mode: str
    tat_signal_description: str
    tat_termination_enabled: bool
    tat_termination_grace_steps: int
    tat_termination_threshold: float
    tat_termination_patience: int
    tat_termination_inclusive: bool
    terminal_tat_penalty: float


REWARD_CONTRACTS = {
    "E": RewardContract(
        "E", "E", "contextual_controlled_reward_v7_oht_cycle_segments_boundary_delta",
        "marginal_tat_ema_ref174p4236_v1",
        "running_global_local_before_update_freeze10000_v1",
        TAT_SIGNAL_MARGINAL_TAT_EMA, "marginal_tat_ema",
        True, 101, 500.0, 1, False, 0.0,
    ),
    "F_RAMP": RewardContract(
        "F_RAMP", "F", "contextual_controlled_reward_v8_tat_confidence_diagnostics",
        "marginal_tat_ema_completion_confidence_n500_v1",
        "running_global_local_before_update_freeze10000_v1",
        TAT_SIGNAL_MARGINAL_TAT_EMA, "marginal_tat_ema_confidence_ramp",
        True, 101, 500.0, 1, False, 0.0,
    ),
    "F_NO_RAMP": RewardContract(
        "F_NO_RAMP", "F", "contextual_controlled_reward_v8_tat_confidence_diagnostics",
        "marginal_tat_ema_unramped_v1",
        "running_global_local_before_update_freeze10000_v1",
        TAT_SIGNAL_MARGINAL_TAT_EMA, "marginal_tat_ema_unramped",
        True, 101, 500.0, 1, False, 0.0,
    ),
    "G": RewardContract(
        "G", "G", "contextual_controlled_reward_v9_global_tat_op_backlog_levels",
        "marginal_tat_ema_ref174p4236_v1",
        "running_global_local_before_update_freeze10000_v1",
        TAT_SIGNAL_MARGINAL_TAT_EMA, "marginal_tat_ema",
        True, 101, 500.0, 1, False, 0.0,
    ),
    "H": RewardContract(
        "H", "H", "contextual_controlled_reward_v10_total_tat_level",
        "signed_total_tat_level_ref174p4236_v1",
        "running_global_local_before_update_freeze10000_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "signed_total_tat_level",
        True, 101, 500.0, 1, False, 0.0,
    ),
    "I": RewardContract(
        "I", "I", "contextual_controlled_reward_v11_fixed_scale_rail100",
        "signed_total_tat_level_ref174p4236_clip1_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "signed_total_tat_level",
        True, 101, 500.0, 1, False, 0.0,
    ),
    "J": RewardContract(
        "J", "J", "contextual_controlled_reward_v12_free_flow_neutral2",
        "signed_total_tat_level_ref174p4236_clip1_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "signed_total_tat_level",
        True, 101, 500.0, 1, False, 0.0,
    ),
    "K": RewardContract(
        "K", "K", "contextual_controlled_reward_v13_balanced_freeflow_neutral2",
        "signed_total_tat_level_ref174p4236_clip1_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "signed_total_tat_level",
        True, 101, 500.0, 1, False, 0.0,
    ),
    "L": RewardContract(
        "L", "L", "contextual_controlled_reward_v14_leading_pressure_rebalanced",
        "signed_total_tat_level_ref165_clip1_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "signed_total_tat_level",
        True, 101, 170.0, 1, False, -2.0,
    ),
    "M": RewardContract(
        "M", "M", "contextual_controlled_reward_v15_tat_one_sided_patience",
        "one_sided_total_tat_excess160_clip10_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "one_sided_total_tat_level",
        True, 10_000, 170.0, 300, True, -20.0,
    ),
    "N": RewardContract(
        "N", "N", "contextual_controlled_reward_v16_tat_one_sided_unbounded",
        "one_sided_total_tat_excess160_unbounded_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "one_sided_total_tat_level",
        True, 10_000, 200.0, 300, True, -20.0,
    ),
    "O": RewardContract(
        "O", "O", "contextual_controlled_reward_v17_k_backlog_growth",
        "signed_total_tat_level_ref174p4236_clip1_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "signed_total_tat_level",
        False, 0, 500.0, 1, True, 0.0,
    ),
    "P": RewardContract(
        "P", "P", "contextual_controlled_reward_v18_n_pressure_ablation",
        "one_sided_total_tat_excess160_unbounded_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "one_sided_total_tat_level",
        True, 10_000, 200.0, 300, True, -20.0,
    ),
    "Q": RewardContract(
        "Q", "Q", "contextual_controlled_reward_v19_n_pressure_ablation_no_idle",
        "one_sided_total_tat_excess160_unbounded_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "one_sided_total_tat_level",
        True, 10_000, 200.0, 300, True, -20.0,
    ),
    "R": RewardContract(
        "R", "R", "contextual_controlled_reward_v20_backlog_pressure_rebalance",
        "one_sided_total_tat_excess160_unbounded_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "one_sided_total_tat_level",
        True, 10_000, 200.0, 300, True, -20.0,
    ),
    "S_REBALANCE": RewardContract(
        "S_REBALANCE", "S", "contextual_controlled_reward_v21_component_rebalance",
        "one_sided_total_tat_excess160_unbounded_v1",
        "fixed_scale_no_reward_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "one_sided_total_tat_level",
        True, 10_000, 200.0, 300, True, -20.0,
    ),
    "S_EHYBRID": RewardContract(
        "S_EHYBRID", "S", "contextual_controlled_reward_v21_e_structure_signed_total_tat",
        "signed_total_tat_level_ref165_v1",
        "running_global_local_before_update_freeze30000_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "signed_total_tat_level",
        True, 10_000, 200.0, 300, True, -20.0,
    ),
    "T": RewardContract(
        "T", "T", "contextual_controlled_reward_v22_global_raw_local_running_norm",
        "signed_total_tat_level_ref165_v1",
        "reward_t_global_raw_local_running_normalizer_v1",
        TAT_SIGNAL_TOTAL_TAT_LEVEL, "signed_total_tat_level",
        True, 10_000, 200.0, 300, True, -20.0,
    ),
    "U": RewardContract(
        "U", "U", "contextual_controlled_reward_v23_completion_tat_global_raw_local_running_norm",
        "actual_new_completion_cmd_tat_ref165_v1",
        "reward_u_global_raw_local_running_normalizer_v1",
        TAT_SIGNAL_COMPLETION_EVENT, "actual_new_completion_cmd_tat_event",
        True, 10_000, 200.0, 300, True, -20.0,
    ),
}

REWARD_ALIASES = {
    "F": "F_RAMP",
    "S": "S_EHYBRID",
}
REWARD_VERSIONS = tuple(REWARD_CONTRACTS)
REWARD_VERSION_CHOICES = tuple((*REWARD_VERSIONS, *REWARD_ALIASES))


def canonical_reward_version(version: str) -> str:
    canonical = str(version).strip().upper().replace("-", "_")
    canonical = REWARD_ALIASES.get(canonical, canonical)
    if canonical not in REWARD_CONTRACTS:
        raise ValueError(
            f"reward_version must be one of {REWARD_VERSION_CHOICES}"
        )
    return canonical


def reward_contract(version: str) -> RewardContract:
    return REWARD_CONTRACTS[canonical_reward_version(version)]


def _reward_profile(**overrides):
    profile = {
        "global_alpha": 0.5,
        "local_alpha": 0.5,
        "rail_tat_weight": 1.0,
        "rail_reward_mode": RAIL_REWARD_FIXED_TAT_REFERENCE,
        "rail_free_flow_neutral_ratio": 2.0,
        "rail_baseline_ratio_reference": None,
        "smooth_b_rl_weight": 0.05,
        "smooth_exp_residual_weight": 0.5,
        "tat_weight": 2.3,
        "op_weight": 0.0,
        "backlog_weight": 0.0025,
        "backlog_growth_enabled": False,
        "backlog_growth_horizon": 300,
        "backlog_growth_scale": 30.0,
        "backlog_growth_weight": 0.0,
        "idle_reserve_target": 200.0,
        "idle_reserve_scale": 50.0,
        "idle_reserve_weight": 0.0,
        "local_oht_weight": 0.3,
        "local_predicted_oht_weight": 0.2,
        "local_stop_weight": 0.3,
        "local_idle_weight": 0.1,
        "local_capacity_weight": 0.1,
        "use_tat": True,
        "use_op": False,
        "use_backlog": True,
        "tat_reference": 165.0,
        "tat_one_sided": False,
        "tat_excess_clip": None,
        "tat_ema_beta": 0.05,
        "global_zero_on_first_tick": False,
        "op_reference": 0.80,
        "tat_confidence_n0": 500.0,
        "tat_confidence_ramp": False,
        "tat_confidence_supported": False,
        "freeze_after_env_steps": 30_000,
        "global_normalization_enabled": False,
        "local_normalization_enabled": True,
        "local_fixed_scale_enabled": False,
        "local_reward_scale": 1.0,
        "tat_raw_clip": None,
        "rail_tat_clip": None,
        "normalizer_epsilon": 1e-6,
        "global_clip": 5.0,
        "local_clip": None,
    }
    profile.update(overrides)
    return profile


_E_PROFILE = _reward_profile(
    tat_weight=9.2,
    op_weight=5.0,
    backlog_weight=0.01,
    tat_reference=174.4236,
    freeze_after_env_steps=10_000,
    global_normalization_enabled=True,
    local_normalization_enabled=True,
    global_zero_on_first_tick=True,
    tat_confidence_supported=True,
)
_F_PROFILE = {
    **_E_PROFILE,
    "global_zero_on_first_tick": False,
}
_G_PROFILE = {
    **_F_PROFILE,
    "use_op": True,
    "backlog_weight": 0.002,
}
_I_PROFILE = _reward_profile(
    rail_tat_weight=100.0,
    smooth_b_rl_weight=0.25,
    tat_weight=9.2,
    op_weight=5.0,
    backlog_weight=0.002,
    local_idle_weight=0.0,
    use_op=True,
    tat_reference=174.4236,
    freeze_after_env_steps=10_000,
    global_normalization_enabled=False,
    local_normalization_enabled=False,
    local_fixed_scale_enabled=True,
    local_reward_scale=3.0,
    tat_raw_clip=1.0,
    rail_tat_clip=0.5,
)
_J_PROFILE = {
    **_I_PROFILE,
    "rail_tat_weight": 1.0,
    "rail_reward_mode": RAIL_REWARD_FREE_FLOW_NEUTRAL_2,
    "rail_tat_clip": 1.0,
}
_K_PROFILE = {
    **_J_PROFILE,
    "rail_tat_weight": 50.0,
    "tat_weight": 18.4,
    "backlog_weight": 0.0005,
    "local_predicted_oht_weight": 0.10,
    "local_reward_scale": 2.0,
}
_L_PROFILE = {
    **_K_PROFILE,
    "rail_tat_weight": 30.0,
    "tat_weight": 11.0,
    "op_weight": 4.0,
    "backlog_weight": 0.0004,
    "backlog_growth_enabled": True,
    "backlog_growth_weight": 0.16,
    "idle_reserve_weight": 0.20,
    "local_predicted_oht_weight": 0.075,
    "tat_reference": 165.0,
}
_N_PROFILE = {
    **_L_PROFILE,
    "tat_one_sided": True,
    "tat_excess_clip": None,
    "tat_raw_clip": None,
}
_P_PROFILE = {
    **_N_PROFILE,
    "rail_tat_weight": 40.0,
    "backlog_weight": 0.0008,
    "backlog_growth_weight": 0.24,
    "idle_reserve_weight": 0.20,
    "local_predicted_oht_weight": 0.10,
}

REWARD_PROFILE_CONFIGS = {
    "E": _E_PROFILE,
    "F_RAMP": {**_F_PROFILE, "tat_confidence_ramp": True},
    "F_NO_RAMP": _F_PROFILE,
    "G": _G_PROFILE,
    "H": {**_G_PROFILE},
    "I": _I_PROFILE,
    "J": _J_PROFILE,
    "K": _K_PROFILE,
    "L": _L_PROFILE,
    "M": {
        **_L_PROFILE,
        "tat_one_sided": True,
        "tat_excess_clip": 10.0,
        "tat_raw_clip": None,
    },
    "N": _N_PROFILE,
    "O": {
        **_K_PROFILE,
        "backlog_growth_enabled": True,
        "backlog_growth_weight": 0.16,
        "idle_reserve_weight": 0.0,
    },
    "P": _P_PROFILE,
    "Q": {**_P_PROFILE, "idle_reserve_weight": 0.0},
    "R": {
        **_P_PROFILE,
        "idle_reserve_weight": 0.0,
        "backlog_weight": 0.008,
        "backlog_growth_weight": 0.16,
    },
    "S_REBALANCE": {
        **_P_PROFILE,
        "idle_reserve_weight": 0.0,
        "tat_weight": 5.5,
        "backlog_weight": 0.0048,
        "backlog_growth_weight": 0.10,
        "local_reward_scale": 0.5,
        "rail_tat_weight": 85.0,
    },
    "S_EHYBRID": _reward_profile(
        tat_weight=9.2,
        backlog_weight=0.01,
        global_normalization_enabled=True,
    ),
    "T": _reward_profile(),
    "U": _reward_profile(),
}


_DEFAULT_REWARD_CONTRACT = reward_contract(REWARD_VERSION)
REWARD_CONTRACT_VERSION = _DEFAULT_REWARD_CONTRACT.contract_version
REWARD_TAT_VERSION = _DEFAULT_REWARD_CONTRACT.tat_version
REWARD_NORMALIZATION_VERSION = _DEFAULT_REWARD_CONTRACT.normalization_version

# Deprecated source-level aliases for compatibility-only callers.
RAIL_TAT_FIXED_REFERENCE = RAIL_REWARD_FIXED_TAT_REFERENCE
RAIL_TAT_FREE_FLOW_RATIO = RAIL_REWARD_BASELINE_RATIO
RAIL_TAT_MODES = RAIL_REWARD_MODES


__all__ = (
    "RAIL_REWARD_BASELINE_RATIO",
    "RAIL_REWARD_FIXED_TAT_REFERENCE",
    "RAIL_REWARD_FREE_FLOW_NEUTRAL_2",
    "RAIL_REWARD_MODES",
    "RAIL_TAT_FIXED_REFERENCE",
    "RAIL_TAT_FREE_FLOW_RATIO",
    "RAIL_TAT_MODES",
    "REWARD_ALIASES",
    "REWARD_CONTRACTS",
    "REWARD_CONTRACT_VERSION",
    "REWARD_NORMALIZATION_VERSION",
    "REWARD_PROFILE_CONFIGS",
    "REWARD_TAT_VERSION",
    "REWARD_VERSION",
    "REWARD_VERSION_CHOICES",
    "REWARD_VERSIONS",
    "RewardContract",
    "TAT_PENALTY_START",
    "TAT_SIGNAL_COMPLETION_EVENT",
    "TAT_SIGNAL_MARGINAL_TAT_EMA",
    "TAT_SIGNAL_MODES",
    "TAT_SIGNAL_TOTAL_TAT_LEVEL",
    "canonical_reward_version",
    "reward_contract",
)
