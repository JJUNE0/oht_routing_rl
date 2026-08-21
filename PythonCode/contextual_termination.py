"""Named, versioned episode-termination policies for contextual training."""

from __future__ import annotations


TAT_TERMINATION_POLICY_VERSION = (
    "tat_termination_episode_gate_v5_named_tat180"
)
WARMUP_EPISODE_TRANSITION_VERSION = (
    "warmup_episode_boundary_v1_terminal_observation"
)
TAT_TERMINATION_REWARD_PROFILE = "reward_profile"
TAT_TERMINATION_EPISODE2_TAT175 = "episode2_tat175"
TAT_TERMINATION_EPISODE2_TAT180 = "episode2_tat180"
TAT_TERMINATION_POLICIES = (
    TAT_TERMINATION_REWARD_PROFILE,
    TAT_TERMINATION_EPISODE2_TAT175,
    TAT_TERMINATION_EPISODE2_TAT180,
)


def tat_termination_profile(policy: str, reward_contract) -> dict:
    """Resolve environment termination without changing the reward formula."""
    if policy == TAT_TERMINATION_EPISODE2_TAT175:
        return {
            "tat_termination_start_episode": 2,
            "tat_termination_enabled": True,
            "tat_termination_grace_steps": 0,
            "early_stop_tat_threshold": 175.0,
            "tat_above_threshold_patience": 1,
            "tat_termination_inclusive": False,
        }
    if policy == TAT_TERMINATION_EPISODE2_TAT180:
        return {
            "tat_termination_start_episode": 2,
            "tat_termination_enabled": True,
            "tat_termination_grace_steps": 0,
            "early_stop_tat_threshold": 180.0,
            "tat_above_threshold_patience": 1,
            "tat_termination_inclusive": False,
        }
    if policy == TAT_TERMINATION_REWARD_PROFILE:
        return {
            "tat_termination_start_episode": 1,
            "tat_termination_enabled": reward_contract.tat_termination_enabled,
            "tat_termination_grace_steps": (
                reward_contract.tat_termination_grace_steps
            ),
            "early_stop_tat_threshold": (
                reward_contract.tat_termination_threshold
            ),
            "tat_above_threshold_patience": (
                reward_contract.tat_termination_patience
            ),
            "tat_termination_inclusive": (
                reward_contract.tat_termination_inclusive
            ),
        }
    raise ValueError(
        f"tat_termination_policy must be one of {TAT_TERMINATION_POLICIES}"
    )


__all__ = (
    "TAT_TERMINATION_EPISODE2_TAT175",
    "TAT_TERMINATION_EPISODE2_TAT180",
    "TAT_TERMINATION_POLICIES",
    "TAT_TERMINATION_POLICY_VERSION",
    "TAT_TERMINATION_REWARD_PROFILE",
    "WARMUP_EPISODE_TRANSITION_VERSION",
    "tat_termination_profile",
)
