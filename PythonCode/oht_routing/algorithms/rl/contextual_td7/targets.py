"""Bellman target helpers in applied-action space.

Two critic aggregation rules live here. `cdq` is the clipped double-Q minimum
of TD3/TD7; `uboc` is the uncertainty-based overestimation correction of UD7,
which replaces the minimum with the ensemble mean less an uncertainty penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


CRITIC_TARGET_CDQ = "cdq"
CRITIC_TARGET_UBOC = "uboc"
CRITIC_TARGET_MODES = (CRITIC_TARGET_CDQ, CRITIC_TARGET_UBOC)

# For two i.i.d. normal samples E[min(X, Y)] = mu - sigma / sqrt(pi), so this
# beta is what makes the UBOC target keep the clipped double-Q expectation
# while cutting its variance (UD7 Theorems 4.1 and 4.2). The UD7 reference
# implementation hard-codes the same constant as 0.5641896.
UBOC_BETA = 1.0 / math.sqrt(math.pi)


def scale_policy_action(policy_action: torch.Tensor, action_scale: float):
    if not torch.isfinite(policy_action).all():
        raise FloatingPointError("policy action contains NaN or Inf")
    return policy_action * float(action_scale)


def target_applied_action(
    policy_action: torch.Tensor,
    *,
    action_scale: float,
    noise_std: float,
    noise_clip: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if not torch.isfinite(policy_action).all():
        raise FloatingPointError("policy action contains NaN or Inf")
    if noise is None:
        noise = torch.randn_like(policy_action) * float(noise_std)
    else:
        noise = noise.to(
            device=policy_action.device, dtype=policy_action.dtype
        )
    noise = noise.clamp(-float(noise_clip), float(noise_clip))
    exploratory = (policy_action + noise).clamp(-1.0, 1.0)
    result = scale_policy_action(exploratory, action_scale)
    if not torch.isfinite(result).all():
        raise FloatingPointError("target applied action contains NaN or Inf")
    return result


def validate_critic_target_mode(mode: str) -> str:
    if mode not in CRITIC_TARGET_MODES:
        raise ValueError(
            f"critic_target_mode must be one of {CRITIC_TARGET_MODES}, "
            f"got {mode!r}"
        )
    return mode


def _require_critic_shape(q_values: torch.Tensor) -> torch.Tensor:
    """Check the shape contract only.

    Finiteness is deliberately not tested here. The critic module already
    validates its own output, and every aggregation below checks its result,
    so a third scan would only add a device synchronization to the hot path.
    """
    if not torch.is_tensor(q_values):
        raise TypeError("critic values must be a torch.Tensor")
    if q_values.ndim != 2 or q_values.shape[1] < 2:
        raise ValueError(
            "critic values must be [B, N] with N >= 2, got "
            f"{tuple(q_values.shape)}"
        )
    return q_values


def _require_beta(beta: float) -> float:
    if not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("uboc beta must be finite and non-negative")
    return float(beta)


def cdq_target_value(q_values: torch.Tensor) -> torch.Tensor:
    """Clipped double-Q aggregation: the minimum over exactly two critics."""
    _require_critic_shape(q_values)
    if q_values.shape[1] != 2:
        raise ValueError(
            "clipped double-Q requires exactly two critics, got "
            f"{int(q_values.shape[1])}"
        )
    return q_values.min(dim=1, keepdim=True).values


def uboc_ensemble_statistics(
    q_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-sample ensemble mean and unbiased standard deviation."""
    _require_critic_shape(q_values)
    mean = q_values.mean(dim=1, keepdim=True)
    variance = q_values.var(dim=1, unbiased=True, keepdim=True)
    # Float error can push a near-identical ensemble marginally below zero.
    deviation = variance.clamp_min(0.0).sqrt()
    return mean, deviation


def uboc_target_value(
    q_values: torch.Tensor, *, beta: float = UBOC_BETA
) -> torch.Tensor:
    """UBOC aggregation: ensemble mean less beta times its spread.

    The penalty scales with how much the critics still disagree, so early
    training is corrected hard and the correction fades as the ensemble
    converges.
    """
    _require_beta(beta)
    mean, deviation = uboc_ensemble_statistics(q_values)
    result = mean - float(beta) * deviation
    if not torch.isfinite(result).all():
        raise FloatingPointError("UBOC target value contains NaN or Inf")
    return result


@dataclass(frozen=True)
class CriticTargetAggregate:
    """One aggregation pass over a [B, N] critic ensemble.

    `value` is the next-state estimate the Bellman target bootstraps. The
    remaining fields describe the ensemble it came from, so the caller can
    report the correction without a second pass over the critics.
    """

    value: torch.Tensor
    ensemble_mean: torch.Tensor
    ensemble_deviation: torch.Tensor
    minimum: torch.Tensor

    @property
    def gap_above_minimum(self) -> torch.Tensor:
        """How much less pessimistic the aggregate is than the minimum."""
        return self.value - self.minimum


def aggregate_critic_target(
    q_values: torch.Tensor, *, mode: str, beta: float = UBOC_BETA
) -> CriticTargetAggregate:
    """Collapse per-critic next-state values into one [B, 1] estimate."""
    validate_critic_target_mode(mode)
    _require_beta(beta)
    if mode == CRITIC_TARGET_CDQ and q_values.shape[1] != 2:
        raise ValueError(
            "clipped double-Q requires exactly two critics, got "
            f"{int(q_values.shape[1])}"
        )
    mean, deviation = uboc_ensemble_statistics(q_values)
    minimum = q_values.min(dim=1, keepdim=True).values
    value = (
        minimum
        if mode == CRITIC_TARGET_CDQ
        else mean - float(beta) * deviation
    )
    if not torch.isfinite(value).all():
        raise FloatingPointError("critic target value contains NaN or Inf")
    return CriticTargetAggregate(
        value=value,
        ensemble_mean=mean,
        ensemble_deviation=deviation,
        minimum=minimum,
    )


def critic_target_value(
    q_values: torch.Tensor, *, mode: str, beta: float = UBOC_BETA
) -> torch.Tensor:
    """Aggregate per-critic next-state values into one [B, 1] estimate."""
    return aggregate_critic_target(q_values, mode=mode, beta=beta).value


def bellman_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    next_value: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Bootstrap an already-aggregated and already-clipped next-state value.

    Value clipping belongs to the aggregate rather than to each critic: the
    minimum commutes with a shared clamp, but the UBOC mean and spread do not.
    """
    shapes = {tuple(x.shape) for x in (reward, done, next_value)}
    if len(shapes) != 1 or reward.ndim != 2 or reward.shape[1] != 1:
        raise ValueError(f"Bellman tensors must share [B,1], got {shapes}")
    result = reward + float(gamma) * (1.0 - done) * next_value
    if not torch.isfinite(result).all():
        raise FloatingPointError("Bellman target contains NaN or Inf")
    return result
