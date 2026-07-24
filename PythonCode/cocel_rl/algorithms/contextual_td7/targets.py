"""Bellman target helpers in applied-action space."""

from __future__ import annotations

import torch


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


def bellman_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    shapes = {tuple(x.shape) for x in (reward, done, q1, q2)}
    if len(shapes) != 1 or reward.ndim != 2 or reward.shape[1] != 1:
        raise ValueError(f"Bellman tensors must share [B,1], got {shapes}")
    result = reward + float(gamma) * (1.0 - done) * torch.minimum(q1, q2)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Bellman target contains NaN or Inf")
    return result
