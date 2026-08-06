"""Gaussian contextual actor for SAC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from cocel_rl.algorithms.contextual_td7.config import ContextualNetworkConfig
from cocel_rl.algorithms.contextual_td7.networks import (
    ActorOutput,
    _require_finite,
    _require_tensor,
)


@dataclass(frozen=True)
class ContextualSACActorOutput:
    action: torch.Tensor
    pre_tanh: torch.Tensor
    log_prob: torch.Tensor
    mean_action: torch.Tensor
    mean_pre_tanh: torch.Tensor
    log_std: torch.Tensor


class ContextualGaussianActor(nn.Module):
    def __init__(
        self,
        config: ContextualNetworkConfig | None = None,
        *,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.config = config or ContextualNetworkConfig()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        cfg = self.config
        self.trunk = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.mean_layer = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.log_std_layer = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        for layer in (self.mean_layer, self.log_std_layer):
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(layer.bias, -3e-3, 3e-3)

    def distribution_parameters(self, state: torch.Tensor):
        _require_tensor("state", state, (self.config.context_dim,))
        features = self.trunk(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features).clamp(
            self.log_std_min, self.log_std_max
        )
        _require_finite("sac_actor.mean", mean)
        _require_finite("sac_actor.log_std", log_std)
        return mean, log_std

    def forward(self, state: torch.Tensor) -> ActorOutput:
        mean, _ = self.distribution_parameters(state)
        action = torch.tanh(mean)
        _require_finite("sac_actor.mean_action", action)
        return ActorOutput(action=action, pre_tanh=mean)

    def sample(self, state: torch.Tensor) -> ContextualSACActorOutput:
        mean, log_std = self.distribution_parameters(state)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = (
            distribution.log_prob(pre_tanh)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean)
        for name, value in (
            ("sac_actor.action", action),
            ("sac_actor.log_prob", log_prob),
            ("sac_actor.mean_action", mean_action),
        ):
            _require_finite(name, value)
        return ContextualSACActorOutput(
            action=action,
            pre_tanh=pre_tanh,
            log_prob=log_prob,
            mean_action=mean_action,
            mean_pre_tanh=mean,
            log_std=log_std,
        )
