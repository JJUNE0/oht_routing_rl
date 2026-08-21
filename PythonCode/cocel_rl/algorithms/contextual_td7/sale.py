"""Decoupled SALE representation modules for structured contextual state."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .config import ContextualNetworkConfig
from .networks import DirectionalContextEncoder
from .stacking import encode_observation_stack, flatten_state_stack


SALE_VERSION = "contextual_sale_stacked_avg_l1_v2"


def avg_l1_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scale = value.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    result = value / scale
    if not torch.isfinite(result).all():
        raise FloatingPointError("AvgL1Norm produced NaN or Inf")
    return result


class SALEStateEncoder(nn.Module):
    def __init__(self, network_config=None, embedding_dim=256):
        super().__init__()
        self.context = DirectionalContextEncoder(
            network_config or ContextualNetworkConfig()
        )
        self.projection = nn.Linear(
            self.context.config.stacked_context_dim, embedding_dim
        )

    def forward(self, *observation):
        encoded = encode_observation_stack(
            self.context, observation, return_attention=False
        )
        context = flatten_state_stack(
            encoded.state, self.context.config.num_stacks
        )
        return avg_l1_norm(self.projection(context))


class SALEStateActionEncoder(nn.Module):
    def __init__(self, embedding_dim=256, action_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, state_embedding, applied_action):
        result = self.network(torch.cat((state_embedding, applied_action), -1))
        if not torch.isfinite(result).all():
            raise FloatingPointError("SALE state-action output is non-finite")
        return result


class SALEOnline(nn.Module):
    def __init__(self, network_config=None, embedding_dim=256):
        super().__init__()
        config = network_config or ContextualNetworkConfig()
        self.state_encoder = SALEStateEncoder(config, embedding_dim)
        self.state_action_encoder = SALEStateActionEncoder(
            embedding_dim, config.stacked_action_dim
        )

    def state(self, observation):
        return self.state_encoder(*observation)

    def state_action(self, state_embedding, applied_action):
        return self.state_action_encoder(state_embedding, applied_action)


def frozen_sale_copy(module):
    result = copy.deepcopy(module)
    result.eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result
