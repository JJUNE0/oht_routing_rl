"""SAC networks adapted from the sibling CO-GYM implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from cocel_rl.algorithms.base import BaseActor, BaseCritic
from cocel_rl.utils.utils import weight_init


class SACCritic(BaseCritic):
    def __init__(self, obs_dim, act_dim, hidden_dims, activation_fc_name):
        super().__init__(activation_fc_name)
        activation = nn.ELU if str(activation_fc_name).lower() == "elu" else nn.ReLU
        self.q1 = self._head(obs_dim + act_dim, hidden_dims, activation)
        self.q2 = self._head(obs_dim + act_dim, hidden_dims, activation)
        self.apply(weight_init)

    @staticmethod
    def _head(input_dim, hidden_dims, activation):
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), activation()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def to_tensor(state, action):
        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return state, action

    def forward(self, state, action):
        state, action = self.to_tensor(state, action)
        state_action = torch.cat((state, action), dim=-1)
        return self.q1(state_action), self.q2(state_action)


class SACPolicy(BaseActor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dims,
        action_bound,
        log_std_bound,
        activation_fc_name,
    ):
        super().__init__(activation_fc_name)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dims = list(hidden_dims)
        self.activation_fc_name = activation_fc_name
        self.action_bound = list(action_bound)
        self.log_std_min = float(log_std_bound[0])
        self.log_std_max = float(log_std_bound[1])

        layers = []
        previous = self.obs_dim
        for width in hidden_dims:
            layers.append(nn.Linear(previous, width))
            previous = width
        self.hidden_layers = nn.ModuleList(layers)
        self.mean_layer = nn.Linear(previous, self.act_dim)
        self.log_std_layer = nn.Linear(previous, self.act_dim)
        self.register_buffer(
            "action_rescale",
            torch.as_tensor(
                (action_bound[1] - action_bound[0]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_rescale_bias",
            torch.as_tensor(
                (action_bound[1] + action_bound[0]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.apply(weight_init)

    @staticmethod
    def to_tensor(state):
        value = torch.as_tensor(state, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        return value

    def forward(self, state):
        value = self.to_tensor(state)
        for layer in self.hidden_layers:
            value = self.activation_fc(layer(value))
        mean = self.mean_layer(value)
        log_std = self.log_std_layer(value).clamp(
            self.log_std_min, self.log_std_max
        )
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        bounded = torch.tanh(pre_tanh)
        action = bounded * self.action_rescale + self.action_rescale_bias
        correction = torch.log(
            self.action_rescale.abs() * (1.0 - bounded.square()) + 1e-6
        )
        log_prob = (distribution.log_prob(pre_tanh) - correction).sum(
            dim=-1, keepdim=True
        )
        deterministic = (
            torch.tanh(mean) * self.action_rescale + self.action_rescale_bias
        )
        return action, log_prob, deterministic

    def get_action(self, state, eval):
        device = next(self.parameters()).device
        value = self.to_tensor(state).to(device)
        with torch.no_grad():
            sampled, _, deterministic = self.sample(value)
            action = deterministic if eval else sampled
        result = action.cpu().numpy()
        original = np.asarray(state)
        return (result[0] if original.ndim == 1 else result), None


class SACONNXPolicy(BaseActor):
    """Deterministic mean policy used by the existing ONNX exporter."""

    def __init__(
        self, obs_dim, act_dim, hidden_dims, action_bound, activation_fc_name
    ):
        super().__init__(activation_fc_name)
        self.hidden_layers = nn.ModuleList()
        previous = obs_dim
        for width in hidden_dims:
            self.hidden_layers.append(nn.Linear(previous, width))
            previous = width
        self.mean_layer = nn.Linear(previous, act_dim)
        self.register_buffer(
            "action_rescale",
            torch.as_tensor(
                (action_bound[1] - action_bound[0]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_rescale_bias",
            torch.as_tensor(
                (action_bound[1] + action_bound[0]) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, state):
        value = state
        for layer in self.hidden_layers:
            value = self.activation_fc(layer(value))
        return torch.tanh(self.mean_layer(value)) * self.action_rescale + self.action_rescale_bias

    def get_action(self, state, eval):
        raise NotImplementedError
