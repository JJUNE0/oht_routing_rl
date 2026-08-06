"""Soft Actor-Critic using the local off-policy learner contract."""

from __future__ import annotations

import torch

from cocel_rl.algorithms.base import BaseAlgorithm
from cocel_rl.buffers.off_policy_buffer import OffPolicyBuffer
from cocel_rl.utils.utils import soft_update

from .network import SACCritic, SACPolicy


ALGORITHM_VERSION = "sac_cogym_auto_entropy_v1"


class SAC(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = "SAC"
        self.type = "off_policy"
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.action_bound = env.action_bound
        algorithm_config = config["algorithm"]

        self.actor = SACPolicy(
            self.obs_dim,
            self.act_dim,
            config["actor_hidden_dims"],
            env.action_bound,
            algorithm_config.get("log_std_bound", [-20.0, 2.0]),
            config["activation_fc"],
        )
        self.critic = SACCritic(
            self.obs_dim,
            self.act_dim,
            config["critic_hidden_dims"],
            config["activation_fc"],
        )
        self.buffer = OffPolicyBuffer(
            self.obs_dim,
            self.act_dim,
            capacity=int(config["buffer_capacity"]),
        )
        device = torch.device(config["device"])
        target_entropy = algorithm_config.get("target_entropy")
        self.target_entropy = (
            -float(self.act_dim)
            if target_entropy is None
            else float(target_entropy)
        )
        initial_alpha = float(algorithm_config.get("initial_alpha", 0.2))
        if initial_alpha <= 0.0:
            raise ValueError("SAC initial_alpha must be positive")
        self.log_alpha = torch.tensor(
            [initial_alpha], device=device, dtype=torch.float32
        ).log().requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=float(algorithm_config.get("temperature_lr", 3e-4)),
        )
        self.tau = float(config["tau"])
        self.applied_action_scale = 1.0
        self.training_steps = 0
        super().__init__()

    def train(
        self,
        buffer,
        critic_optimizer,
        critic,
        target_critic,
        policy_optimizer,
        policy,
        target_policy,
        iteration,
        encoder_optimizer,
    ):
        del target_policy, encoder_optimizer
        stats = None
        for _ in range(iteration):
            self.training_steps += 1
            states, actions, rewards, next_states, dones = buffer.sample(
                self.config["batch_size"], device=self.config["device"]
            )
            with torch.no_grad():
                next_actions, next_log_prob, _ = policy.sample(next_states)
                next_actions = next_actions * self.applied_action_scale
                target_q1, target_q2 = target_critic(next_states, next_actions)
                target_q = (
                    torch.minimum(target_q1, target_q2)
                    - self.log_alpha.exp() * next_log_prob
                )
                bellman = rewards + (
                    1.0 - dones
                ) * self.config["gamma"] * target_q

            q1, q2 = critic(states, actions)
            critic_loss = (q1 - bellman).square().mean() + (
                q2 - bellman
            ).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()

            sampled_actions, log_prob, _ = policy.sample(states)
            sampled_actions = sampled_actions * self.applied_action_scale
            policy_q1, policy_q2 = critic(states, sampled_actions)
            actor_loss = (
                self.log_alpha.exp().detach() * log_prob
                - torch.minimum(policy_q1, policy_q2)
            ).mean()
            policy_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            policy_optimizer.step()

            alpha_loss = -(
                self.log_alpha.exp()
                * (log_prob + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            soft_update(critic, target_critic, self.tau)

            tensors = (critic_loss, actor_loss, alpha_loss, q1, q2, log_prob)
            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                raise FloatingPointError("SAC update produced NaN or Inf")
            stats = {
                "critic": float(critic_loss.detach().cpu()),
                "actor": float(actor_loss.detach().cpu()),
                "q1": float(q1.mean().detach().cpu()),
                "q2": float(q2.mean().detach().cpu()),
                "alpha": float(self.log_alpha.exp().detach().cpu()),
                "alpha_loss": float(alpha_loss.detach().cpu()),
                "entropy": float((-log_prob).mean().detach().cpu()),
                "log_prob": float(log_prob.mean().detach().cpu()),
            }
        return stats or {}
