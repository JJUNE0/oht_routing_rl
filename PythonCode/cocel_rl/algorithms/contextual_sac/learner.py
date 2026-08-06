"""Contextual Soft Actor-Critic with automatic entropy tuning."""

from __future__ import annotations

import copy
import math
import time

import torch
import torch.nn.functional as F

from cocel_rl.algorithms.contextual_td7.config import ContextualNetworkConfig
from cocel_rl.algorithms.contextual_td7.learner_types import ContextualLearnerUpdate
from cocel_rl.algorithms.contextual_td7.networks import (
    ContextualTwinCritic,
    DirectionalContextEncoder,
)
from cocel_rl.algorithms.contextual_td7.targets import scale_policy_action

from .config import ContextualSACLearnerConfig, LEARNER_VERSION
from .networks import ContextualGaussianActor


def _observation_args(batch, *, next_state=False):
    prefix = "next_" if next_state else ""
    return tuple(
        getattr(batch, f"{prefix}{name}")
        for name in (
            "center_local",
            "incoming_local",
            "outgoing_local",
            "incoming_relation",
            "outgoing_relation",
            "global_state",
        )
    )


def _clip_grad_norm(parameters, max_norm):
    value = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm, error_if_nonfinite=True
    )
    return float(value.detach().cpu())


def _parameter_distance(source, target):
    values = [
        (left.detach().double() - right.detach().double()).square().sum()
        for left, right in zip(source.parameters(), target.parameters())
    ]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def _soft_update(source, target, tau):
    with torch.no_grad():
        for online, fixed in zip(source.parameters(), target.parameters()):
            fixed.mul_(1.0 - tau).add_(online, alpha=tau)


def _twin_parameter_diagnostics(critic):
    q1 = dict(critic.q1.named_parameters())
    q2 = dict(critic.q2.named_parameters())
    if q1.keys() != q2.keys():
        raise RuntimeError("Q1/Q2 parameter schemas differ")
    difference = torch.cat([
        (q1[name].detach().double() - q2[name].detach().double()).reshape(-1)
        for name in q1
    ])
    return {
        "critic/parameter_l2_distance": float(
            torch.linalg.vector_norm(difference).cpu()
        ),
        "critic/parameter_max_abs_diff": float(
            difference.abs().max().cpu()
        ),
    }


class ContextualSACLearner:
    def __init__(
        self,
        replay,
        *,
        network_config: ContextualNetworkConfig | None = None,
        config: ContextualSACLearnerConfig | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ):
        self.replay = replay
        self.network_config = network_config or ContextualNetworkConfig()
        self.config = config or ContextualSACLearnerConfig()
        self.device = torch.device(device)
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        self.encoder = DirectionalContextEncoder(self.network_config).to(self.device)
        self.actor = ContextualGaussianActor(
            self.network_config,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        ).to(self.device)
        self.critic = ContextualTwinCritic(self.network_config).to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for module in (self.target_encoder, self.target_critic):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=self.config.encoder_lr,
            eps=self.config.adam_eps,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr,
            eps=self.config.adam_eps,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr,
            eps=self.config.adam_eps,
        )
        self.log_alpha = torch.tensor(
            [self.config.initial_alpha], device=self.device
        ).log().requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.temperature_lr,
            eps=self.config.adam_eps,
        )
        self.target_entropy = (
            -float(self.network_config.action_dim)
            if self.config.target_entropy is None
            else float(self.config.target_entropy)
        )
        self.applied_action_scale = float(self.config.action_scale)
        self.learner_update_count = 0
        self.actor_update_count = 0
        self.target_update_count = 0
        self.last_actor_loss = 0.0
        self.last_actor_grad_norm = 0.0
        self.last_actor_update_step = 0
        self.last_diagnostics = {}

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def twin_parameter_diagnostics(self):
        return _twin_parameter_diagnostics(self.critic)

    def gate_status(self, *, action_enabled_env_steps):
        builder = self.replay.observation_builder
        normalizer_frozen = bool(
            getattr(builder.local_normalizer, "frozen", False)
            and getattr(builder.global_normalizer, "frozen", False)
        )
        return {
            "minimum_replay_env_steps": (
                self.replay.size_env_steps >= self.config.minimum_replay_env_steps
            ),
            "minimum_action_enabled_env_steps": (
                int(action_enabled_env_steps)
                >= self.config.minimum_action_enabled_env_steps
            ),
            "normalizer_frozen": (
                normalizer_frozen or not self.config.require_normalizer_frozen
            ),
        }

    def can_learn(self, *, action_enabled_env_steps):
        return all(self.gate_status(
            action_enabled_env_steps=action_enabled_env_steps
        ).values())

    def set_applied_action_scale(self, scale):
        value = float(scale)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("applied action scale must be in (0, 1]")
        self.applied_action_scale = value

    def update(self, batch=None):
        started = time.perf_counter()
        if batch is None:
            batch = self.replay.sample(self.config.batch_size, device=self.device)
        self.learner_update_count += 1
        step = self.learner_update_count

        with torch.no_grad():
            next_state = self.target_encoder(
                *_observation_args(batch, next_state=True),
                return_attention=False,
            ).state
            next_sample = self.actor.sample(next_state)
            next_applied = scale_policy_action(
                next_sample.action, self.applied_action_scale
            )
            target_pair = self.target_critic(next_state, next_applied)
            soft_value = (
                torch.minimum(target_pair.q1, target_pair.q2)
                - self.alpha * next_sample.log_prob
            )
            target_q = batch.reward + (
                1.0 - batch.done
            ) * self.config.gamma * soft_value

        online_state = self.encoder(
            *_observation_args(batch), return_attention=False
        ).state
        q_pair = self.critic(online_state, batch.applied_action)
        q1_loss = F.mse_loss(q_pair.q1, target_q)
        q2_loss = F.mse_loss(q_pair.q2, target_q)
        critic_loss = q1_loss + q2_loss
        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        encoder_grad = _clip_grad_norm(
            self.encoder.parameters(), self.config.encoder_grad_clip
        )
        critic_grad = _clip_grad_norm(
            self.critic.parameters(), self.config.critic_grad_clip
        )
        self.encoder_optimizer.step()
        self.critic_optimizer.step()

        with torch.no_grad():
            actor_state = self.encoder(
                *_observation_args(batch), return_attention=False
            ).state
        actor_sample = self.actor.sample(actor_state)
        actor_applied = scale_policy_action(
            actor_sample.action, self.applied_action_scale
        )
        critic_flags = [parameter.requires_grad for parameter in self.critic.parameters()]
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        actor_q = self.critic(actor_state, actor_applied)
        actor_loss = (
            self.alpha.detach() * actor_sample.log_prob
            - torch.minimum(actor_q.q1, actor_q.q2)
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = _clip_grad_norm(
            self.actor.parameters(), self.config.actor_grad_clip
        )
        self.actor_optimizer.step()
        for parameter, flag in zip(self.critic.parameters(), critic_flags):
            parameter.requires_grad_(flag)

        alpha_loss = -(
            self.alpha * (
                actor_sample.log_prob + self.target_entropy
            ).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        _soft_update(self.encoder, self.target_encoder, self.config.tau)
        _soft_update(self.critic, self.target_critic, self.config.tau)
        self.actor_update_count += 1
        self.target_update_count += 1
        self.last_actor_loss = float(actor_loss.detach().cpu())
        self.last_actor_grad_norm = actor_grad
        self.last_actor_update_step = step

        td1 = (q_pair.q1.detach() - target_q).abs()
        td2 = (q_pair.q2.detach() - target_q).abs()
        finite = (
            critic_loss, actor_loss, alpha_loss, q_pair.q1, q_pair.q2,
            target_q, actor_sample.log_prob, self.alpha,
        )
        if not all(bool(torch.isfinite(value).all()) for value in finite):
            raise FloatingPointError("contextual SAC update produced NaN or Inf")
        diagnostics = {
            "learner/critic_loss": float(critic_loss.detach().cpu()),
            "learner/actor_loss_last": self.last_actor_loss,
            "learner/actor_grad_norm_last": actor_grad,
            "learner/actor_last_update_step": float(step),
            "learner/actor_updates_total": float(self.actor_update_count),
            "learner/actor_updated_this_step": 1.0,
            "critic/q1_mean": float(q_pair.q1.detach().mean().cpu()),
            "critic/q2_mean": float(q_pair.q2.detach().mean().cpu()),
            "critic/q1_loss": float(q1_loss.detach().cpu()),
            "critic/q2_loss": float(q2_loss.detach().cpu()),
            "critic/target_q_mean": float(target_q.mean().detach().cpu()),
            "critic/target_q_std": float(target_q.std().detach().cpu()),
            "critic/td_error_mean": float(
                torch.maximum(td1, td2).mean().cpu()
            ),
            "grad/encoder_norm": encoder_grad,
            "grad/critic_norm": critic_grad,
            "target/encoder_distance": _parameter_distance(
                self.encoder, self.target_encoder
            ),
            "target/critic_distance": _parameter_distance(
                self.critic, self.target_critic
            ),
            **self.twin_parameter_diagnostics(),
            "update/learner_count": float(step),
            "update/actor_count": float(self.actor_update_count),
            "update/target_count": float(self.target_update_count),
            "action/policy_mean": float(
                actor_sample.mean_action.detach().mean().cpu()
            ),
            "action/policy_std": float(
                actor_sample.mean_action.detach().std().cpu()
            ),
            "action/applied_mean": float(actor_applied.detach().mean().cpu()),
            "action/applied_std": float(actor_applied.detach().std().cpu()),
            "learner/replay_policy_action_std": float(
                batch.policy_action.detach().std().cpu()
            ),
            "learner/replay_applied_action_std": float(
                batch.applied_action.detach().std().cpu()
            ),
            "learner/applied_action_scale": self.applied_action_scale,
            "learner/critic_action_matches_applied": 1.0,
            "sac/alpha": float(self.alpha.detach().cpu()),
            "sac/alpha_loss": float(alpha_loss.detach().cpu()),
            "sac/entropy": float(
                (-actor_sample.log_prob).mean().detach().cpu()
            ),
            "sac/log_prob_mean": float(
                actor_sample.log_prob.mean().detach().cpu()
            ),
            "sac/log_std_mean": float(
                actor_sample.log_std.mean().detach().cpu()
            ),
            "sac/target_entropy": self.target_entropy,
            "sale/enabled": 0.0,
            "lap/enabled": 0.0,
            "numeric/learner_finite_ratio": 1.0,
            "timing/total_learner_update_ms": (
                time.perf_counter() - started
            ) * 1000.0,
        }
        if not all(math.isfinite(float(value)) for value in diagnostics.values()):
            raise FloatingPointError("contextual SAC diagnostics contain NaN or Inf")
        self.last_diagnostics = diagnostics
        return ContextualLearnerUpdate(
            diagnostics=diagnostics,
            actor_updated=True,
            target_updated=True,
        )


__all__ = ["ContextualSACLearner", "LEARNER_VERSION"]
