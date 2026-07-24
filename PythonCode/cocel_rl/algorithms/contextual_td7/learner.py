"""Offline contextual TD7-style learner using applied-action critic inputs."""

from __future__ import annotations

import copy
import math
import time

import torch
import torch.nn.functional as F

from .config import ContextualLearnerConfig, ContextualNetworkConfig
from .learner_types import ContextualLearnerUpdate
from .networks import (
    ContextualActor,
    ContextualTwinCritic,
    DirectionalContextEncoder,
)
from .targets import bellman_target, scale_policy_action, target_applied_action
from .sale import SALEOnline, frozen_sale_copy


LEARNER_VERSION = "contextual_td7_independent_twin_critic_v4"
LEARNER_PERFORMANCE_VERSION = "contextual_td7_sparse_diagnostics_v1"
DIAGNOSTICS_INTERVAL = 10


def _observation_args(batch, *, next_state=False):
    prefix = "next_" if next_state else ""
    return (
        getattr(batch, f"{prefix}center_local"),
        getattr(batch, f"{prefix}incoming_local"),
        getattr(batch, f"{prefix}outgoing_local"),
        getattr(batch, f"{prefix}incoming_relation"),
        getattr(batch, f"{prefix}outgoing_relation"),
        getattr(batch, f"{prefix}global_state"),
    )


def _clip_grad_norm(parameters, max_norm: float) -> float:
    total = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm, error_if_nonfinite=True
    )
    return float(total.detach().cpu())


def _parameter_distance(source, target) -> float:
    squares = [
        torch.sum((left.detach().double() - right.detach().double()).square())
        for left, right in zip(source.parameters(), target.parameters())
    ]
    return float(torch.sqrt(torch.stack(squares).sum()).cpu()) if squares else 0.0


def _gradient_norm(parameters) -> float:
    squares = [
        torch.sum(parameter.grad.detach().double().square())
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.sqrt(torch.stack(squares).sum()).cpu()) if squares else 0.0


def _parameter_ids(parameters) -> set[int]:
    return {id(parameter) for parameter in parameters}


def _optimizer_parameter_ids(optimizer) -> tuple[list[int], set[int]]:
    ordered = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    return ordered, set(ordered)


def _twin_parameter_diagnostics(critic) -> dict[str, float]:
    q1 = dict(critic.q1.named_parameters())
    q2 = dict(critic.q2.named_parameters())
    if q1.keys() != q2.keys():
        raise RuntimeError("Q1/Q2 exclusive parameter schemas differ")
    differences = [
        (q1[name].detach().double() - q2[name].detach().double()).reshape(-1)
        for name in q1
    ]
    if not differences:
        raise RuntimeError("Q1/Q2 have no exclusive trainable parameters")
    flattened = torch.cat(differences)
    return {
        # Q-specific trainable heads only; the shared SALE task projection and
        # the external contextual encoder are intentionally excluded.
        "critic/parameter_l2_distance": float(
            torch.linalg.vector_norm(flattened).cpu()
        ),
        "critic/parameter_max_abs_diff": float(
            flattened.abs().max().cpu()
        ),
    }


class ContextualTD7Learner:
    def __init__(
        self,
        replay,
        *,
        network_config: ContextualNetworkConfig | None = None,
        config: ContextualLearnerConfig | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ):
        self.replay = replay
        self.network_config = network_config or ContextualNetworkConfig()
        self.config = config or ContextualLearnerConfig()
        self.device = torch.device(device)
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        self.encoder = DirectionalContextEncoder(self.network_config).to(self.device)
        sale_kwargs = (
            {
                "sale_embedding_dim": self.config.sale_embedding_dim,
                "sale_feature_dim": self.config.sale_feature_dim,
            }
            if self.config.sale_enabled else {}
        )
        self.actor = ContextualActor(
            self.network_config, **sale_kwargs
        ).to(self.device)
        self.critic = ContextualTwinCritic(
            self.network_config, **sale_kwargs
        ).to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.sale_online = (
            SALEOnline(
                self.network_config, self.config.sale_embedding_dim
            ).to(self.device)
            if self.config.sale_enabled else None
        )
        self.sale_fixed = (
            frozen_sale_copy(self.sale_online).to(self.device)
            if self.config.sale_enabled else None
        )
        self.sale_target_fixed = (
            frozen_sale_copy(self.sale_online).to(self.device)
            if self.config.sale_enabled else None
        )
        for module in (self.target_encoder, self.target_actor, self.target_critic):
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
        self._validate_optimizer_parameter_ownership()
        self.sale_optimizer = (
            torch.optim.Adam(
                self.sale_online.parameters(), lr=self.config.sale_lr,
                eps=self.config.adam_eps,
            )
            if self.config.sale_enabled else None
        )
        self.learner_update_count = 0
        self.actor_update_count = 0
        self.target_update_count = 0
        self.last_actor_loss = 0.0
        self.last_actor_grad_norm = 0.0
        self.last_actor_update_step = 0
        self.last_diagnostics: dict[str, float] = {}
        self._distance_diagnostics = {
            "target/encoder_distance": 0.0,
            "target/actor_distance": 0.0,
            "target/critic_distance": 0.0,
            "sale/fixed_online_distance": 0.0,
            "sale/target_fixed_distance": 0.0,
            **self.twin_parameter_diagnostics(),
        }
        self.sale_update_count = 0
        self.sale_fixed_generation = 0
        self.current_target_q_min = float("inf")
        self.current_target_q_max = float("-inf")
        self.fixed_target_q_min = float("inf")
        self.fixed_target_q_max = float("-inf")
        self.applied_action_scale = float(self.config.action_scale)

    def optimizer_parameter_ownership(self) -> dict[str, set[int]]:
        q1_ids = _parameter_ids(self.critic.q1.parameters())
        q2_ids = _parameter_ids(self.critic.q2.parameters())
        critic_ids = _parameter_ids(self.critic.parameters())
        _, critic_optimizer_ids = _optimizer_parameter_ids(
            self.critic_optimizer
        )
        return {
            "q1": q1_ids,
            "q2": q2_ids,
            "shared_encoder": _parameter_ids(self.encoder.parameters()),
            "critic_shared": critic_ids - q1_ids - q2_ids,
            "critic_optimizer": critic_optimizer_ids,
            "target_critic": _parameter_ids(
                self.target_critic.parameters()
            ),
        }

    def _validate_optimizer_parameter_ownership(self) -> None:
        ownership = self.optimizer_parameter_ownership()
        ordered, optimizer_ids = _optimizer_parameter_ids(
            self.critic_optimizer
        )
        expected = (
            ownership["q1"]
            | ownership["q2"]
            | ownership["critic_shared"]
        )
        if ownership["q1"] & ownership["q2"]:
            raise RuntimeError("Q1 and Q2 share exclusive parameter objects")
        if len(ordered) != len(optimizer_ids):
            raise RuntimeError("critic optimizer contains duplicate parameters")
        if optimizer_ids != expected:
            raise RuntimeError(
                "critic optimizer does not own every online critic parameter "
                "exactly once"
            )
        if optimizer_ids & ownership["target_critic"]:
            raise RuntimeError(
                "target critic parameters must not belong to the optimizer"
            )
        if optimizer_ids & ownership["shared_encoder"]:
            raise RuntimeError(
                "contextual encoder is owned by its separate optimizer"
            )

    def twin_parameter_diagnostics(self) -> dict[str, float]:
        return _twin_parameter_diagnostics(self.critic)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def gate_status(self, *, action_enabled_env_steps: int) -> dict[str, bool]:
        builder = self.replay.observation_builder
        normalizer_frozen = bool(
            getattr(builder.local_normalizer, "frozen", False)
            and getattr(builder.global_normalizer, "frozen", False)
        )
        return {
            "minimum_replay_env_steps": (
                self.replay.size_env_steps
                >= self.config.minimum_replay_env_steps
            ),
            "minimum_action_enabled_env_steps": (
                int(action_enabled_env_steps)
                >= self.config.minimum_action_enabled_env_steps
            ),
            "normalizer_frozen": (
                normalizer_frozen or not self.config.require_normalizer_frozen
            ),
        }

    def can_learn(self, *, action_enabled_env_steps: int) -> bool:
        return all(self.gate_status(
            action_enabled_env_steps=action_enabled_env_steps
        ).values())

    def set_applied_action_scale(self, scale: float) -> None:
        value = float(scale)
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError("applied action scale must be in (0, 1]")
        self.applied_action_scale = value

    def _hard_update_targets(self):
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.config.sale_enabled:
            previous_fixed = copy.deepcopy(self.sale_fixed.state_dict())
            self.sale_target_fixed.load_state_dict(previous_fixed)
            self.sale_fixed.load_state_dict(self.sale_online.state_dict())
            self.sale_fixed_generation += 1
        if (
            math.isfinite(self.current_target_q_min)
            and math.isfinite(self.current_target_q_max)
        ):
            self.fixed_target_q_min = self.current_target_q_min
            self.fixed_target_q_max = self.current_target_q_max
        self.current_target_q_min = float("inf")
        self.current_target_q_max = float("-inf")
        self.target_update_count += 1

    def update(self, batch=None) -> ContextualLearnerUpdate:
        total_start = time.perf_counter()
        if batch is None:
            batch = self.replay.sample(
                self.config.batch_size, device=self.device
            )
        self.learner_update_count += 1
        step = self.learner_update_count

        encoder_start = time.perf_counter()
        online_encoding = self.encoder(
            *_observation_args(batch), return_attention=False
        )
        self._sync()
        encoder_forward_ms = (time.perf_counter() - encoder_start) * 1000

        sale_loss_value = 0.0
        sale_grad = 0.0
        sale_forward_ms = 0.0
        sale_backward_ms = 0.0
        sale_zs = sale_zsa = None
        if self.config.sale_enabled:
            sale_started = time.perf_counter()
            sale_zs = self.sale_online.state(_observation_args(batch))
            sale_zsa = self.sale_online.state_action(
                sale_zs, batch.applied_action
            )
            with torch.no_grad():
                sale_next = self.sale_online.state(
                    _observation_args(batch, next_state=True)
                )
            sale_loss = F.mse_loss(sale_zsa, sale_next.detach())
            self._sync()
            sale_forward_ms = (time.perf_counter() - sale_started) * 1000.0
            sale_started = time.perf_counter()
            self.sale_optimizer.zero_grad(set_to_none=True)
            sale_loss.backward()
            sale_grad = _clip_grad_norm(
                self.sale_online.parameters(), self.config.encoder_grad_clip
            )
            self.sale_optimizer.step()
            self._sync()
            sale_backward_ms = (time.perf_counter() - sale_started) * 1000.0
            self.sale_update_count += 1
            sale_loss_value = float(sale_loss.detach().cpu())

        with torch.no_grad():
            next_encoding = self.target_encoder(
                *_observation_args(batch, next_state=True),
                return_attention=False,
            )
            target_sale_zs = target_sale_zsa = None
            if self.config.sale_enabled:
                target_sale_zs = self.sale_target_fixed.state(
                    _observation_args(batch, next_state=True)
                )
                next_policy = self.target_actor(
                    next_encoding.state, target_sale_zs
                ).action
            else:
                next_policy = self.target_actor(next_encoding.state).action
            next_applied = target_applied_action(
                next_policy,
                action_scale=self.applied_action_scale,
                noise_std=self.config.target_noise,
                noise_clip=self.config.target_noise_clip,
            )
            if self.config.sale_enabled:
                target_sale_zsa = self.sale_target_fixed.state_action(
                    target_sale_zs, next_applied
                )
            target_q_pair = self.target_critic(
                next_encoding.state, next_applied,
                target_sale_zs, target_sale_zsa,
            )
            tq1, tq2 = target_q_pair.q1, target_q_pair.q2
            if (
                math.isfinite(self.fixed_target_q_min)
                and math.isfinite(self.fixed_target_q_max)
            ):
                tq1 = tq1.clamp(
                    self.fixed_target_q_min, self.fixed_target_q_max
                )
                tq2 = tq2.clamp(
                    self.fixed_target_q_min, self.fixed_target_q_max
                )
            target_q = bellman_target(
                batch.reward, batch.done,
                tq1, tq2,
                gamma=self.config.gamma,
            )
            target_q_min = float(target_q.min().detach().cpu())
            target_q_max = float(target_q.max().detach().cpu())
            if not math.isfinite(target_q_min) or not math.isfinite(target_q_max):
                raise FloatingPointError("target-Q bounds are non-finite")
            self.current_target_q_min = min(
                self.current_target_q_min, target_q_min
            )
            self.current_target_q_max = max(
                self.current_target_q_max, target_q_max
            )

            fixed_zs = fixed_zsa = None
            if self.config.sale_enabled:
                fixed_zs = self.sale_fixed.state(_observation_args(batch))
                fixed_zsa = self.sale_fixed.state_action(
                    fixed_zs, batch.applied_action
                )

        critic_start = time.perf_counter()
        # Contract: replay policy_action is diagnostics only. Critic supervision
        # consumes the residual that was actually applied to the simulator.
        q_pair = self.critic(
            online_encoding.state, batch.applied_action, fixed_zs, fixed_zsa
        )
        loss_function = (
            F.smooth_l1_loss
            if self.config.resolved_critic_loss_mode == "huber"
            else F.mse_loss
        )
        q1_loss = loss_function(q_pair.q1, target_q)
        q2_loss = loss_function(q_pair.q2, target_q)
        critic_loss = q1_loss + q2_loss
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("critic loss is non-finite")
        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        q1_grad = _gradient_norm(self.critic.q1.parameters())
        q2_grad = _gradient_norm(self.critic.q2.parameters())
        encoder_grad = _clip_grad_norm(
            self.encoder.parameters(), self.config.encoder_grad_clip
        )
        critic_grad = _clip_grad_norm(
            self.critic.parameters(), self.config.critic_grad_clip
        )
        self.encoder_optimizer.step()
        self.critic_optimizer.step()
        td_priority = torch.maximum(
            (q_pair.q1.detach() - target_q).abs(),
            (q_pair.q2.detach() - target_q).abs(),
        ).reshape(-1)
        if self.config.lap_enabled:
            self.replay.update_priorities(batch.sample_keys, td_priority.cpu().numpy())
        self._sync()
        critic_update_ms = (time.perf_counter() - critic_start) * 1000

        actor_updated = step % self.config.policy_update_delay == 0
        actor_loss_value = 0.0
        actor_grad = 0.0
        actor_update_ms = 0.0
        policy_mean = policy_std = applied_mean = applied_std = 0.0
        if actor_updated:
            actor_start = time.perf_counter()
            # The encoder is owned by the critic representation update only.
            # Actor optimization uses a detached post-critic encoding.
            with torch.no_grad():
                actor_state = self.encoder(
                    *_observation_args(batch), return_attention=False
                ).state
                actor_sale_zs = fixed_zs
            policy_output = self.actor(actor_state, actor_sale_zs)
            actor_applied = scale_policy_action(
                policy_output.action, self.applied_action_scale
            )
            critic_flags = [p.requires_grad for p in self.critic.parameters()]
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            actor_sale_zsa = (
                self.sale_fixed.state_action(actor_sale_zs, actor_applied)
                if self.config.sale_enabled else None
            )
            q1_actor = self.critic.q1_value(
                actor_state, actor_applied, actor_sale_zs, actor_sale_zsa
            )
            pretanh_penalty = (
                self.config.pretanh_penalty
                * policy_output.pre_tanh.square().mean()
            )
            actor_loss = -q1_actor.mean() + pretanh_penalty
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = _clip_grad_norm(
                self.actor.parameters(), self.config.actor_grad_clip
            )
            self.actor_optimizer.step()
            for parameter, flag in zip(self.critic.parameters(), critic_flags):
                parameter.requires_grad_(flag)
            self.actor_update_count += 1
            actor_loss_value = float(actor_loss.detach().cpu())
            self.last_actor_loss = actor_loss_value
            self.last_actor_grad_norm = actor_grad
            self.last_actor_update_step = step
            policy_mean = float(policy_output.action.detach().mean().cpu())
            policy_std = float(policy_output.action.detach().std().cpu())
            applied_mean = float(actor_applied.detach().mean().cpu())
            applied_std = float(actor_applied.detach().std().cpu())
            self._sync()
            actor_update_ms = (time.perf_counter() - actor_start) * 1000

        target_updated = step % self.config.target_update_interval == 0
        if target_updated:
            self._hard_update_targets()
        if (
            step == 1
            or target_updated
            or step % DIAGNOSTICS_INTERVAL == 0
        ):
            self._distance_diagnostics.update({
                "target/encoder_distance": _parameter_distance(
                    self.encoder, self.target_encoder
                ),
                "target/actor_distance": _parameter_distance(
                    self.actor, self.target_actor
                ),
                "target/critic_distance": _parameter_distance(
                    self.critic, self.target_critic
                ),
                "sale/fixed_online_distance": (
                    _parameter_distance(self.sale_fixed, self.sale_online)
                    if self.config.sale_enabled else 0.0
                ),
                "sale/target_fixed_distance": (
                    _parameter_distance(
                        self.sale_target_fixed, self.sale_fixed
                    )
                    if self.config.sale_enabled else 0.0
                ),
                **self.twin_parameter_diagnostics(),
            })

        td1 = (q_pair.q1.detach() - target_q).abs()
        td2 = (q_pair.q2.detach() - target_q).abs()
        q_abs_diff = (q_pair.q1.detach() - q_pair.q2.detach()).abs()
        finite_values = (
            critic_loss.detach(), q_pair.q1.detach(), q_pair.q2.detach(),
            target_q.detach(), td1, td2,
        )
        finite_ratio = float(torch.cat(
            [value.reshape(-1).isfinite().float() for value in finite_values]
        ).mean().cpu())
        diagnostics = {
            "learner/critic_loss": float(critic_loss.detach().cpu()),
            "learner/actor_loss_last": self.last_actor_loss,
            "learner/actor_grad_norm_last": self.last_actor_grad_norm,
            "learner/actor_last_update_step": float(
                self.last_actor_update_step
            ),
            "learner/actor_updates_total": float(self.actor_update_count),
            "learner/actor_updated_this_step": float(actor_updated),
            # The contextual encoder is optimized jointly by the critic
            # objective; there is no independent encoder loss.
            "learner/encoder_joint_critic_loss": float(
                critic_loss.detach().cpu()
            ),
            "critic/q1_mean": float(q_pair.q1.detach().mean().cpu()),
            "critic/q2_mean": float(q_pair.q2.detach().mean().cpu()),
            "critic/q_abs_diff_mean": float(q_abs_diff.mean().cpu()),
            "critic/q_abs_diff_max": float(q_abs_diff.max().cpu()),
            "critic/q1_loss": float(q1_loss.detach().cpu()),
            "critic/q2_loss": float(q2_loss.detach().cpu()),
            "critic/q1_grad_norm": q1_grad,
            "critic/q2_grad_norm": q2_grad,
            "critic/q_min": float(
                torch.minimum(q_pair.q1, q_pair.q2).detach().min().cpu()
            ),
            "critic/q_max": float(
                torch.maximum(q_pair.q1, q_pair.q2).detach().max().cpu()
            ),
            "critic/target_q_mean": float(target_q.mean().cpu()),
            "critic/target_q_std": float(target_q.std().cpu()),
            "critic/td_error_mean": float(
                torch.maximum(td1, td2).mean().cpu()
            ),
            "critic/td_error_max": float(
                torch.maximum(td1, td2).max().cpu()
            ),
            "grad/encoder_norm": encoder_grad,
            "grad/critic_norm": critic_grad,
            **self._distance_diagnostics,
            "update/learner_count": float(self.learner_update_count),
            "update/actor_count": float(self.actor_update_count),
            "update/target_count": float(self.target_update_count),
            "action/policy_mean": policy_mean,
            "action/policy_std": policy_std,
            "action/applied_mean": applied_mean,
            "action/applied_std": applied_std,
            "numeric/learner_finite_ratio": finite_ratio,
            "sale/enabled": float(self.config.sale_enabled),
            "sale/loss": sale_loss_value,
            "sale/online_grad_norm": sale_grad,
            "sale/fixed_grad_norm": 0.0,
            "sale/target_fixed_grad_norm": 0.0,
            "sale/update_count": float(self.sale_update_count),
            "sale/fixed_generation": float(self.sale_fixed_generation),
            "sale/zs_mean": (
                float(sale_zs.detach().mean().cpu())
                if sale_zs is not None else 0.0
            ),
            "sale/zs_std": (
                float(sale_zs.detach().std().cpu())
                if sale_zs is not None else 0.0
            ),
            "sale/zs_abs_mean": (
                float(sale_zs.detach().abs().mean().cpu())
                if sale_zs is not None else 0.0
            ),
            "sale/zsa_mean": (
                float(sale_zsa.detach().mean().cpu())
                if sale_zsa is not None else 0.0
            ),
            "sale/zsa_std": (
                float(sale_zsa.detach().std().cpu())
                if sale_zsa is not None else 0.0
            ),
            "sale/prediction_error_mean": (
                float((sale_zsa.detach() - sale_next).abs().mean().cpu())
                if sale_zsa is not None else 0.0
            ),
            "sale/prediction_error_max": (
                float((sale_zsa.detach() - sale_next).abs().max().cpu())
                if sale_zsa is not None else 0.0
            ),
            "timing/sale_forward_ms": sale_forward_ms,
            "timing/sale_backward_ms": sale_backward_ms,
            "lap/enabled": float(self.config.lap_enabled),
            "lap/td_error_mean": float(td_priority.mean().cpu()),
            "lap/td_error_max": float(td_priority.max().cpu()),
            "value/current_target_q_min": (
                float(self.current_target_q_min)
                if math.isfinite(self.current_target_q_min) else 0.0
            ),
            "value/current_target_q_max": (
                float(self.current_target_q_max)
                if math.isfinite(self.current_target_q_max) else 0.0
            ),
            "value/fixed_target_q_min": (
                float(self.fixed_target_q_min)
                if math.isfinite(self.fixed_target_q_min) else 0.0
            ),
            "value/fixed_target_q_max": (
                float(self.fixed_target_q_max)
                if math.isfinite(self.fixed_target_q_max) else 0.0
            ),
            "learner/replay_policy_action_std": float(
                batch.policy_action.detach().std().cpu()
            ),
            "learner/replay_applied_action_std": float(
                batch.applied_action.detach().std().cpu()
            ),
            "learner/applied_action_scale": self.applied_action_scale,
            "learner/critic_action_matches_applied": 1.0,
            "timing/encoder_forward_ms": encoder_forward_ms,
            "timing/critic_update_ms": critic_update_ms,
            "timing/actor_update_ms": actor_update_ms,
            "timing/total_learner_update_ms": (
                time.perf_counter() - total_start
            ) * 1000,
        }
        diagnostics.update({
            key: value for key, value in self.replay.diagnostics().items()
            if key.startswith("lap/") or key in {
                "replay/sample_materialize_ms",
                "replay/host_to_device_ms",
            }
        })
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError("learner diagnostics contain NaN or Inf")
        self.last_diagnostics = diagnostics
        return ContextualLearnerUpdate(
            diagnostics=diagnostics,
            actor_updated=actor_updated,
            target_updated=target_updated,
        )
