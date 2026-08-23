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
from .stacking import (
    encode_observation_stack,
    flatten_action_stack,
    flatten_state_stack,
    replace_current_action,
)


DIAGNOSTICS_INTERVAL = 10


def _observation_args(batch, *, next_state=False):
    prefix = "next_" if next_state else ""
    return (
        getattr(batch, f"{prefix}center_local"),
        getattr(batch, f"{prefix}incoming_local"),
        getattr(batch, f"{prefix}outgoing_local"),
        batch.center_rail_index,
        batch.incoming_rail_indices,
        batch.outgoing_rail_indices,
        getattr(batch, f"{prefix}incoming_relation"),
        getattr(batch, f"{prefix}outgoing_relation"),
        getattr(batch, f"{prefix}global_state"),
    )


def _clip_grad_norm(parameters, max_norm: float) -> torch.Tensor:
    total = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm, error_if_nonfinite=True
    )
    return total.detach()


def _parameter_distance_tensor(source, target) -> torch.Tensor:
    squares = [
        torch.sum((left.detach().double() - right.detach().double()).square())
        for left, right in zip(source.parameters(), target.parameters())
    ]
    if not squares:
        return torch.zeros((), dtype=torch.float64)
    return torch.sqrt(torch.stack(squares).sum())


def _gradient_norm(parameters) -> torch.Tensor:
    squares = [
        torch.sum(parameter.grad.detach().double().square())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return torch.zeros((), dtype=torch.float64)
    return torch.sqrt(torch.stack(squares).sum())


def _scalar_tensors_to_floats(values) -> dict[str, float]:
    """Transfer a group of scalar diagnostics with one device synchronization."""
    if not values:
        return {}
    keys = tuple(values)
    scalars = [values[key].detach().reshape(()) for key in keys]
    device = scalars[0].device
    scalars = [value.to(device=device, dtype=torch.float64) for value in scalars]
    host_values = torch.stack(scalars).cpu().tolist()
    return dict(zip(keys, (float(value) for value in host_values)))


def _parameter_ids(parameters) -> set[int]:
    return {id(parameter) for parameter in parameters}


def _optimizer_parameter_ids(optimizer) -> tuple[list[int], set[int]]:
    ordered = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    return ordered, set(ordered)


def _twin_parameter_diagnostic_tensors(critic) -> dict[str, torch.Tensor]:
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
        "critic/parameter_l2_distance": torch.linalg.vector_norm(flattened),
        "critic/parameter_max_abs_diff": flattened.abs().max(),
    }


def _twin_parameter_diagnostics(critic) -> dict[str, float]:
    return _scalar_tensors_to_floats(
        _twin_parameter_diagnostic_tensors(critic)
    )


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
        if (
            int(getattr(replay, "num_stacks", 1))
            != self.network_config.num_stacks
            or int(getattr(replay, "stack_interval", 1))
            != self.network_config.stack_interval
        ):
            raise ValueError(
                "replay and network stack configurations must match"
            )
        topology = getattr(replay, "topology", None)
        if topology is not None:
            physical_count = len(topology.all_rail_ids)
            neighbor_count = int(topology.incoming_neighbor_ids.shape[1])
            if self.network_config.num_rails != physical_count:
                raise ValueError(
                    "network num_rails must match replay physical topology: "
                    f"network={self.network_config.num_rails}, "
                    f"topology={physical_count}"
                )
            if (
                self.network_config.neighbor_count != neighbor_count
                or topology.outgoing_neighbor_ids.shape[1] != neighbor_count
            ):
                raise ValueError(
                    "network neighbor_count must match replay topology"
                )
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

    def _stage_timer_start(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def _stage_timer_stop(self, started):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return started, event
        return (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _stage_timer_elapsed(timer) -> float:
        if isinstance(timer, tuple):
            started, stopped = timer
            return float(started.elapsed_time(stopped))
        return float(timer)

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

        encoder_timer = self._stage_timer_start()
        online_encoding = encode_observation_stack(
            self.encoder,
            _observation_args(batch),
            return_attention=False,
        )
        online_state = flatten_state_stack(
            online_encoding.state, self.network_config.num_stacks
        )
        replay_previous_applied = flatten_action_stack(
            batch.previous_applied_action,
            num_stacks=self.network_config.num_stacks,
            action_dim=self.network_config.action_dim,
        )
        replay_applied = flatten_action_stack(
            batch.applied_action,
            num_stacks=self.network_config.num_stacks,
            action_dim=self.network_config.action_dim,
        )
        encoder_timer = self._stage_timer_stop(encoder_timer)

        sale_grad = None
        sale_forward_timer = None
        sale_backward_timer = None
        sale_zs = sale_zsa = None
        if self.config.sale_enabled:
            sale_forward_timer = self._stage_timer_start()
            sale_zs = self.sale_online.state(_observation_args(batch))
            sale_zsa = self.sale_online.state_action(
                sale_zs, replay_applied
            )
            with torch.no_grad():
                sale_next = self.sale_online.state(
                    _observation_args(batch, next_state=True)
                )
            sale_loss = F.mse_loss(sale_zsa, sale_next.detach())
            sale_forward_timer = self._stage_timer_stop(sale_forward_timer)
            sale_backward_timer = self._stage_timer_start()
            self.sale_optimizer.zero_grad(set_to_none=True)
            sale_loss.backward()
            sale_grad = _clip_grad_norm(
                self.sale_online.parameters(), self.config.encoder_grad_clip
            )
            self.sale_optimizer.step()
            sale_backward_timer = self._stage_timer_stop(sale_backward_timer)
            self.sale_update_count += 1

        with torch.no_grad():
            next_encoding = encode_observation_stack(
                self.target_encoder,
                _observation_args(batch, next_state=True),
                return_attention=False,
            )
            next_state = flatten_state_stack(
                next_encoding.state, self.network_config.num_stacks
            )
            next_previous_applied = flatten_action_stack(
                batch.next_previous_applied_action,
                num_stacks=self.network_config.num_stacks,
                action_dim=self.network_config.action_dim,
            )
            target_sale_zs = target_sale_zsa = None
            if self.config.sale_enabled:
                target_sale_zs = self.sale_target_fixed.state(
                    _observation_args(batch, next_state=True)
                )
                next_policy = self.target_actor(
                    next_state,
                    target_sale_zs,
                    previous_action=next_previous_applied,
                ).action
            else:
                next_policy = self.target_actor(
                    next_state,
                    previous_action=next_previous_applied,
                ).action
            next_current_applied = target_applied_action(
                next_policy,
                action_scale=self.applied_action_scale,
                noise_std=self.config.target_noise,
                noise_clip=self.config.target_noise_clip,
            )
            next_applied = replace_current_action(
                batch.next_applied_action,
                next_current_applied,
                num_stacks=self.network_config.num_stacks,
                action_dim=self.network_config.action_dim,
            )
            if self.config.sale_enabled:
                target_sale_zsa = self.sale_target_fixed.state_action(
                    target_sale_zs, next_applied
                )
            target_q_pair = self.target_critic(
                next_state, next_applied,
                target_sale_zs, target_sale_zsa,
                previous_action=next_previous_applied,
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
            target_q_bounds = _scalar_tensors_to_floats({
                "min": target_q.min(),
                "max": target_q.max(),
            })
            target_q_min = target_q_bounds["min"]
            target_q_max = target_q_bounds["max"]
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
                    fixed_zs, replay_applied
                )

        critic_timer = self._stage_timer_start()
        # Contract: replay policy_action is diagnostics only. Critic supervision
        # consumes the residual that was actually applied to the simulator.
        q_pair = self.critic(
            online_state,
            replay_applied,
            fixed_zs,
            fixed_zsa,
            previous_action=replay_previous_applied,
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
        critic_timer = self._stage_timer_stop(critic_timer)

        actor_updated = step % self.config.policy_update_delay == 0
        actor_grad = None
        actor_timer = None
        if actor_updated:
            actor_timer = self._stage_timer_start()
            # The encoder is owned by the critic representation update only.
            # Actor optimization uses a detached post-critic encoding.
            with torch.no_grad():
                actor_encoding = encode_observation_stack(
                    self.encoder,
                    _observation_args(batch),
                    return_attention=False,
                )
                actor_state = flatten_state_stack(
                    actor_encoding.state, self.network_config.num_stacks
                )
                actor_sale_zs = fixed_zs
            policy_output = self.actor(
                actor_state,
                actor_sale_zs,
                previous_action=replay_previous_applied,
            )
            actor_current_applied = scale_policy_action(
                policy_output.action, self.applied_action_scale
            )
            actor_applied = replace_current_action(
                batch.applied_action,
                actor_current_applied,
                num_stacks=self.network_config.num_stacks,
                action_dim=self.network_config.action_dim,
            )
            critic_flags = [p.requires_grad for p in self.critic.parameters()]
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            actor_sale_zsa = (
                self.sale_fixed.state_action(actor_sale_zs, actor_applied)
                if self.config.sale_enabled else None
            )
            q1_actor = self.critic.q1_value(
                actor_state,
                actor_applied,
                actor_sale_zs,
                actor_sale_zsa,
                previous_action=replay_previous_applied,
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
            actor_timer = self._stage_timer_stop(actor_timer)

        target_updated = step % self.config.target_update_interval == 0
        if target_updated:
            self._hard_update_targets()
        if (
            step == 1
            or target_updated
            or step % DIAGNOSTICS_INTERVAL == 0
        ):
            distance_tensors = {
                "target/encoder_distance": _parameter_distance_tensor(
                    self.encoder, self.target_encoder
                ),
                "target/actor_distance": _parameter_distance_tensor(
                    self.actor, self.target_actor
                ),
                "target/critic_distance": _parameter_distance_tensor(
                    self.critic, self.target_critic
                ),
                "sale/fixed_online_distance": (
                    _parameter_distance_tensor(
                        self.sale_fixed, self.sale_online
                    )
                    if self.config.sale_enabled
                    else torch.zeros((), device=self.device)
                ),
                "sale/target_fixed_distance": (
                    _parameter_distance_tensor(
                        self.sale_target_fixed, self.sale_fixed
                    )
                    if self.config.sale_enabled
                    else torch.zeros((), device=self.device)
                ),
                **_twin_parameter_diagnostic_tensors(self.critic),
            }
            self._distance_diagnostics.update(
                _scalar_tensors_to_floats(distance_tensors)
            )

        td1 = (q_pair.q1.detach() - target_q).abs()
        td2 = (q_pair.q2.detach() - target_q).abs()
        q_abs_diff = (q_pair.q1.detach() - q_pair.q2.detach()).abs()
        finite_values = (
            critic_loss.detach(), q_pair.q1.detach(), q_pair.q2.detach(),
            target_q.detach(), td1, td2,
        )
        finite_ratio = torch.cat(
            [value.reshape(-1).isfinite().float() for value in finite_values]
        ).mean()
        tensor_diagnostics = {
            "learner/critic_loss": critic_loss,
            "learner/encoder_joint_critic_loss": critic_loss,
            "critic/q1_mean": q_pair.q1.mean(),
            "critic/q2_mean": q_pair.q2.mean(),
            "critic/q_abs_diff_mean": q_abs_diff.mean(),
            "critic/q_abs_diff_max": q_abs_diff.max(),
            "critic/q1_loss": q1_loss,
            "critic/q2_loss": q2_loss,
            "critic/q1_grad_norm": q1_grad,
            "critic/q2_grad_norm": q2_grad,
            "critic/q_min": torch.minimum(q_pair.q1, q_pair.q2).min(),
            "critic/q_max": torch.maximum(q_pair.q1, q_pair.q2).max(),
            "critic/target_q_mean": target_q.mean(),
            "critic/target_q_std": target_q.std(),
            "critic/td_error_mean": torch.maximum(td1, td2).mean(),
            "critic/td_error_max": torch.maximum(td1, td2).max(),
            "grad/encoder_norm": encoder_grad,
            "grad/critic_norm": critic_grad,
            "numeric/learner_finite_ratio": finite_ratio,
            "lap/td_error_mean": td_priority.mean(),
            "lap/td_error_max": td_priority.max(),
            "learner/replay_policy_action_std": batch.policy_action.std(),
            "learner/replay_applied_action_std": batch.applied_action.std(),
        }
        if self.config.sale_enabled:
            tensor_diagnostics.update({
                "sale/loss": sale_loss,
                "sale/online_grad_norm": sale_grad,
                "sale/zs_mean": sale_zs.mean(),
                "sale/zs_std": sale_zs.std(),
                "sale/zs_abs_mean": sale_zs.abs().mean(),
                "sale/zsa_mean": sale_zsa.mean(),
                "sale/zsa_std": sale_zsa.std(),
                "sale/prediction_error_mean": (
                    sale_zsa.detach() - sale_next
                ).abs().mean(),
                "sale/prediction_error_max": (
                    sale_zsa.detach() - sale_next
                ).abs().max(),
            })
        if actor_updated:
            tensor_diagnostics.update({
                "_actor_loss": actor_loss,
                "_actor_grad_norm": actor_grad,
                "action/policy_mean": policy_output.action.mean(),
                "action/policy_std": policy_output.action.std(),
                "action/applied_mean": actor_current_applied.mean(),
                "action/applied_std": actor_current_applied.std(),
            })
        scalar_diagnostics = _scalar_tensors_to_floats(tensor_diagnostics)

        if actor_updated:
            self.last_actor_loss = scalar_diagnostics.pop("_actor_loss")
            self.last_actor_grad_norm = scalar_diagnostics.pop(
                "_actor_grad_norm"
            )
            self.last_actor_update_step = step
        diagnostics = {
            **scalar_diagnostics,
            "learner/actor_loss_last": self.last_actor_loss,
            "learner/actor_grad_norm_last": self.last_actor_grad_norm,
            "learner/actor_last_update_step": float(
                self.last_actor_update_step
            ),
            "learner/actor_updates_total": float(self.actor_update_count),
            "learner/actor_updated_this_step": float(actor_updated),
            # The contextual encoder is optimized jointly by the critic
            # objective; there is no independent encoder loss.
            **self._distance_diagnostics,
            "update/learner_count": float(self.learner_update_count),
            "update/actor_count": float(self.actor_update_count),
            "update/target_count": float(self.target_update_count),
            "action/policy_mean": scalar_diagnostics.get(
                "action/policy_mean", 0.0
            ),
            "action/policy_std": scalar_diagnostics.get(
                "action/policy_std", 0.0
            ),
            "action/applied_mean": scalar_diagnostics.get(
                "action/applied_mean", 0.0
            ),
            "action/applied_std": scalar_diagnostics.get(
                "action/applied_std", 0.0
            ),
            "sale/enabled": float(self.config.sale_enabled),
            "sale/loss": scalar_diagnostics.get("sale/loss", 0.0),
            "sale/online_grad_norm": scalar_diagnostics.get(
                "sale/online_grad_norm", 0.0
            ),
            "sale/fixed_grad_norm": 0.0,
            "sale/target_fixed_grad_norm": 0.0,
            "sale/update_count": float(self.sale_update_count),
            "sale/fixed_generation": float(self.sale_fixed_generation),
            "sale/zs_mean": scalar_diagnostics.get("sale/zs_mean", 0.0),
            "sale/zs_std": scalar_diagnostics.get("sale/zs_std", 0.0),
            "sale/zs_abs_mean": scalar_diagnostics.get(
                "sale/zs_abs_mean", 0.0
            ),
            "sale/zsa_mean": scalar_diagnostics.get("sale/zsa_mean", 0.0),
            "sale/zsa_std": scalar_diagnostics.get("sale/zsa_std", 0.0),
            "sale/prediction_error_mean": scalar_diagnostics.get(
                "sale/prediction_error_mean", 0.0
            ),
            "sale/prediction_error_max": scalar_diagnostics.get(
                "sale/prediction_error_max", 0.0
            ),
            "timing/sale_forward_ms": (
                self._stage_timer_elapsed(sale_forward_timer)
                if sale_forward_timer is not None else 0.0
            ),
            "timing/sale_backward_ms": (
                self._stage_timer_elapsed(sale_backward_timer)
                if sale_backward_timer is not None else 0.0
            ),
            "lap/enabled": float(self.config.lap_enabled),
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
            "learner/applied_action_scale": self.applied_action_scale,
            "learner/critic_action_matches_applied": 1.0,
            "timing/encoder_forward_ms": self._stage_timer_elapsed(
                encoder_timer
            ),
            "timing/critic_update_ms": self._stage_timer_elapsed(
                critic_timer
            ),
            "timing/actor_update_ms": (
                self._stage_timer_elapsed(actor_timer)
                if actor_timer is not None else 0.0
            ),
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
