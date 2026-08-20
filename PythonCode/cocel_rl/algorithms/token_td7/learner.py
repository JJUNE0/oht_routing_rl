import copy
import math
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from .networks import RegionTD7Actor, RegionTD7Critic
from .numerics import (
    NumericalIntegrityError,
    assert_finite_modules,
    assert_finite_tensors,
    assert_finite_tree,
)
from .replay_buffer import RegionReplayBuffer


def _clip_grad_norm(named_parameters, max_norm, stage, context=None):
    """clip_grad_norm_는 클리핑과 동시에 클리핑 전(raw) total norm을 돌려주므로,
    grad/*_norm 진단 로그는 그대로 두면서 폭발(NaN/Inf)만 차단할 수 있다."""
    named_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.grad is not None
    ]
    if not named_parameters:
        return 0.0, {}

    assert_finite_tensors(
        stage,
        ((f"gradient.{name}", parameter.grad) for name, parameter in named_parameters),
        context=context,
    )
    per_parameter = {}
    squared_norms = []
    for name, parameter in named_parameters:
        gradient = parameter.grad.detach()
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        norm = torch.linalg.vector_norm(gradient, ord=2, dtype=torch.float64)
        norm_value = float(norm.cpu())
        if not math.isfinite(norm_value):
            raise NumericalIntegrityError(
                f"[{stage}] FP64 gradient norm is non-finite for parameter={name}"
                f"{f' | {context}' if context else ''}"
            )
        per_parameter[name] = norm_value
        squared_norms.append(norm_value * norm_value)

    total = math.sqrt(math.fsum(squared_norms))
    if not math.isfinite(total):
        raise NumericalIntegrityError(
            f"[{stage}] FP64 gradient total norm is non-finite"
            f"{f' | {context}' if context else ''}"
        )

    clip_coefficient = float(max_norm) / (total + 1e-12)
    if clip_coefficient < 1.0:
        with torch.no_grad():
            for _, parameter in named_parameters:
                parameter.grad.mul_(clip_coefficient)
    return total, per_parameter


def _masked_mean_square(value, mask):
    """Mean square over real tokens only; padded policy outputs are excluded."""
    valid = mask.unsqueeze(-1).expand_as(value).bool()
    selected = value[valid]
    if selected.numel() == 0:
        return value.sum() * 0.0
    return selected.square().mean()


class TokenTD7Learner:
    def __init__(self, rail_dim, action_dim, action_bound, context_dim, config):
        self.config = config
        self.device = config["device"]
        algo = config["algorithm"]
        token_cfg = config.get("token_td7", {})
        zs_dim = int(algo.get("zs_dim", 256))
        hdim = int(algo.get("hdim", 256))
        embed_dim = int(token_cfg.get("embed_dim", 64))
        num_heads = int(token_cfg.get("num_heads", 4))
        expl = float(algo.get("noise_scale", 0.1))

        self.actor = RegionTD7Actor(
            rail_dim=rail_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            context_dim=context_dim,
            zs_dim=zs_dim,
            hdim=hdim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            exploration_noise=expl,
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = RegionTD7Critic(
            action_dim=action_dim,
            zs_dim=zs_dim,
            hdim=hdim,
            embed_dim=embed_dim,
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.mlp.parameters(),
            lr=config["actor_lr"],
            eps=config["adam_eps"],
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config["critic_lr"],
            eps=config["adam_eps"],
        )
        self.encoder_optimizer = torch.optim.Adam(
            self.actor.encoder.parameters(),
            lr=algo.get("encoder_lr", 3e-4),
        )
        self.buffer = RegionReplayBuffer(
            capacity=int(config["buffer_capacity"]),
            device=self.device,
            prioritized=True,
            min_priority=float(algo.get("min_priority", 1.0)),
            lap_alpha=float(algo.get("lap_alpha", 0.4)),
        )

        self.max_rollout = int(config.get("max_rollout", 1))
        self.gamma = float(config.get("gamma", 0.99))
        self.encoder_grad_clip = float(algo.get("encoder_grad_clip", 1.0))
        self.encoder_grad_abort_norm = float(algo.get("encoder_grad_abort_norm", 1_000.0))
        self.failure_artifact_topk = int(algo.get("failure_artifact_topk", 16))
        self.numeric_artifact_dir = config.get(
            "numeric_artifact_dir",
            os.path.join(config.get("save_dir", "."), "failure_artifacts"),
        )
        self.actor_grad_clip = float(algo.get("actor_grad_clip", 25.0))
        self.critic_grad_clip = float(algo.get("critic_grad_clip", 10.0))
        self.policy_freq = int(algo.get("policy_update_delay", 2))
        self.target_update_rate = int(algo.get("target_update_rate", 250))
        self.target_noise = float(algo.get("target_noise_scale", 0.2))
        self.noise_clip = float(algo.get("target_noise_clip", 0.5))
        self.action_low = float(action_bound[0])
        self.action_high = float(action_bound[1])

        self.training_steps = 0
        self.total_losses = {}
        self.value_max = -1e8
        self.value_min = 1e8
        self.target_max = 0.0
        self.target_min = 0.0
        self.numeric_failure = None

        # curriculum scale이 buffer에 저장된 action과 actor loss action의 분포를 정렬한다.
        # The caller updates this before each learn() call.
        self.curriculum_scale = 1.0
        self.pretanh_penalty_coef = float(algo.get("pretanh_penalty_coef", 0.1))

    def _numeric_context(self, training_step=None):
        step = self.training_steps if training_step is None else int(training_step)
        sampled = self.buffer.ind
        sample_text = "none"
        if sampled is not None:
            sample_text = sampled[:8].tolist()
        return (
            f"training_step={step}, buffer_size={self.buffer.size}, "
            f"sample_indices={sample_text}"
        )

    def _latch_numeric_failure(self, error):
        if self.numeric_failure is None:
            self.numeric_failure = str(error)
            print(f"[NUMERIC FAIL-FAST] {self.numeric_failure}")
        for optimizer in (
            self.encoder_optimizer,
            self.critic_optimizer,
            self.actor_optimizer,
        ):
            optimizer.zero_grad(set_to_none=True)

    def _guard_tensors(self, stage, named_tensors, training_step):
        try:
            assert_finite_tensors(
                stage,
                named_tensors,
                context=self._numeric_context(training_step),
            )
        except NumericalIntegrityError as error:
            self._latch_numeric_failure(error)
            raise

    def _guard_modules(self, stage, named_modules, training_step):
        try:
            assert_finite_modules(
                stage,
                named_modules,
                context=self._numeric_context(training_step),
            )
        except NumericalIntegrityError as error:
            self._latch_numeric_failure(error)
            raise

    def _safe_clip_grad_norm(self, stage, named_parameters, max_norm, training_step):
        try:
            total, _ = _clip_grad_norm(
                named_parameters,
                max_norm,
                stage,
                context=self._numeric_context(training_step),
            )
            return total
        except NumericalIntegrityError as error:
            self._latch_numeric_failure(error)
            raise

    @staticmethod
    def _cpu_detached_tree(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {
                key: TokenTD7Learner._cpu_detached_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return type(value)(
                TokenTD7Learner._cpu_detached_tree(item) for item in value
            )
        return value

    @staticmethod
    def _best_effort_gradient_norms(named_parameters):
        norms = {}
        for name, parameter in named_parameters:
            if parameter.grad is None:
                continue
            try:
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    gradient = gradient.coalesce().values()
                norms[name] = float(
                    torch.linalg.vector_norm(
                        gradient,
                        ord=2,
                        dtype=torch.float64,
                    ).cpu()
                )
            except Exception:
                norms[name] = float("nan")
        return norms

    def _save_encoder_failure_artifact(
        self,
        reason,
        training_step,
        batch,
        encoder_loss,
        pred_zs,
        next_zs,
        per_parameter_grad_norms,
    ):
        """Persist the failing batch and a compact, high-value activation trace."""
        try:
            per_sample_loss = F.mse_loss(pred_zs, next_zs, reduction="none")
            per_sample_loss = per_sample_loss.flatten(1).mean(1).detach()
            top_count = min(
                max(1, self.failure_artifact_topk),
                int(per_sample_loss.numel()),
            )
            top_positions = torch.topk(per_sample_loss, k=top_count).indices

            current_activations = self.actor.encoder.diagnostic_activations(
                batch["rail_feat"].index_select(0, top_positions),
                batch["action"].index_select(0, top_positions),
                batch["context"].index_select(0, top_positions),
                batch["mask"].index_select(0, top_positions),
            )
            next_activations = self.actor.encoder.backbone.diagnostic_activations(
                batch["next_rail_feat"].index_select(0, top_positions),
                batch["next_context"].index_select(0, top_positions),
                batch["next_mask"].index_select(0, top_positions),
            )
            replay = self.buffer.diagnostic_priority_snapshot()
            sampled_indices = replay.get("sample_indices")
            top_replay_indices = None
            if sampled_indices is not None:
                top_replay_indices = sampled_indices[
                    top_positions.detach().cpu().numpy()
                ]

            artifact = {
                "format_version": 1,
                "reason": str(reason),
                "training_step": int(training_step),
                "buffer_size": int(self.buffer.size),
                "encoder_loss": float(encoder_loss.detach().cpu()),
                "encoder_grad_clip": float(self.encoder_grad_clip),
                "encoder_grad_abort_norm": float(self.encoder_grad_abort_norm),
                "gradient_norm_dtype": "float64",
                "parameter_gradient_norms_raw": dict(per_parameter_grad_norms),
                "encoder_state_dict": self._cpu_detached_tree(
                    self.actor.encoder.state_dict()
                ),
                "encoder_optimizer_state_dict": self._cpu_detached_tree(
                    self.encoder_optimizer.state_dict()
                ),
                "batch": self._cpu_detached_tree(batch),
                "replay": replay,
                "per_sample_encoder_loss": per_sample_loss.cpu(),
                "top_batch_positions": top_positions.cpu(),
                "top_replay_indices": top_replay_indices,
                "top_current_activations": self._cpu_detached_tree(current_activations),
                "top_next_activations": self._cpu_detached_tree(next_activations),
                "top_pred_zs": pred_zs.detach().index_select(0, top_positions).cpu(),
                "top_next_zs": next_zs.detach().index_select(0, top_positions).cpu(),
            }

            os.makedirs(self.numeric_artifact_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            filename = f"encoder_failure_step_{int(training_step)}_{stamp}.pt"
            path = os.path.abspath(os.path.join(self.numeric_artifact_dir, filename))
            temporary_path = path + ".tmp"
            torch.save(artifact, temporary_path)
            os.replace(temporary_path, path)
            print(f"[NUMERIC ARTIFACT] saved encoder failure artifact: {path}")
            return path
        except Exception as artifact_error:
            print(f"[NUMERIC ARTIFACT] save failed: {artifact_error}")
            return None

    def _hard_sync_targets(self, training_step):
        # Validate every source before the first copy so the four-way update is
        # all-or-nothing with respect to numerical integrity.
        self._guard_modules(
            "pre_target_sync",
            (
                ("actor.fixed_encoder", self.actor.fixed_encoder),
                ("actor.encoder", self.actor.encoder),
                ("actor.mlp", self.actor.mlp),
                ("critic", self.critic),
            ),
            training_step,
        )
        self.target_actor.fixed_encoder.load_state_dict(self.actor.fixed_encoder.state_dict())
        self.actor.fixed_encoder.load_state_dict(self.actor.encoder.state_dict())
        self.target_actor.mlp.load_state_dict(self.actor.mlp.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.buffer.reset_max_priority()
        self.target_min = self.value_min
        self.target_max = self.value_max

    def learn(self):
        if self.numeric_failure is not None:
            raise NumericalIntegrityError(
                "learner is locked after a previous numerical failure | "
                f"{self.numeric_failure}"
            )
        if self.buffer.size <= 0:
            self.total_losses = {}
            return

        stats = {}
        for _ in range(self.max_rollout):
            training_step = self.training_steps + 1
            try:
                batch = self.buffer.sample(self.config["batch_size"])
            except NumericalIntegrityError as error:
                self._latch_numeric_failure(error)
                raise
            states = batch["rail_feat"]
            actions = batch["action"]
            rewards = batch["reward"]
            next_states = batch["next_rail_feat"]
            dones = batch["done"]
            mask = batch["mask"]
            next_mask = batch["next_mask"]
            context = batch["context"]
            next_context = batch["next_context"]
            not_done = 1.0 - dones

            self._guard_tensors(
                "batch",
                (
                    ("rail_feat", states),
                    ("action", actions),
                    ("reward", rewards),
                    ("next_rail_feat", next_states),
                    ("done", dones),
                    ("context", context),
                    ("next_context", next_context),
                ),
                training_step,
            )

            with torch.no_grad():
                next_zs = self.actor.encoder.zs(next_states, next_context, next_mask)
            encoder_H, encoder_C, encoder_zs = self.actor.encoder.token_state(
                states,
                context,
                mask,
            )
            pred_zs = self.actor.encoder.zsa_from_encoded(
                encoder_H,
                encoder_C,
                encoder_zs,
                actions,
                mask,
            )
            self._guard_tensors(
                "encoder/forward",
                (("next_zs", next_zs), ("pred_zs", pred_zs)),
                training_step,
            )
            encoder_loss = F.mse_loss(pred_zs, next_zs)
            self._guard_tensors(
                "encoder/loss",
                (("encoder_loss", encoder_loss),),
                training_step,
            )

            self.encoder_optimizer.zero_grad(set_to_none=True)
            encoder_loss.backward()
            try:
                encoder_grad, encoder_parameter_grad_norms = _clip_grad_norm(
                    self.actor.encoder.named_parameters(),
                    self.encoder_grad_clip,
                    "encoder/gradient",
                    context=self._numeric_context(training_step),
                )
            except NumericalIntegrityError as error:
                encoder_parameter_grad_norms = self._best_effort_gradient_norms(
                    self.actor.encoder.named_parameters()
                )
                artifact_path = self._save_encoder_failure_artifact(
                    error,
                    training_step,
                    batch,
                    encoder_loss,
                    pred_zs,
                    next_zs,
                    encoder_parameter_grad_norms,
                )
                enriched = NumericalIntegrityError(
                    f"{error} | artifact={artifact_path or 'save_failed'}"
                )
                self._latch_numeric_failure(enriched)
                raise enriched from error

            if encoder_grad > self.encoder_grad_abort_norm:
                reason = (
                    "[encoder/gradient] FP64 raw total norm exceeded abort threshold "
                    f"raw_norm={encoder_grad:.9g}, "
                    f"threshold={self.encoder_grad_abort_norm:.9g} | "
                    f"{self._numeric_context(training_step)}"
                )
                artifact_path = self._save_encoder_failure_artifact(
                    reason,
                    training_step,
                    batch,
                    encoder_loss,
                    pred_zs,
                    next_zs,
                    encoder_parameter_grad_norms,
                )
                error = NumericalIntegrityError(
                    f"{reason} | artifact={artifact_path or 'save_failed'}"
                )
                self._latch_numeric_failure(error)
                raise error
            self.encoder_optimizer.step()
            self._guard_modules(
                "encoder/post_step",
                (("actor.encoder", self.actor.encoder),),
                training_step,
            )

            t_cf0 = time.perf_counter()
            with torch.no_grad():
                target_H, target_C, fixed_target_zs = self.target_actor.fixed_encoder.token_state(
                    next_states,
                    next_context,
                    next_mask,
                )
                next_action = self.target_actor.mlp(
                    target_H,
                    target_C,
                    fixed_target_zs,
                    next_mask,
                )
                noise = (torch.randn_like(next_action) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(self.action_low, self.action_high)
                next_action = next_action * next_mask.unsqueeze(-1).float()
                fixed_target_zsa = self.target_actor.fixed_encoder.zsa_from_encoded(
                    target_H,
                    target_C,
                    fixed_target_zs,
                    next_action,
                    next_mask,
                )
                q_next = self.target_critic(
                    target_H,
                    target_C,
                    fixed_target_zs,
                    fixed_target_zsa,
                    next_action,
                    next_mask,
                )
                q_next = q_next.min(1, keepdim=True)[0]
                target_q = rewards + not_done * self.gamma * q_next.clamp(self.target_min, self.target_max)
                fixed_H, fixed_C, fixed_zs = self.actor.fixed_encoder.token_state(states, context, mask)
                fixed_zsa = self.actor.fixed_encoder.zsa_from_encoded(
                    fixed_H,
                    fixed_C,
                    fixed_zs,
                    actions,
                    mask,
                )

            self._guard_tensors(
                "critic/target_forward",
                (
                    ("target_H", target_H),
                    ("target_C", target_C),
                    ("fixed_target_zs", fixed_target_zs),
                    ("next_action", next_action),
                    ("fixed_target_zsa", fixed_target_zsa),
                    ("q_next", q_next),
                    ("target_q", target_q),
                    ("fixed_H", fixed_H),
                    ("fixed_C", fixed_C),
                    ("fixed_zs", fixed_zs),
                    ("fixed_zsa", fixed_zsa),
                ),
                training_step,
            )
            self.value_max = max(self.value_max, float(target_q.max()))
            self.value_min = min(self.value_min, float(target_q.min()))

            q = self.critic(fixed_H, fixed_C, fixed_zs, fixed_zsa, actions, mask)
            critic_forward_ms = (time.perf_counter() - t_cf0) * 1000.0
            td_loss = (q - target_q).abs()
            critic_loss = F.smooth_l1_loss(q, target_q.expand_as(q), reduction="mean")
            self._guard_tensors(
                "critic/loss",
                (("q", q), ("td_loss", td_loss), ("critic_loss", critic_loss)),
                training_step,
            )

            t_cb0 = time.perf_counter()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad = self._safe_clip_grad_norm(
                "critic/gradient",
                self.critic.named_parameters(),
                self.critic_grad_clip,
                training_step,
            )
            self.critic_optimizer.step()
            self._guard_modules(
                "critic/post_step",
                (("critic", self.critic),),
                training_step,
            )
            critic_backward_ms = (time.perf_counter() - t_cb0) * 1000.0

            priority = td_loss.max(1)[0].detach()
            try:
                self.buffer.update_priority(priority)
            except NumericalIntegrityError as error:
                self._latch_numeric_failure(error)
                raise

            actor_loss_value = stats.get("loss/actor", 0.0)
            actor_pretanh_penalty_value = stats.get("loss/actor_pretanh_penalty", 0.0)
            actor_pretanh_valid_fraction = stats.get("actor/pretanh_valid_fraction", 0.0)
            actor_grad = stats.get("grad/actor_norm", 0.0)
            actor_forward_ms = 0.0
            if training_step % self.policy_freq == 0:
                t_af0 = time.perf_counter()
                with torch.no_grad():
                    actor_H, actor_C, actor_zs = self.actor.fixed_encoder.token_state(states, context, mask)
                # pretanh를 직접 계산해 (1) curriculum scale 적용, (2) penalty 계산에 재사용.
                # curriculum_scale: buffer에 저장된 action(raw * scale)과 분포를 맞춰
                # critic이 학습한 구간 밖 extrapolation으로 생기는 가짜 양(+) gradient를 제거.
                pretanh = self.actor.mlp._pre_tanh(actor_H, actor_C, actor_zs)
                new_action = (self.actor.max_action
                              * torch.tanh(pretanh)
                              * mask.unsqueeze(-1).float()
                              * self.curriculum_scale)
                new_zsa = self.actor.fixed_encoder.zsa_from_encoded(
                    actor_H,
                    actor_C,
                    actor_zs,
                    new_action,
                    mask,
                )
                q_actor = self.critic(actor_H, actor_C, actor_zs, new_zsa, new_action, mask)
                valid_pretanh = mask.unsqueeze(-1).expand_as(pretanh).to(pretanh.dtype)
                pretanh_penalty = self.pretanh_penalty_coef * _masked_mean_square(
                    pretanh, mask
                )
                actor_pretanh_valid_fraction = float(valid_pretanh.mean().detach().cpu())
                actor_loss = -q_actor.mean() + pretanh_penalty
                self._guard_tensors(
                    "actor/loss",
                    (
                        ("pretanh", pretanh),
                        ("new_action", new_action),
                        ("new_zsa", new_zsa),
                        ("q_actor", q_actor),
                        ("pretanh_penalty", pretanh_penalty),
                        ("actor_loss", actor_loss),
                    ),
                    training_step,
                )
                actor_forward_ms = (time.perf_counter() - t_af0) * 1000.0

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = self._safe_clip_grad_norm(
                    "actor/gradient",
                    self.actor.mlp.named_parameters(),
                    self.actor_grad_clip,
                    training_step,
                )
                self.actor_optimizer.step()
                self._guard_modules(
                    "actor/post_step",
                    (("actor.mlp", self.actor.mlp),),
                    training_step,
                )
                actor_loss_value = float(actor_loss.detach().cpu())
                actor_pretanh_penalty_value = float(pretanh_penalty.detach().cpu())

            q_det = q.detach()
            critic_diag = self.critic.diagnostics(
                fixed_H,
                fixed_C,
                fixed_zs,
                fixed_zsa,
                actions,
                mask,
            )
            if critic_diag:
                self._guard_tensors(
                    "critic/diagnostics",
                    ((name, torch.as_tensor(value)) for name, value in critic_diag.items()),
                    training_step,
                )

            if training_step % self.target_update_rate == 0:
                self._hard_sync_targets(training_step)

            stats = {
                **self.buffer.last_sample_info,
                "loss/critic": float(critic_loss.detach().cpu()),
                "loss/actor": float(actor_loss_value),
                "loss/actor_pretanh_penalty": float(actor_pretanh_penalty_value),
                "actor/pretanh_valid_fraction": float(actor_pretanh_valid_fraction),
                "loss/encoder": float(encoder_loss.detach().cpu()),
                "loss/q1": float(q_det[:, 0].mean().cpu()),
                "loss/q2": float(q_det[:, 1].mean().cpu()),
                "critic/q_mean": float(q_det.mean().cpu()),
                "critic/q_std": float(q_det.std(unbiased=False).cpu()),
                "critic/q_range": float((q_det.max() - q_det.min()).cpu()),
                "grad/actor_norm": float(actor_grad),
                "grad/critic_norm": float(critic_grad),
                "grad/encoder_norm": float(encoder_grad),
                "learner/critic_forward_ms": float(critic_forward_ms),
                "learner/critic_backward_ms": float(critic_backward_ms),
                "learner/actor_forward_ms": float(actor_forward_ms),
                **critic_diag,
            }
            self.training_steps = training_step

        self.total_losses = stats

    def set_curriculum_scale(self, scale: float):
        """Synchronize the current curriculum scale before learning."""
        self.curriculum_scale = float(scale)

    def get_params(self):
        params = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "training_steps": self.training_steps,
            "value_max": self.value_max,
            "value_min": self.value_min,
            "target_max": self.target_max,
            "target_min": self.target_min,
        }
        try:
            assert_finite_tree(
                "checkpoint/save",
                params,
                context=self._numeric_context(),
            )
        except NumericalIntegrityError as error:
            self._latch_numeric_failure(error)
            raise
        return params

    def load_params(self, params):
        bound_keys = ("value_max", "value_min", "target_max", "target_min")
        missing_bound_keys = [key for key in bound_keys if key not in params]
        try:
            assert_finite_tree(
                "checkpoint/load",
                params,
                context=f"current_training_step={self.training_steps}",
            )
        except NumericalIntegrityError as error:
            self._latch_numeric_failure(error)
            raise
        self.actor.load_state_dict(params["actor"])
        self.critic.load_state_dict(params["critic"])
        self.target_actor.load_state_dict(params["target_actor"])
        self.target_critic.load_state_dict(params["target_critic"])
        self.actor_optimizer.load_state_dict(params["actor_optimizer"])
        self.critic_optimizer.load_state_dict(params["critic_optimizer"])
        self.encoder_optimizer.load_state_dict(params["encoder_optimizer"])
        self.training_steps = int(params.get("training_steps", 0))
        self.value_max = float(params.get("value_max", self.value_max))
        self.value_min = float(params.get("value_min", self.value_min))
        self.target_max = float(params.get("target_max", self.target_max))
        self.target_min = float(params.get("target_min", self.target_min))
        self.numeric_failure = None
        if missing_bound_keys:
            print(
                "[Resume] legacy checkpoint has no TD7 value/target bounds "
                f"({', '.join(missing_bound_keys)}); constructor defaults remain until "
                f"the next target sync (at most {self.target_update_rate} learner updates)."
            )

    def probe_transition(self, rail_feat, action, context, mask):
        device = self.device
        rail = torch.as_tensor(rail_feat, dtype=torch.float32, device=device).unsqueeze(0)
        act = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
        ctx = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        if rail.shape[1] == 0:
            return {}
        with torch.no_grad():
            H, C, zs = self.actor.fixed_encoder.token_state(rail, ctx, m)
            zsa = self.actor.fixed_encoder.zsa(rail, act, ctx, m)
            q_base = self.critic(H, C, zs, zsa, act, m).mean()

            all_delta = (act + 0.1).clamp(self.action_low, self.action_high)
            zsa_all = self.actor.fixed_encoder.zsa(rail, all_delta, ctx, m)
            q_all = self.critic(H, C, zs, zsa_all, all_delta, m).mean()

            one_delta = act.clone()
            one_delta[:, 0:1, :] = (one_delta[:, 0:1, :] + 0.1).clamp(self.action_low, self.action_high)
            zsa_one = self.actor.fixed_encoder.zsa(rail, one_delta, ctx, m)
            q_one = self.critic(H, C, zs, zsa_one, one_delta, m).mean()

        return {
            "critic/q_one_region_delta": float((q_all - q_base).abs().detach().cpu()),
            "critic/q_one_token_delta": float((q_one - q_base).abs().detach().cpu()),
        }

    def probe_pretanh(self, rail_feat, context, mask):
        """영구 진단 probe: RegionPolicyHead의 tanh 직전 raw 값 분포.
        dead-ReLU(절대값 작은데도 토큰마다 출력이 똑같음) vs tanh saturation(절대값이
        크게 튀어서 포화) 구분용. 비용은 forward 1회뿐이라 매 진단 로그(100스텝)마다 호출."""
        device = self.device
        rail = torch.as_tensor(rail_feat, dtype=torch.float32, device=device).unsqueeze(0)
        ctx = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        if rail.shape[1] == 0:
            return {}
        with torch.no_grad():
            pre_tanh = self.actor.pre_tanh_action(rail, ctx, m)
            valid = pre_tanh[m.unsqueeze(-1).expand_as(pre_tanh)]
            if valid.numel() == 0:
                return {}

            # 입력 크기(abs mean)는 '기여'를 보장하지 않음 — 첫 Linear의 가중치를 거친
            # 블록별 부분합으로 H/C/zs 중 실제로 pre-tanh를 키우는 항을 가린다.
            contributions = self.actor.pre_tanh_contrib_action(rail, ctx, m)
            values = {}
            for name, contribution in contributions.items():
                valid_mask = m.unsqueeze(-1).expand_as(contribution)
                values[name] = contribution[valid_mask].abs().mean()
            state_denom = (values["state_H"] + values["state_C"]).clamp(min=1e-8)
            joint_denom = (values["joint_state"] + values["joint_zs"]).clamp(min=1e-8)
        return {
            "actor/pretanh_abs_mean": float(valid.abs().mean().detach().cpu()),
            "actor/pretanh_std": float(valid.std(unbiased=False).detach().cpu()),
            "actor/pretanh_max_abs": float(valid.abs().max().detach().cpu()),
            "actor/state_proj_contrib_H": float(values["state_H"].detach().cpu()),
            "actor/state_proj_contrib_C": float(values["state_C"].detach().cpu()),
            "actor/state_proj_contrib_C_ratio": float(
                (values["state_C"] / state_denom).detach().cpu()
            ),
            "actor/joint_contrib_state": float(values["joint_state"].detach().cpu()),
            "actor/joint_contrib_zs": float(values["joint_zs"].detach().cpu()),
            "actor/joint_contrib_zs_ratio": float(
                (values["joint_zs"] / joint_denom).detach().cpu()
            ),
        }

    def probe_weight_norms(self):
        """Track state-projection and post-AvgL1Norm joint weight blocks."""
        return self.actor.policy_head_weight_norms()

    def probe_layer_scale(self, rail_feat, context, mask):
        """Track every policy stage and the restored AvgL1Norm boundary."""
        device = self.device
        rail = torch.as_tensor(rail_feat, dtype=torch.float32, device=device).unsqueeze(0)
        ctx = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        if rail.shape[1] == 0:
            return {}
        with torch.no_grad():
            return self.actor.layer_scale_probe_action(rail, ctx, m)

    def probe_bias_norms(self):
        """진단용: 입력 데이터 없이 각 Linear의 bias/weight 절댓값 평균만 본다.
        pretanh_std가 pretanh_abs_mean에 비해 매우 작으면(모든 rail이 거의 같은 큰 값),
        그 공통 오프셋이 마지막 Linear의 bias에서 오는지 weight 쪽에서 오는지 가른다."""
        return self.actor.policy_head_bias_norms()
