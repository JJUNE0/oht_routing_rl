import copy
import time

import torch
import torch.nn.functional as F

from .networks import RegionTD7Actor, RegionTD7Critic
from .replay_buffer import RegionReplayBuffer


def _clip_grad_norm(parameters, max_norm):
    """clip_grad_norm_는 클리핑과 동시에 클리핑 전(raw) total norm을 돌려주므로,
    grad/*_norm 진단 로그는 그대로 두면서 폭발(NaN/Inf)만 차단할 수 있다."""
    parameters = list(parameters)
    if not parameters:
        return 0.0
    total = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return float(total)


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

        # curriculum scale이 buffer에 저장된 action과 actor loss action의 분포를 정렬한다.
        # ClientAlgorithm_region이 매 learn() 호출 전 set_curriculum_scale()로 갱신.
        self.curriculum_scale = 1.0
        self.pretanh_penalty_coef = float(algo.get("pretanh_penalty_coef", 0.1))

    def learn(self):
        if self.buffer.size <= 0:
            self.total_losses = {}
            return

        stats = {}
        for _ in range(self.max_rollout):
            self.training_steps += 1
            batch = self.buffer.sample(self.config["batch_size"])
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

            with torch.no_grad():
                next_zs = self.actor.encoder.zs(next_states, next_context, next_mask)
            zs = self.actor.encoder.zs(states, context, mask)
            pred_zs = self.actor.encoder.zsa(states, actions, context, mask)
            encoder_loss = F.mse_loss(pred_zs, next_zs)

            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            encoder_grad = _clip_grad_norm(self.actor.encoder.parameters(), self.encoder_grad_clip)
            self.encoder_optimizer.step()

            t_cf0 = time.perf_counter()
            with torch.no_grad():
                target_H, target_C, fixed_target_zs = self.target_actor.fixed_encoder.token_state(
                    next_states,
                    next_context,
                    next_mask,
                )
                next_action = self.target_actor.policy_action(next_states, next_context, next_mask)
                noise = (torch.randn_like(next_action) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(self.action_low, self.action_high)
                next_action = next_action * next_mask.unsqueeze(-1).float()
                fixed_target_zsa = self.target_actor.fixed_encoder.zsa(
                    next_states,
                    next_action,
                    next_context,
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
                self.value_max = max(self.value_max, float(target_q.max()))
                self.value_min = min(self.value_min, float(target_q.min()))
                fixed_H, fixed_C, fixed_zs = self.actor.fixed_encoder.token_state(states, context, mask)
                fixed_zsa = self.actor.fixed_encoder.zsa(states, actions, context, mask)

            q = self.critic(fixed_H, fixed_C, fixed_zs, fixed_zsa, actions, mask)
            critic_forward_ms = (time.perf_counter() - t_cf0) * 1000.0
            td_loss = (q - target_q).abs()
            critic_loss = F.smooth_l1_loss(q, target_q.expand_as(q), reduction="mean")

            t_cb0 = time.perf_counter()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad = _clip_grad_norm(self.critic.parameters(), self.critic_grad_clip)
            self.critic_optimizer.step()
            critic_backward_ms = (time.perf_counter() - t_cb0) * 1000.0

            priority = td_loss.max(1)[0].detach()
            self.buffer.update_priority(priority)

            actor_loss_value = stats.get("loss/actor", 0.0)
            actor_pretanh_penalty_value = stats.get("loss/actor_pretanh_penalty", 0.0)
            actor_grad = stats.get("grad/actor_norm", 0.0)
            actor_forward_ms = 0.0
            if self.training_steps % self.policy_freq == 0:
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
                new_zsa = self.actor.fixed_encoder.zsa(states, new_action, context, mask)
                q_actor = self.critic(actor_H, actor_C, actor_zs, new_zsa, new_action, mask)
                pretanh_penalty = self.pretanh_penalty_coef * pretanh.pow(2).mean()
                actor_loss = -q_actor.mean() + pretanh_penalty
                actor_forward_ms = (time.perf_counter() - t_af0) * 1000.0

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad = _clip_grad_norm(self.actor.mlp.parameters(), self.actor_grad_clip)
                self.actor_optimizer.step()
                actor_loss_value = float(actor_loss.detach().cpu())
                actor_pretanh_penalty_value = float(pretanh_penalty.detach().cpu())

            if self.training_steps % self.target_update_rate == 0:
                self.target_actor.fixed_encoder.load_state_dict(self.actor.fixed_encoder.state_dict())
                self.actor.fixed_encoder.load_state_dict(self.actor.encoder.state_dict())
                self.target_actor.mlp.load_state_dict(self.actor.mlp.state_dict())
                self.target_critic.load_state_dict(self.critic.state_dict())
                self.buffer.reset_max_priority()
                self.target_min = self.value_min
                self.target_max = self.value_max

            q_det = q.detach()
            critic_diag = self.critic.diagnostics(
                fixed_H,
                fixed_C,
                fixed_zs,
                fixed_zsa,
                actions,
                mask,
            )
            stats = {
                **self.buffer.last_sample_info,
                "loss/critic": float(critic_loss.detach().cpu()),
                "loss/actor": float(actor_loss_value),
                "loss/actor_pretanh_penalty": float(actor_pretanh_penalty_value),
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

        self.total_losses = stats

    def set_curriculum_scale(self, scale: float):
        """ClientAlgorithm_region이 매 learn() 전 현재 curriculum scale을 동기화."""
        self.curriculum_scale = float(scale)

    def get_params(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "training_steps": self.training_steps,
        }

    def load_params(self, params):
        self.actor.load_state_dict(params["actor"])
        self.critic.load_state_dict(params["critic"])
        self.target_actor.load_state_dict(params["target_actor"])
        self.target_critic.load_state_dict(params["target_critic"])
        self.actor_optimizer.load_state_dict(params["actor_optimizer"])
        self.critic_optimizer.load_state_dict(params["critic_optimizer"])
        self.encoder_optimizer.load_state_dict(params["encoder_optimizer"])
        self.training_steps = int(params.get("training_steps", 0))

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
            contrib_H, contrib_C, contrib_zs = self.actor.pre_tanh_contrib_action(rail, ctx, m)
            valid_mask = m.unsqueeze(-1).expand_as(contrib_H)
            abs_H = contrib_H[valid_mask].abs().mean()
            abs_C = contrib_C[valid_mask].abs().mean()
            abs_zs = contrib_zs[valid_mask].abs().mean()
            denom = (abs_H + abs_C + abs_zs).clamp(min=1e-8)
        return {
            "actor/pretanh_abs_mean": float(valid.abs().mean().detach().cpu()),
            "actor/pretanh_std": float(valid.std(unbiased=False).detach().cpu()),
            "actor/pretanh_max_abs": float(valid.abs().max().detach().cpu()),
            "actor/pretanh_contrib_H": float(abs_H.detach().cpu()),
            "actor/pretanh_contrib_C": float(abs_C.detach().cpu()),
            "actor/pretanh_contrib_zs": float(abs_zs.detach().cpu()),
            "actor/pretanh_contrib_C_ratio": float((abs_C / denom).detach().cpu()),
        }

    def probe_weight_norms(self):
        """진단용: 입력 데이터 없이 RegionPolicyHead 첫 Linear의 H/C/zs 가중치 블록 norm만 본다.
        contrib_zs가 압도적인데 zs 입력 자체는 avg_l1_norm으로 묶여있다면, 범인이 입력이 아니라
        이 가중치 블록(특히 zs)이 학습 중 커진 것인지 직접 확인하기 위함."""
        return self.actor.policy_head_weight_norms()

    def probe_layer_scale(self, rail_feat, context, mask):
        """영구 진단 probe: net 내부 5개 stage(fc0/relu0/fc1/relu1/fc2=pre-tanh)의
        abs_mean을 순서대로 찍는다. pretanh_contrib_H+C+zs(첫 Linear 직후, 보통 <1.5)와
        pretanh_abs_mean(최종, 보통 8~14) 사이의 격차가 정확히 어느 stage에서 벌어지는지
        핀포인트하기 위함 — 입력단 정규화가 아니라 중간 레이어가 범인인지 직접 확인."""
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
