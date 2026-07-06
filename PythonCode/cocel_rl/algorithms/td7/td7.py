"""TD7 algorithm (TD3 + SALE encoder + LAP + checkpoint-style fixed targets).

learner.py 가 넘겨주는 인자:
  train(buffer, critic_optimizer, critic, target_critic,
        policy_optimizer(=actor.mlp 전용), policy(=actor), target_policy(=target_actor),
        iteration, encoder_optimizer(=actor.encoder 전용))

인코더 3종:
  policy.encoder            : 매 스텝 학습 (SALE loss)
  policy.fixed_encoder      : 현재 Q/정책용 스냅샷  (target_update_rate 마다 encoder 로 갱신)
  target_policy.fixed_encoder: 타깃 Q용 더 오래된 스냅샷 (그 직전 fixed_encoder 로 갱신)
"""
import torch
import torch.nn.functional as F

from cocel_rl.algorithms.base import BaseAlgorithm
from cocel_rl.buffers.off_policy_buffer import LAP
from cocel_rl.utils.utils import LAP_huber
from .network import Actor, Critic


class TD7(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = 'TD7'
        self.type = 'off_policy'
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.action_bound = env.action_bound

        algo = config['algorithm']
        zs_dim = int(algo.get('zs_dim', 256))
        hdim = int(algo.get('hdim', 256))
        expl = float(algo.get('noise_scale', 0.1))

        self.actor = Actor(self.obs_dim, self.act_dim, self.action_bound, zs_dim, hdim, expl)
        self.critic = Critic(self.obs_dim, self.act_dim, zs_dim, hdim)

        self.max_action = float(max(abs(self.action_bound[0]), abs(self.action_bound[1])))
        self.buffer = LAP(self.obs_dim, self.act_dim, device=config['device'],
                          capacity=int(config['buffer_capacity']),
                          normalize_action=True, max_action=self.max_action,
                          prioritized=True)

        self.log_alpha = None
        self.alpha_optimizer = None
        self.tau = config['tau']
        self.training_steps = 0

        # TD7 하이퍼파라미터
        self.gamma = config['gamma']
        self.target_update_rate = int(algo.get('target_update_rate', 250))
        self.policy_freq = int(algo.get('policy_update_delay', 2))
        self.target_noise = float(algo.get('target_noise_scale', 0.2))
        self.noise_clip = float(algo.get('target_noise_clip', 0.5))
        self.lap_alpha = float(algo.get('lap_alpha', 0.4))
        self.min_priority = float(algo.get('min_priority', 1.0))

        # 가치 타깃 클리핑(TD7 안정화): 관측된 타깃 Q 범위로 부트스트랩을 제한
        self.value_max = -1e8
        self.value_min = 1e8
        self.target_max = 0.0
        self.target_min = 0.0

        self._stats = {}
        super().__init__()

    def train(self, buffer, critic_optimizer, critic, target_critic, policy_optimizer, policy, target_policy, iteration, encoder_optimizer):
        for _ in range(iteration):
            self.training_steps += 1

            states, actions, rewards, next_states, dones = buffer.sample(self.config['batch_size'])
            not_done = 1.0 - dones

            # ------------------------------------------------------------------ #
            # 1) SALE 인코더 학습:  g(f(s), a) ≈ f(s')                            #
            # ------------------------------------------------------------------ #
            with torch.no_grad():
                next_zs = policy.encoder.zs(next_states)
            zs = policy.encoder.zs(states)
            pred_zs = policy.encoder.zsa(zs, actions)
            encoder_loss = F.mse_loss(pred_zs, next_zs)

            encoder_optimizer.zero_grad()
            encoder_loss.backward()
            encoder_optimizer.step()

            # ------------------------------------------------------------------ #
            # 2) Critic 학습 (LAP Huber)                                          #
            # ------------------------------------------------------------------ #
            with torch.no_grad():
                fixed_target_zs = target_policy.fixed_encoder.zs(next_states)
                noise = (torch.randn_like(actions) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (target_policy.mlp(next_states, fixed_target_zs) + noise).clamp(
                    self.action_bound[0], self.action_bound[1])
                fixed_target_zsa = target_policy.fixed_encoder.zsa(fixed_target_zs, next_action)

                q_next = target_critic(next_states, next_action, fixed_target_zsa, fixed_target_zs)
                q_next = q_next.min(1, keepdim=True)[0]
                target_q = rewards + not_done * self.gamma * q_next.clamp(self.target_min, self.target_max)

                # 관측된 타깃 Q 범위 갱신
                self.value_max = max(self.value_max, float(target_q.max()))
                self.value_min = min(self.value_min, float(target_q.min()))

                fixed_zs = policy.fixed_encoder.zs(states)
                fixed_zsa = policy.fixed_encoder.zsa(fixed_zs, actions)

            q = critic(states, actions, fixed_zsa, fixed_zs)
            td_loss = (q - target_q).abs()
            critic_loss = LAP_huber(td_loss, self.min_priority)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ------------------------------------------------------------------ #
            # 3) LAP 우선순위 갱신                                                 #
            # ------------------------------------------------------------------ #
            priority = td_loss.max(1)[0].clamp(min=self.min_priority).pow(self.lap_alpha)
            buffer.update_priority(priority)

            # ------------------------------------------------------------------ #
            # 4) Actor 학습 (policy_freq 마다, mlp 만 갱신)                        #
            # ------------------------------------------------------------------ #
            if self.training_steps % self.policy_freq == 0:
                with torch.no_grad():
                    actor_zs = policy.fixed_encoder.zs(states)
                new_action = policy.mlp(states, actor_zs)
                new_zsa = policy.fixed_encoder.zsa(actor_zs, new_action)
                q_actor = critic(states, new_action, new_zsa, actor_zs)
                actor_loss = -q_actor.mean()

                policy_optimizer.zero_grad()
                actor_loss.backward()
                policy_optimizer.step()
                self._stats['actor'] = float(actor_loss.detach().cpu())

            # ------------------------------------------------------------------ #
            # 5) 주기적 하드 타깃 업데이트 + 인코더 스냅샷 롤오버                    #
            # ------------------------------------------------------------------ #
            if self.training_steps % self.target_update_rate == 0:
                # fixed_encoder_target <- (직전) fixed_encoder
                target_policy.fixed_encoder.load_state_dict(policy.fixed_encoder.state_dict())
                # fixed_encoder <- 최신 학습 encoder
                policy.fixed_encoder.load_state_dict(policy.encoder.state_dict())
                # actor mlp / critic 하드 카피
                target_policy.mlp.load_state_dict(policy.mlp.state_dict())
                target_critic.load_state_dict(critic.state_dict())
                # LAP / 가치 클리핑 범위 롤오버
                buffer.reset_max_priority()
                self.target_min = self.value_min
                self.target_max = self.value_max

            self._stats.update({
                'critic': float(critic_loss.detach().cpu()),
                'encoder': float(encoder_loss.detach().cpu()),
                'q1': float(q[:, 0].mean().detach().cpu()),
                'q2': float(q[:, 1].mean().detach().cpu()),
            })

        return self._stats
