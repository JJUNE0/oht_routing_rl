import torch
from cocel_rl.algorithms.base import BaseAlgorithm
from cocel_rl.buffers.off_policy_buffer import OffPolicyBuffer
from cocel_rl.utils.utils import soft_update
from .network import Critic, Actor


class TD3(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = 'TD3'
        self.type = 'off_policy'
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.action_bound = env.action_bound

        self.actor = Actor(self.obs_dim, self.act_dim, env.action_bound, config['algorithm']['noise_scale'],
                                 config['actor_hidden_dims'], config['activation_fc'])

        self.critic = Critic(self.obs_dim, self.act_dim, config['critic_hidden_dims'], config['activation_fc'])
        
        self.buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['buffer_capacity']))

        self.log_alpha = None
        self.alpha_optimizer = None

        self.tau = config['tau']
        
        self.training_steps = 0

        super().__init__()

    def train(self, buffer, critic_optimizer, critic, target_critic, policy_optimizer, policy, target_policy, iteration, encoder_optimizer):
        for _ in range(iteration):
            self.training_steps += 1
            states, actions, rewards, next_states, dones = buffer.sample(self.config['batch_size'], device=self.config['device'])
            # Calculate the Critic loss
            with torch.no_grad():
                target_act_noise = (torch.randn_like(actions) * self.config['algorithm']['target_noise_scale']).clamp(
                    -self.config['algorithm']['target_noise_clip'], self.config['algorithm']['target_noise_clip']).to(self.config['device'])
                next_target_action = (target_policy(next_states) + target_act_noise).clamp(self.action_bound[0],
                                                                                           self.action_bound[1])
                next_q_values_1, next_q_values_2 = target_critic(next_states, next_target_action)
                next_q_values = torch.min(next_q_values_1, next_q_values_2)
                target_q_values = rewards + (1 - dones) * self.config['gamma'] * next_q_values

            q_values_1, q_values_2 = critic(states, actions)
            critic_loss = ((q_values_1 - target_q_values) ** 2).mean() + ((q_values_2 - target_q_values) ** 2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # Calculate the Actor Loss
            actor_loss = -critic.Q_A(states, policy(states)).mean()
            
            # twin-delay
            if self.training_steps % self.config['algorithm']['policy_update_delay'] == 0:
                
                policy_optimizer.zero_grad()
                actor_loss.backward()
                policy_optimizer.step()
                soft_update(policy, target_policy, self.tau)
                soft_update(critic, target_critic, self.tau)

        return {
                    "critic": float(critic_loss.detach().cpu()),
                    "actor": float(actor_loss.detach().cpu()),
                    "q1": float(q_values_1.mean().detach().cpu()),
                    "q2": float(q_values_2.mean().detach().cpu()),
                
                }