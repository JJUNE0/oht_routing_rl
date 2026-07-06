import copy
import torch

class Learner:
    def __init__(self, algorithm,  config):
        self.config = config

        self.algorithm = copy.deepcopy(algorithm)
        
        self.buffer_type = self.algorithm.buffer.type
        self.buffer = self.algorithm.buffer

        self.actor = self.algorithm.actor
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = self.algorithm.critic
        self.target_critic = copy.deepcopy(self.critic)

        self.actor = self.actor.to(config['device'])
        self.target_actor = self.target_actor.to(config['device'])
        self.critic = self.critic.to(config['device'])
        self.target_critic = self.target_critic.to(config['device'])

        # Set up optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'],eps=config['adam_eps'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'], eps=config['adam_eps'])
        self.encoder_optimizer = None

        if self.algorithm.name == 'TD7':
            self.actor_optimizer = torch.optim.Adam(self.actor.mlp.parameters(), lr=config['actor_lr'], eps=config['adam_eps'])
            self.encoder_optimizer = torch.optim.Adam(self.actor.encoder.parameters(), lr=config['algorithm']['encoder_lr'])

        if self.algorithm.name == 'TD7' and self.config['device'] == 'cuda':
            self.buffer.load_cuda()

        self.max_data_size = int(config['max_rollout'])

        self.epochs = 0
        self.total_steps = 0

        self.total_losses = {}
        

    def get_params(self):
        if self.encoder_optimizer is not None:
            encoder_optimizer_state_dict = self.encoder_optimizer.state_dict()
        else:
            encoder_optimizer_state_dict = None

        if self.algorithm.alpha_optimizer is not None and self.algorithm.log_alpha is not None:
            log_alpha = self.algorithm.log_alpha.item()
            alpha_optimizer_state_dict = self.algorithm.alpha_optimizer.state_dict()
        else:
            log_alpha = None
            alpha_optimizer_state_dict = None

        return (self.actor, self.critic, self.actor_optimizer.state_dict(), self.critic_optimizer.state_dict(),
                encoder_optimizer_state_dict, log_alpha, alpha_optimizer_state_dict, self.buffer)


    def learn(self):
        stats = self.algorithm.train(self.buffer, self.critic_optimizer, self.critic, self.target_critic,
                                  self.actor_optimizer, self.actor, self.target_actor,
                                  iteration=self.max_data_size, encoder_optimizer=self.encoder_optimizer)

        if isinstance(stats, dict):
            self.total_losses = stats
        else:
            self.total_losses = {}





