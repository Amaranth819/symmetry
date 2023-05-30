import torch
import torch.nn as nn
import copy
import math
from basepolicy import BasePolicy
from data import Batch
from network import GaussianActorNet, CriticNet


class SACPolicy(BasePolicy):
    def __init__(
        self, 
        actor : GaussianActorNet,
        actor_optim : torch.optim.Optimizer,
        critic1 : CriticNet,
        critic1_optim : torch.optim.Optimizer,
        critic2 : CriticNet,
        critic2_optim : torch.optim.Optimizer,
        alpha = 0.2,
        log_alpha_optimizer_lr = None,
        target_entropy = -3, 
        gamma = 0.99, 
        tau = 0.005,
        observation_space = None,
        action_space = None,
        action_scaling = False,
        action_bounding_func = 'tanh',
        device = 'cuda'
    ) -> None:
        super().__init__(observation_space, action_space, action_scaling, action_bounding_func, device)

        # Actor and critic
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_optim = critic1, critic1_optim
        self.critic2, self.critic2_optim = critic2, critic2_optim

        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_target.eval()
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_target.eval()

        # Alpha
        self.alpha = alpha
        if log_alpha_optimizer_lr is None:
            self.is_auto_alpha = False
            self.log_alpha = nn.Parameter(torch.ones(1).float() * math.log(self.alpha), requires_grad = False)
            self.log_alpha_optimizer = None
        else:
            self.is_auto_alpha = True
            self.log_alpha = nn.Parameter(torch.ones(1).float() * math.log(self.alpha), requires_grad = True)
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = log_alpha_optimizer_lr)

        # Other parameters
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau

        # Device
        self = self.to(torch.device(self.device_str))


    def train(self, mode : bool):
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self
    

    def _update_critic(self, batch : Batch, critic : CriticNet, critic_optimizer : torch.optim.Optimizer):
        obs, act, rew, done = batch.obs, batch.act, batch.rew, batch.done

        # Compute returns
        target_q = self._target_q(batch)
        returns = rew + (1.0 - done) * self.gamma * target_q

        # Predict returns
        q = critic.forward(obs, act)
        critic_loss = (q - returns).pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        return critic_loss
    

    def _target_q(self, batch : Batch):
        next_obs = batch.next_obs
        with torch.no_grad():
            next_obs_act, next_obs_act_log_prob = self.forward(next_obs)
            target_q1 = self.critic1_target.forward(next_obs, next_obs_act)
            target_q2 = self.critic2_target.forward(next_obs, next_obs_act)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_obs_act_log_prob
            return target_q
    

    def forward(self, obs, deterministic = False):
        obs = super().forward(obs)
        mu, std = self.actor.forward(obs)
        dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
        act = mu if deterministic else dist.rsample()
        log_prob = dist.log_prob(act)

        # Squash action to [-1, 1]
        eps = 1e-4
        squashed_act = torch.tanh(act)
        squashed_log_prob = log_prob - torch.sum(torch.log(1 - squashed_act.pow(2) + eps), -1)
        
        return squashed_act, squashed_log_prob


    def learn(self, batch: Batch):
        # Summary
        result_log = {}

        # Critic
        critic1_loss = self._update_critic(batch, self.critic1, self.critic1_optim)
        result_log['critic1_loss'] = critic1_loss.item()
        critic2_loss = self._update_critic(batch, self.critic2, self.critic2_optim)
        result_log['critic2_loss'] = critic2_loss.item()

        # Actor 
        obs = batch.obs
        new_act, new_act_log_prob = self.forward(obs, False)
        current_q1 = self.critic1.forward(obs, new_act)
        current_q2 = self.critic2.forward(obs, new_act)
        actor_loss = (self.alpha * new_act_log_prob - torch.min(current_q1, current_q2)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        result_log['actor_loss'] = actor_loss.item()

        # Alpha
        if self.is_auto_alpha:
            alpha_loss = (self.log_alpha.exp() * -(new_act_log_prob.detach() + self.target_entropy)).mean()
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp().item()
            result_log['alpha_loss'] = alpha_loss.item()

        # Soft update for critics
        self.soft_update(self.critic1_target, self.critic1, self.tau)
        self.soft_update(self.critic2_target, self.critic2, self.tau)

        return result_log
    

    def save(self, path):
        sd = {}
        sd['actor'] = self.actor.state_dict()
        sd['critic1'] = self.critic1.state_dict()
        sd['critic1_optimizer'] = self.critic1_optim.state_dict()
        sd['critic2'] = self.critic2.state_dict()
        sd['critic2_optimizer'] = self.critic2_optim.state_dict()
        sd['log_alpha'] = self.log_alpha
        if self.is_auto_alpha:
            sd['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()
        torch.save(sd, path)
        print(f'Save {type(self).__name__} to path {path}!')


    def load(self, path):
        sd = torch.load(path)
        self.actor.load_state_dict(sd['actor'])
        self.critic1.load_state_dict(sd['critic1'])
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_target.eval()
        self.critic2.load_state_dict(sd['critic2'])
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_target.eval()
        self.log_alpha = sd['log_alpha']
        self.alpha = self.log_alpha.detach().exp().item()
        if self.is_auto_alpha:
            self.log_alpha_optimizer.load_state_dict(sd['log_alpha_optimizer'])
        print(f'Load {type(self).__name__} from path {path}!')



if __name__ == '__main__':
    obs_dim = 4
    act_dim = 2
    hidden_dims = [16, 16]
    actor = GaussianActorNet(obs_dim, act_dim, hidden_dims)
    actor_optim = torch.optim.Adam(actor.parameters(), lr = 0.1)
    critic1 = CriticNet(obs_dim, act_dim, hidden_dims)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr = 0.1)
    critic2 = CriticNet(obs_dim, act_dim, hidden_dims)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr = 0.1)

    sac = SACPolicy(actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, device = 'cuda')

    obs = torch.ones((3, obs_dim))
    print(sac.forward(obs))