import torch
import copy
from sac import SACPolicy
from data import Batch
from network import GaussianActorNet, CriticNet


class SymmetricSACPolicy(SACPolicy):
    def __init__(
        self, 
        obs_mirror_func,
        act_mirror_func,
        symmetry_loss_weight,
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
        action_bounding_func = '',
        device = 'cuda'
    ) -> None:
        super().__init__(
            actor, actor_optim,
            critic1, critic1_optim, critic2, critic2_optim,
            alpha, log_alpha_optimizer_lr, 
            target_entropy, gamma, tau,
            observation_space, action_space, action_scaling, action_bounding_func,
            device
        )

        self.symmetry_loss_weight = symmetry_loss_weight
        self.obs_mirror_func = obs_mirror_func
        self.act_mirror_func = act_mirror_func


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
        result_log['actor_loss'] = actor_loss.item()

        # Symmetry loss for actor
        obs = batch.obs
        act_mu, act_std = self.actor.forward(obs)
        mirror_obs = self.obs_mirror_func(obs)

        # # 6.4 Symmetry loss 1: || M_a(\pi(s)) - \pi( M_s(s) ) ||^2_2
        # sym_act_mu, sym_act_std = self.actor.forward(mirror_obs)
        # with torch.no_grad():
        #     sym_act_mu_gt = self.act_mirror_func(act_mu)
        #     sym_act_std_gt = self.act_mirror_func(act_std)
        # symmetry_mu_loss = self.symmetry_loss_weight * torch.mean((sym_act_mu - sym_act_mu_gt)**2)
        # result_log['symmetry_mu_loss'] = symmetry_mu_loss.item()
        # symmetry_std_loss = self.symmetry_loss_weight * torch.mean((sym_act_std - sym_act_std_gt)**2)
        # result_log['symmetry_std_loss'] = symmetry_std_loss.item()
        # total_actor_loss = actor_loss + symmetry_mu_loss + symmetry_std_loss
        # result_log['total_actor_loss'] = total_actor_loss.item()

        # 6.7 Symmetry loss 2: || \pi(s) - M_a( \pi( M_s(s) ) ) ||^2_2
        # This requires the mirroring function to not modify the leaf variable in-place.
        sym_act_mu, sym_act_std = self.actor.forward(mirror_obs)
        rev_sym_act_mu = self.act_mirror_func(sym_act_mu)
        rev_sym_act_std = self.act_mirror_func(sym_act_std)
        symmetry_mu_loss = self.symmetry_loss_weight * torch.mean((act_mu - rev_sym_act_mu)**2)
        result_log['symmetry_mu_loss'] = symmetry_mu_loss.item()
        symmetry_std_loss = self.symmetry_loss_weight * torch.mean((act_std - rev_sym_act_std)**2)
        result_log['symmetry_std_loss'] = symmetry_std_loss.item()
        total_actor_loss = actor_loss + symmetry_mu_loss + symmetry_std_loss
        result_log['total_actor_loss'] = total_actor_loss.item()

        self.actor_optim.zero_grad()
        total_actor_loss.backward()
        self.actor_optim.step()
        


        # Alpha
        if self.is_auto_alpha:
            alpha_loss = (self.log_alpha.exp() * -(new_act_log_prob.detach() + self.target_entropy)).mean()
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp().item()
            result_log['alpha_loss'] = alpha_loss.item()
            result_log['alpha'] = self.alpha
            result_log['log_alpha'] = self.log_alpha.detach().item()

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