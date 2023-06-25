import torch
import torch.nn as nn


'''
    Helper functions
'''
def create_mlp(in_dim, out_dim, hidden_dims, activation_fn = nn.ReLU, last_activation_fn = None):
    layer_dims = [in_dim] + hidden_dims
    layers = []

    for curr_dim, next_dim in zip(layer_dims[:-1], layer_dims[1:]):
        layers.append(nn.Linear(curr_dim, next_dim))
        if activation_fn:
            layers.append(activation_fn())

    layers.append(nn.Linear(layer_dims[-1], out_dim))
    if last_activation_fn:
        layers.append(last_activation_fn())

    return nn.Sequential(*layers)



def truncated_normal_init(t, mean = 0.0, std = 0.01):
    torch.nn.init.normal_(t, mean = mean, std = std)
    while True:
        cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
        if not torch.sum(cond):
            break
        t = torch.where(cond, torch.nn.init.normal_(torch.ones_like(t), mean = mean, std = std), t)
    return t



def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            input_dim = m.in_features
            truncated_normal_init(m.weight, std = 1 / (2 * input_dim ** 0.5))
            torch.nn.init.zeros_(m.bias.data)



'''
    Swish activation function
'''
class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)



'''
    Actor Network
'''
class GaussianActorNet(nn.Module):
    def __init__(
        self, 
        obs_dim, 
        act_dim, 
        hidden_dims = [256, 256], 
        min_logstd = -20, 
        max_logstd = 2
    ) -> None:
        super().__init__()

        self.mu_net = create_mlp(obs_dim, act_dim, hidden_dims, nn.ReLU, last_activation_fn = nn.Tanh)
        self.logstd_net = create_mlp(obs_dim, act_dim, hidden_dims, nn.ReLU)
        self.logstd_clamp_func = lambda logstd: torch.clamp(logstd, min_logstd, max_logstd)


    def forward(self, obs):
        mu = self.mu_net(obs)
        logstd = self.logstd_net(obs)
        std = self.logstd_clamp_func(logstd).exp()
        return mu, std
    


'''
    Symmetric policy
'''
class PhaseNet(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        phase_dim,
        hidden_dims = [256, 256],
        activation_fn = nn.ReLU,
        last_activation_fn = None
    ) -> None:
        super().__init__()

        layer_dims = [obs_dim] + hidden_dims
        layers = []

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim + phase_dim, out_dim))
            if activation_fn:
                layers.append(activation_fn())

        layers.append(nn.Linear(layer_dims[-1] + phase_dim, act_dim))
        if last_activation_fn:
            layers.append(last_activation_fn())

        self.layers = nn.Sequential(*layers)


    def forward(self, x, phase):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(torch.cat([x, phase], -1))
            else:
                x = layer(x)
        return x



class SymmetricPhaseNet(nn.Module):
    def __init__(
        self,
        center_obs_inds,
        right_obs_inds,
        left_obs_inds,
        right_phase_inds,
        left_phase_inds,
        center_act_inds,
        right_act_inds,
        left_act_inds,
        hidden_dims = [128, 128],
        min_logstd = -20,
        max_logstd = 2,
    ) -> None:
        '''
            Try to attach the phase variable to every hidden layer
        '''
        super().__init__()

        center_obs_dim = len(center_obs_inds)
        assert len(left_obs_inds) == len(right_obs_inds)
        side_obs_dim = len(left_obs_inds)
        assert len(left_phase_inds) == len(right_phase_inds)
        phase_dim = len(left_phase_inds)

        center_act_dim = len(center_act_inds)
        assert len(left_act_inds) == len(right_act_inds)
        side_act_dim = len(left_act_inds)

        self.center_obs_inds = center_obs_inds
        self.right_obs_inds = right_obs_inds
        self.left_obs_inds = left_obs_inds
        self.left_phase_inds = left_phase_inds
        self.right_phase_inds = right_phase_inds
        self.center_act_inds = center_act_inds
        self.right_act_inds = right_act_inds
        self.left_act_inds = left_act_inds
        
        # Map the actor output back to the action space
        act_inds = center_act_inds + right_act_inds + left_act_inds
        self.act_inverse_mapping_inds = [0 for _ in act_inds]
        for e, idx in enumerate(act_inds):
            self.act_inverse_mapping_inds[idx] = e

        # Network
        self.logstd_clamp_func = lambda logstd: torch.clamp(logstd, min_logstd, max_logstd)
        self.center_mu_net = create_mlp(center_obs_dim, center_act_dim, hidden_dims, nn.ReLU, nn.Tanh)
        self.center_logstd_net = create_mlp(center_obs_dim, center_act_dim, hidden_dims, nn.ReLU)
        self.side_mu_net = PhaseNet(side_obs_dim + center_obs_dim, side_act_dim, phase_dim, hidden_dims, nn.ReLU, nn.Tanh)
        self.side_logstd_net = PhaseNet(side_obs_dim + center_obs_dim, side_act_dim, phase_dim, hidden_dims, nn.ReLU)


    def forward(self, obs):
        center_obs = obs[..., self.center_obs_inds]
        right_obs = obs[..., self.right_obs_inds]
        left_obs = obs[..., self.left_obs_inds]
        right_phase = obs[..., self.right_phase_inds]
        left_phase = obs[..., self.left_phase_inds]

        act_center_mu = self.center_mu_net(center_obs)
        act_center_std = self.logstd_clamp_func(self.center_logstd_net(center_obs)).exp()

        act_right_mu = self.side_mu_net(torch.cat([right_obs, center_obs], -1), right_phase)
        act_right_logstd = self.side_logstd_net(torch.cat([right_obs, center_obs], -1), right_phase)
        act_right_std = self.logstd_clamp_func(act_right_logstd).exp()

        act_left_mu = self.side_mu_net(torch.cat([left_obs, center_obs], -1), left_phase)
        act_left_logstd = self.side_logstd_net(torch.cat([left_obs, center_obs], -1), left_phase)
        act_left_std = self.logstd_clamp_func(act_left_logstd).exp()

        act_mu = torch.cat([act_center_mu, act_right_mu, act_left_mu], -1)
        act_mu = act_mu[..., self.act_inverse_mapping_inds]
        act_std = torch.cat([act_center_std, act_right_std, act_left_std], -1)
        act_std = act_std[..., self.act_inverse_mapping_inds]

        return act_mu, act_std



'''
    Critic Network
'''
class CriticNet(nn.Module):
    def __init__(
        self, 
        obs_dim, 
        act_dim = 0, 
        hidden_dims = [256, 256]
    ) -> None:
        super().__init__()
        self.q_net = create_mlp(obs_dim + act_dim, 1, hidden_dims, nn.ReLU)


    def forward(self, obs, act = None):
        inputs = obs if act is None else torch.cat([obs, act], -1)
        q = self.q_net(inputs).squeeze(-1)
        return q
    

if __name__ == '__main__':
    from customized_envs.SymmetricHumanoidEnv_v0 import SymmetricHumanoidEnv_inds
    env_space_indices = SymmetricHumanoidEnv_inds
    actor = SymmetricPhaseNet(
        center_obs_inds = env_space_indices['center_obs'],
        right_obs_inds = env_space_indices['right_obs'],
        left_obs_inds = env_space_indices['left_obs'],
        right_phase_inds = env_space_indices['right_phase'],
        left_phase_inds = env_space_indices['left_phase'],
        center_act_inds = env_space_indices['center_act'],
        right_act_inds = env_space_indices['right_act'],
        left_act_inds = env_space_indices['left_act'],
        hidden_dims = [200, 200],
        min_logstd = -20, 
        max_logstd = 2
    )
    obs = torch.randn(47)
    mu, std = actor.forward(obs)
    print(mu.size(), std.size())