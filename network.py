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