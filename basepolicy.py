'''
    Reference: 
    1. https://tianshou.readthedocs.io/en/master/_modules/tianshou/policy/base.html#BasePolicy
'''

import torch
import torch.nn as nn
import numpy as np
from data import Batch, BaseBuffer
from gym.spaces import Box


class BasePolicy(nn.Module):
    def __init__(
        self,
        observation_space = None,
        action_space = None,
        action_scaling = False,
        action_bounding_func = 'tanh',
        device = 'cuda'
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        # Whether scale action from [-1,1] to [action_space.low, action_space.high]
        self.action_scaling = action_scaling
        assert action_bounding_func in ('', 'tanh', 'clip')
        self.action_bounding_func = action_bounding_func

        # Device
        assert device in ('auto', 'cpu', 'cuda')
        if device == 'auto':
            self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device_str = device


    def exploration_action(self, act):
        return act
    

    def soft_update(self, target, source, tau):
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.copy_(source_param * tau + target_param * (1.0 - tau))


    def hard_update(self, target, source):
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.copy_(source_param)
    

    def map_action(self, act : np.ndarray):
        if isinstance(self.action_space, Box):
            if self.action_bounding_func == 'clip':
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bounding_func == 'tanh':
                act = np.tanh(act)
            if self.action_scaling:
                assert np.min(act) >= -1 and np.max(act) <= 1
                low = self.action_space.low
                high = self.action_space.high
                act = low + (act + 1.0) * (high - low) / 2.0
        return act
    

    def map_action_inverse(self, act : np.ndarray, eps = 1e-4):
        if isinstance(self.action_space, Box):
            if self.action_scaling:
                low = self.action_space.low
                high = self.action_space.high
                scale = high - low
                scale[scale < eps] += eps
                act = 2.0 * (act - low) / scale - 1.0
            if self.action_bounding_func == 'tanh':
                act = (np.log(1.0 + act) - np.log(1.0 - act)) / 2.0
        return act
    

    def forward(self, obs, deterministic = False, **kwargs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(torch.device(self.device_str))
        elif isinstance(obs, torch.Tensor):
            obs = obs.float().to(torch.device(self.device_str))
        return obs
    

    def learn(self, batch : Batch, **kwargs):
        raise NotImplementedError
    

    def preprocess_fn(self, batch : Batch):
        batch.to_torch(self.device_str)
        batch.float()
        return batch
    

    def update(self, sample_size : int, buffer : BaseBuffer, **kwargs):
        batch = buffer.sample(sample_size)
        batch = self.preprocess_fn(batch)
        result = self.learn(batch, **kwargs)
        return result


    def save(self, path):
        raise NotImplementedError


    def load(self, path):
        raise NotImplementedError