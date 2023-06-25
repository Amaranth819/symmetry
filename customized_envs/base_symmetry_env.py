import numpy as np
import torch
import gymnasium as gym


class BaseSymmetryEnv(gym.Wrapper):
    def __init__(self, env, obs_indices_dict, act_indices_dict) -> None:
        super().__init__(env)
        self.obs_indices_dict = obs_indices_dict
        self.act_indices_dict = act_indices_dict
        self.obs_mirrored_indices = self._get_mirrored_indices(self.obs_indices_dict)
        self.act_mirrored_indices = self._get_mirrored_indices(self.act_indices_dict)
    

    def _get_mirrored_indices(self, indices_dict):
        indices = indices_dict['common_indices'] + indices_dict['negated_indices'] + indices_dict['right_indices'] + indices_dict['left_indices']
        if len(indices) == 0:
            raise NotImplementedError('Need to define mirror indices first!')
        mirrored_indices = [0 for _ in range(len(indices))]
        for i, ind in enumerate(indices):
            mirrored_indices[ind] = i
        return mirrored_indices
    

    def _mirror_func(self, x, indices_dict, mirrored_indices):
        # For the quaternion orientation in observation, [x,y,z,w] -> [x,-y,z,-w] since [roll,pitch,yaw] -> [-roll,pitch,-yaw].
        common_x = x[..., indices_dict['common_indices']]
        negated_x = x[..., indices_dict['negated_indices']]
        right_x = x[..., indices_dict['right_indices']]
        left_x = x[..., indices_dict['left_indices']]
        
        if isinstance(x, np.ndarray):
            mirror_x = np.concatenate([common_x, -negated_x, left_x, right_x], -1)
        elif isinstance(x, torch.Tensor):
            mirror_x = torch.cat([common_x, -negated_x, left_x, right_x], -1)
        else:
            raise TypeError
        
        return mirror_x[..., mirrored_indices]


    def act_mirror_func(self, act):
        return self._mirror_func(act, self.act_indices_dict, self.act_mirrored_indices)
    

    def obs_mirror_func(self, obs):
        return self._mirror_func(obs, self.obs_indices_dict, self.obs_mirrored_indices)
    

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    

    def step(self, action):
        return self.env.step(action)
    

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    


if __name__ == '__main__':
    obs = torch.randn(8, requires_grad = True)
    net = torch.nn.Linear(8, 3)
    act = net(obs)

    obs_indices_dict = {
        'common_indices' : [0, 1],
        'negated_indices' : [],
        'right_indices' : [2, 3, 6],
        'left_indices' : [4, 5, 7]
    }
    act_indices_dict = {
        'common_indices' : [0],
        'negated_indices' : [],
        'right_indices' : [1],
        'left_indices' : [2]
    }
    t = BaseSymmetryEnv(None, obs_indices_dict, act_indices_dict)
    print(obs)
    print(t.obs_mirror_func(obs))
    print(act)
    print(t.act_mirror_func(act))