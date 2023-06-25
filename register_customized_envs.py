import gymnasium as gym    
from customized_envs.SymmetricHumanoidEnv_v0 import SymmetricHumanoidEnv_v0
from customized_envs.base_symmetry_env import BaseSymmetryEnv


def register_symmetric_env(env_id, env_class, obs_indices_dict, act_indices_dict, max_episode_steps = 1000):
    def make_symmetric_env(*args, **kwargs):
        return BaseSymmetryEnv(
            env = env_class(*args, **kwargs),
            obs_indices_dict = obs_indices_dict,
            act_indices_dict = act_indices_dict
        )
    
    gym.envs.register(
        id = env_id,
        entry_point = make_symmetric_env,
        max_episode_steps = max_episode_steps
    )


def register_customized_envs():
    gym.envs.register(
        id = 'ReducedObsSpaceHumanoidEnv-v0',
        entry_point = 'customized_envs.ReducedObsSpaceHumanoidEnv:ReducedObsSpaceHumanoidEnv',
        max_episode_steps = 1000
    )

    gym.envs.register(
        id = 'SymmetricWalker2dEnv-v0',
        entry_point = 'customized_envs.SymmetricWalker2dEnv:SymmetricWalker2dEnv',
        max_episode_steps = 1000
    )

    register_symmetric_env(
        env_id = 'SymmetricHumanoidEnv-v0',
        env_class = SymmetricHumanoidEnv_v0,
        obs_indices_dict = {
            'common_indices' : [0, 1, 3, 6, 22, 24, 26, 29],
            'negated_indices' : [2, 4, 5, 7, 23, 25, 27, 28, 30],
            'right_indices' : [8, 9, 10, 11, 16, 17, 18, 31, 32, 33, 34, 39, 40, 41, 45],
            'left_indices' : [12, 13, 14, 15, 19, 20, 21, 35, 36, 37, 38, 42, 43, 44, 46]
        },
        act_indices_dict = {
            'common_indices' : [0],
            'negated_indices' : [1, 2],
            'right_indices' : [3, 4, 5, 6, 11, 12, 13],
            'left_indices' : [7, 8, 9, 10, 14, 15, 16]
        },
        max_episode_steps = 1000
    )



if __name__ == '__main__':
    register_customized_envs()
    x = gym.make('SymmetricHumanoidEnv-v0')
    obs, _ = x.reset()
    print(obs)
    print(x.obs_mirror_func(obs))