import gymnasium as gym    


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

    gym.envs.register(
        id = 'SymmetricHumanoidEnv-v0',
        entry_point = 'customized_envs.SymmetricHumanoidEnv:SymmetricHumanoidEnv',
        max_episode_steps = 1000
    )



if __name__ == '__main__':
    register_customized_envs()
    x = gym.make('SymmetricHumanoidEnv-v0')
    x.reset()