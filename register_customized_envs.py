import gymnasium as gym    


def register_customized_envs():
    gym.envs.register(
        id = 'ReducedObsSpaceHumanoidEnv-v0',
        entry_point = 'customized_envs.ReducedObsSpaceHumanoidEnv:ReducedObsSpaceHumanoidEnv',
        max_episode_steps = 1000
    )



if __name__ == '__main__':
    register_customized_envs()
    x = gym.make('ReducedObsSpaceHumanoidEnv-v0')
    x.reset()
    print(x.observation_space.shape)
    next_obs, reward, done, truncated, info = x.step(x.action_space.sample())
    print(next_obs.shape)