import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


class ReducedObsSpaceHumanoidEnv(HumanoidEnv):
    '''
        Only include the joint states in the observation space
    '''
    def __init__(
        self, 
        forward_reward_weight = 1.25, 
        ctrl_cost_weight = 0.1, 
        healthy_reward = 5, 
        terminate_when_unhealthy = True, 
        healthy_z_range = (1.0, 2.0), 
        reset_noise_scale = 0.01, 
        exclude_current_positions_from_observation = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(47,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            5,
            observation_space=observation_space,
            default_camera_config = DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )


    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity
            )
        )
    


def register_custom_mujocoenvs():
    gym.envs.register(
        id = 'ReducedObsSpaceHumanoidEnv-v0',
        entry_point = 'modified_mujoco_envs:ReducedObsSpaceHumanoidEnv',
        max_episode_steps = 1000
    )



if __name__ == '__main__':
    register_custom_mujocoenvs()
    x = gym.make('ReducedObsSpaceHumanoidEnv-v0')
    x.reset()
    print(x.observation_space.shape)
    next_obs, reward, done, truncated, info = x.step(x.action_space.sample())
    print(next_obs.shape)