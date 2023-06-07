import numpy as np
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


'''
    Comparing to the original Humanoid environment in Gymnasium, ReducedObsSpaceHumanoidEnv only includes
    the joint positions/velocities into the observation space.
'''
class ReducedObsSpaceHumanoidEnv(HumanoidEnv):
    '''
        Only include the joint states in the observation space
    '''
    
    def __init__(
        self, 
        frame_skip = 5,
        forward_reward_weight = 1.25, 
        ctrl_cost_weight = 0.1, 
        healthy_reward = 5, 
        terminate_when_unhealthy = True, 
        healthy_z_range = (1.0, 2.0), 
        reset_noise_scale = 0.01,
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
            exclude_current_positions_from_observation = True,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64
        )

        self.frame_skip = frame_skip
        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            frame_skip = frame_skip,
            observation_space=observation_space,
            default_camera_config = DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )


    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        position = position[2:]
        velocity = self.data.qvel.flat.copy()

        return np.concatenate(
            (
                position,
                velocity
            )
        )
    

if __name__ == '__main__':
    env = ReducedObsSpaceHumanoidEnv()
    print(env.observation_space, env.action_space)