import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import utils
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}



class SymmetricWalker2dEnv(Walker2dEnv):
    def __init__(
        self, 
        period = 1.0,
        frame_skip = 5,
        forward_reward_weight = 1, 
        ctrl_cost_weight = 0.001, 
        healthy_reward = 1, 
        terminate_when_unhealthy = True, 
        healthy_z_range = (0.8, 2.0), 
        healthy_angle_range = (-1.0, 1.0), 
        reset_noise_scale = 0.005, 
        exclude_current_positions_from_observation = True, 
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        '''
            Observation space: jpos + jvel + phase
        '''
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(17 + 1,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18 + 1,), dtype=np.float64
            )

        self.metadata["render_fps"] = int(1.0 / (0.002 * frame_skip))
        MujocoEnv.__init__(
            self,
            "walker2d.xml",
            frame_skip = frame_skip, # self.dt = 0.002 * frame_skip
            observation_space = observation_space,
            default_camera_config = DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Phase information
        self.period = period
        self.T = 0.0 # Total running time (truncated to 4 digits)

        # Setup 
        self.symmetrical_state_indices()
        self.symmetrical_action_indices()


    def symmetrical_state_indices(self):
        self.right_state_indices = np.array([2, 3, 4, 11, 12, 13])
        self.left_state_indices = np.array([5, 6, 7, 14, 15, 16])

        if not self._exclude_current_positions_from_observation:
            # Only x-position is excluded.
            self.right_state_indices += 1
            self.left_state_indices += 1


    def symmetrical_action_indices(self):
        self.right_action_indices = np.array([0, 1, 2])
        self.left_action_indices = np.array([3, 4, 5])


    def reset_model(self):
        self.T = 0.0
        return super().reset_model()


    def _get_phase(self):
        return np.round(np.remainder(self.T, self.period), 4)


    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        phase = self._get_phase()

        observation = np.concatenate((
            position, 
            velocity, 
            [phase]
        )).ravel()
        return observation


    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "T" : self.T,
            "phase" : observation[-1]
        }

        if self.render_mode == "human":
            self.render()

        self.T = np.round(self.T + self.dt, 4)

        return observation, reward, terminated, False, info



if __name__ == '__main__':
    env = SymmetricWalker2dEnv()
    env.reset()

    for i in range(1000):
        _, _, _, _, info = env.step(env.action_space.sample())
        print(i, info['T'], info['phase'])