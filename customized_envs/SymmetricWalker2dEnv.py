import copy
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


def obs_mirror_func(obs):
    # Joint states
    right = [2, 3, 4, 11, 12, 13]
    left = [5, 6, 7, 14, 15, 16]
    mirror_obs = copy.copy(obs)
    mirror_obs[..., right], mirror_obs[..., left] = obs[..., left], obs[..., right]
    if obs.size(-1) == 19:
        # Phase variable
        mirror_obs[..., -2], mirror_obs[..., -1] = mirror_obs[..., -1], mirror_obs[..., -2]
    return mirror_obs


def act_mirror_func(act):
    right = [0, 1, 2]
    left = [3, 4, 5]
    mirror_act = copy.copy(act)
    mirror_act[..., right], mirror_act[..., left] = act[..., left], act[..., right]
    return mirror_act


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
        include_phase_into_obs_space = True,
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
            exclude_current_positions_from_observation = True,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        '''
            Observation space: jpos + jvel (+ phase * 2) 
        '''
        if include_phase_into_obs_space:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
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
        self.include_phase_into_obs_space = include_phase_into_obs_space


    def reset_model(self):
        self.T = 0.0
        return super().reset_model()


    def _get_phase(self):
        phi = np.round(np.remainder(self.T, self.period), 4) * np.pi
        right_phi = np.sin(phi)
        left_phi = np.cos(phi)
        return phi, [right_phi, left_phi]


    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        position = position[1:]
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)
        phi, phases = self._get_phase()

        if self.include_phase_into_obs_space:
            observation = np.concatenate((
                position, 
                velocity, 
                phases
            )).ravel()
        else:
            observation = np.concatenate((
                position, 
                velocity, 
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
        phi, phases = self._get_phase()
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "T" : self.T,
            "phi" : phi,
            "phases" : phases
        }

        if self.render_mode == "human":
            self.render()

        self.T = np.round(self.T + self.dt, 4)

        return observation, reward, terminated, False, info



if __name__ == '__main__':
    env = SymmetricWalker2dEnv(include_phase_into_obs_space = True)
    print(env.observation_space, env.action_space)