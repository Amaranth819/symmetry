import copy
import numpy as np
from gymnasium.spaces import Box
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class SymmetricHumanoidEnv_v0(HumanoidEnv):
    def __init__(
        self, 
        period = 1.0,
        frame_skip = 5, 
        forward_reward_weight = 1.25, 
        ctrl_cost_weight = 0.1, 
        healthy_reward = 5, 
        terminate_when_unhealthy = True, 
        healthy_z_range = (1.0, 2.0),
        reset_noise_scale = 0.01,
        **kwargs
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

        '''
            Observation space: jpos + jvel (+ 2 phases ) 
        '''
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float64
        )

        self.frame_skip = frame_skip
        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            frame_skip = frame_skip, # self.dt = 0.003 * frame_skip, see <option:timestep> in humanoid.xml
            observation_space = observation_space,
            default_camera_config = DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Phase information
        self.period = period
        self.T = 0.0 # Total running time (truncated to 4 digits)


    def reset_model(self):
        self.T = 0.0
        return super().reset_model()
    

    def _get_phase(self):
        theta_right, theta_left = 0.0, 0.5 # Phase offset
        phi = np.round(np.remainder(self.T, self.period), 4) # phi in [0, 1)
        phase_right = np.sin((phi + theta_right) * 2 * np.pi)
        phase_left = np.sin((phi + theta_left) * 2 * np.pi)
        return phi, [phase_right, phase_left]
    

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        position = position[2:]
        velocity = self.data.qvel.flat.copy()
        phi, phases = self._get_phase()

        observation = np.concatenate((
            position, 
            velocity, 
            phases
        )).ravel()
        return observation
    

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - ctrl_cost
        terminated = self.terminated
        phi, phases = self._get_phase()
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "T" : self.T,
            "phi" : phi,
            "phases" : phases
        }

        if self.render_mode == "human":
            self.render()

        self.T = np.round(self.T + self.dt, 4)

        return observation, reward, terminated, False, info
    


if __name__ == '__main__':
    # env = SymmetricHumanoidEnv()
    # obs, _ = env.reset()
    # print(obs)
    # print(env.obs_mirror_func(obs))
    # for i in range(1000):
    #     _, _, _, _, info = env.step(env.action_space.sample())
    #     print(i, info["T"], info["phases"])
    pass