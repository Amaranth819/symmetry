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


def obs_mirror_func(obs):
    # center, negated, right, left
    mirror_indices = [
        0, 1, 2, 3, 4, 5, 6, 7, 
        12, 13, 14, 15, 8, 9, 10, 11, 
        19, 20, 21, 16, 17, 18,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 
        35, 36, 37, 38, 31, 32, 33, 34,
        42, 43, 44, 39, 40, 41,
        45, 46,
        48, 47
    ]
    return obs[..., mirror_indices]


def act_mirror_func(act):
    # right, left -> left, right
    mirror_indices = [
        0, 1, 2,
        7, 8, 9, 10, 3, 4, 5, 6,
        14, 15, 16, 11, 12, 13
    ]
    return act[..., mirror_indices]



SymmetricHumanoidEnv_inds = {
    'center_obs' : [0, 1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    'right_obs' : [8, 9, 10, 11, 16, 17, 18, 31, 32, 33, 34, 39, 40, 41],
    'left_obs' : [12, 13, 14, 15, 19, 20, 21, 35, 36, 37, 38, 42, 43, 44],
    'right_phase' : [47],
    'left_phase' : [48],
    'center_act' : [0, 1, 2],
    'right_act' : [3, 4, 5, 6, 11, 12, 13],
    'left_act' : [7, 8, 9, 10, 14, 15, 16]
}



class SymmetricHumanoidEnv(HumanoidEnv):
    def __init__(
        self, 
        period = 1.0,
        frame_skip = 5, 
        forward_reward_weight = 1.25, 
        ctrl_cost_weight = 0.1, 
        healthy_reward = 1.0, # Set to 1.0 here, different to the default value in HumanoidEnv
        terminate_when_unhealthy = True, 
        healthy_z_range = (1.0, 2.0),
        reset_noise_scale = 0.01,
        desired_velocity = [0.5, 0.0], # [x, y]
        desired_torso_height = 1.0,
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
            Observation space: jpos + jvel + desired velocity (2) + phase (2) 
        '''
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(49,), dtype=np.float64
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
        self.T = 0.0 # Total running time (rounding to 4 digits)

        # Periodicity
        self.epsilon = 0.1
        self.theta_rf = 0.0
        self.theta_lf = 0.5 # Half stride
        self.velocity_coef_func = lambda phase: np.sin(phase) / np.sqrt(np.sin(phase)**2 + self.epsilon**2)  
        self.grf_coef_func = lambda phase: np.sin(phase + np.pi) / np.sqrt(np.sin(phase + np.pi)**2 + self.epsilon**2)  

        # Desired velocity
        self.desired_velocity = desired_velocity


    def reset_model(self):
        self.T = 0.0
        return super().reset_model()
    

    def _get_phase(self):
        phi = np.round(np.remainder(self.T, self.period), 4) # phi in [0, 1)
        rf_phase = (phi + self.theta_rf) * 2 * np.pi
        rf_phase_sin = np.sin(rf_phase)
        lf_phase = (phi + self.theta_lf) * 2 * np.pi
        lf_phase_sin = np.sin(lf_phase)
        return (rf_phase, lf_phase), (rf_phase_sin, lf_phase_sin)
    

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        position = position[2:]
        velocity = self.data.qvel.flat.copy()
        (rf_phase, lf_phase), (rf_phase_sin, lf_phase_sin) = self._get_phase()

        observation = np.concatenate((
            position, 
            velocity, 
            self.desired_velocity,
            [rf_phase_sin, lf_phase_sin]
        )).ravel()
        return observation
    

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        (rf_phase, lf_phase), _ = self._get_phase()
        rf_ext_force, lf_ext_force = self._get_foot_contact_force()
        rf_linvel, lf_linvel = self._get_foot_velocity()

        observation = self._get_obs()

        # Rewards
        reward = 0.0
        info = {}

        # Alive bonus
        healthy_reward = self.healthy_reward
        info['healthy_reward'] = healthy_reward
        reward += healthy_reward

        forward_vel_reward = np.exp(-2 * np.abs(x_velocity - self.desired_velocity[0]))
        info['forward_vel_reward'] = forward_vel_reward
        reward += forward_vel_reward

        lateral_vel_reward = np.exp(-5 * np.abs(y_velocity - self.desired_velocity[1]))
        info['lateral_vel_reward'] = lateral_vel_reward
        reward += lateral_vel_reward

        rf_vel_reward_coef = self.velocity_coef_func(rf_phase)
        if rf_vel_reward_coef >= 0:
            # Encourage foot swing
            rf_vel_reward = 1 - np.exp(-2 * rf_vel_reward_coef * rf_linvel)
        else:
            # Punish foot swing
            rf_vel_reward = np.exp(2 * rf_vel_reward_coef * rf_linvel)
        info['rf_vel_reward'] = rf_vel_reward
        reward += rf_vel_reward

        lf_vel_reward_coef = self.velocity_coef_func(lf_phase)
        if lf_vel_reward_coef >= 0:
            # Encourage foot swing
            lf_vel_reward = 1 - np.exp(-2 * lf_vel_reward_coef * lf_linvel)
        else:
            # Punish foot swing
            lf_vel_reward = np.exp(2 * lf_vel_reward_coef * lf_linvel)
        info['lf_vel_reward'] = lf_vel_reward
        reward += lf_vel_reward

        rf_grf_reward_coef = self.grf_coef_func(rf_phase)
        if rf_grf_reward_coef < 0:
            # Encourage foot swing
            rf_grf_reward = np.exp(0.01 * rf_grf_reward_coef * rf_ext_force)
        else:
            # Punish foot swing
            rf_grf_reward = 1 - np.exp(-0.01 * rf_grf_reward_coef * rf_ext_force)
        info['rf_grf_reward'] = rf_grf_reward
        reward += rf_grf_reward

        lf_grf_reward_coef = self.grf_coef_func(lf_phase)
        if lf_grf_reward_coef < 0:
            # Encourage foot swing
            lf_grf_reward = np.exp(0.01 * lf_grf_reward_coef * lf_ext_force)
        else:
            # Punish foot swing
            lf_grf_reward = 1 - np.exp(-0.01 * lf_grf_reward_coef * lf_ext_force)
        info['lf_grf_reward'] = lf_grf_reward
        reward += lf_grf_reward

        ctrl_cost = -self.control_cost(action)
        info['ctrl_cost'] = ctrl_cost
        reward += ctrl_cost

        terminated = self.terminated
        
        if self.render_mode == "human":
            self.render()

        self.T = np.round(self.T + self.dt, 4)

        return observation, reward, terminated, False, info


    
    def _get_foot_contact_force(self):
        # # Get the body names
        # nbody = env.model.nbody
        # for n in range(nbody):
        #     print(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, n))
        
        # Foot names: right_foot (idx 6), left_foot (idx 9)
        # cfrc_ext: rotation-translation format
        rf_ext_force = np.linalg.norm(self.data.cfrc_ext[6][3:])
        lf_ext_force = np.linalg.norm(self.data.cfrc_ext[9][3:])
        return rf_ext_force, lf_ext_force
    

    def _get_foot_velocity(self):
        # [3D rot; 3D tran] 
        # Currently only consider the linear velocities
        rf_linvel = np.linalg.norm(self.data.cvel[6][3:])
        lf_linvel = np.linalg.norm(self.data.cvel[9][3:])
        return rf_linvel, lf_linvel


if __name__ == '__main__':
    env = SymmetricHumanoidEnv()
    env.reset()

    from collections import defaultdict
    log = defaultdict(lambda: [])
    for _ in range(1000):
        _, _, _, _, info = env.step(env.action_space.sample())
        log['healthy_reward'].append(info['healthy_reward'])
        log['forward_vel_reward'].append(info['forward_vel_reward'])
        log['lateral_vel_reward'].append(info['lateral_vel_reward'])
        log['rf_vel_reward'].append(info['rf_vel_reward'])
        log['lf_vel_reward'].append(info['lf_vel_reward'])
        log['rf_grf_reward'].append(info['rf_grf_reward'])
        log['lf_grf_reward'].append(info['lf_grf_reward'])
        log['ctrl_cost'].append(info['ctrl_cost'])
    
    import matplotlib.pyplot as plt
    xs = np.arange(1000)
    for key, val_list in log.items():
        plt.plot(xs, val_list, label = key)
    plt.legend()
    plt.savefig('temp.png')
    plt.close()