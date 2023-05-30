import gymnasium as gym
import numpy as np
import pickle
import cloudpickle
from multiprocessing import Pipe, Process

'''
    Reference: 
    1. https://squadrick.dev/journal/efficient-multi-gym-environments.html
    2. https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/vec_env/subproc_vec_env.html#SubprocVecEnv
'''
GYM_RESERVED_KEYS = [
    "metadata", "reward_range", "spec", "action_space", "observation_space", "_max_episode_steps", 'model'
]


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x
    
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
        
    def __setstate__(self, ob):
        self.x = pickle.loads(ob)
        
    def __call__(self, **params):
        return self.x(**params)


def worker(remote, parent_remote, env_fn, params):
    parent_remote.close()
    env = env_fn(**params)
    episode_reward = 0
    episode_step = 0

    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            obs, reward, done, _, info = env.step(data)
            episode_reward += reward
            episode_step += 1
            if episode_step >= env._max_episode_steps:
                done = True
            if done:
                obs, _ = env.reset()
                info['episode_reward'] = episode_reward
                info['episode_step'] = episode_step
                episode_reward = 0
                episode_step = 0
            info['terminate'] = done
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            obs, _ = env.reset(**data)
            episode_reward = 0
            episode_step = 0
            remote.send(obs)
        elif cmd == 'render':
            remote.send(env.render(mode = data))
        elif cmd == 'sample_action':
            remote.send(env.action_space.sample())
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif hasattr(env, cmd):
            attr = getattr(env, cmd)
            if callable(attr):
                remote.send(attr(**data))  
            else:
                remote.send(attr)
        else:
            raise NotImplementedError("`{}` is not implemented in the worker.".format(cmd))


class SubprocVecEnv(object):
    def __init__(self, env_fn_list, params_list = None) -> None:
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fn_list)
        if params_list is None:
            params_list = [{} for _ in range(self.n_envs)]
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])

        self.ps = []
        for wrk, rem, fn, params in zip(self.work_remotes, self.remotes, env_fn_list, params_list):
            process = Process(target = worker, args = (wrk, rem, CloudpickleWrapper(fn), params))
            process.daemon = True
            process.start()
            self.ps.append(process)
            wrk.close()

        # for i in range(self.n_envs):
        #     for key in GYM_RESERVED_KEYS[:-1]: # Except "model"
        #         self.remotes[i].send((key, None))
        #         setattr(self, key, self.remotes[i].recv())


    def get_env_attribute(self, env_idx, attr_name):
        self.remotes[env_idx].send((attr_name, None))
        return self.remotes[env_idx].recv()


    def set_env_attribute(self, env_idx, attr_name, val):
        self.remotes[env_idx].send(('set' + attr_name, val))


    def get_env_func(self, env_idx, func_name, **params):
        self.remotes[env_idx].send((func_name, params))
        return self.remotes[env_idx].recv()


    def sample_actions(self):
        for remote in self.remotes:
            remote.send(('sample_action', None))
        return np.stack([remote.recv() for remote in self.remotes])


    def step_async(self, actions):
        if self.waiting:
            raise TypeError
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    
    def step_wait(self):
        if not self.waiting:
            raise TypeError
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones), np.stack(infos)


    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


    def reset(self, **params):
        for remote in self.remotes:
            remote.send(('reset', params))
        return np.stack([remote.recv() for remote in self.remotes])


    # def render(self, mode):
    #     for remote in self.remotes:
    #         remote.send(('render', mode))
    #     return np.stack([remote.recv() for remote in self.remotes])


    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


    @property
    def observation_space(self):
        self.remotes[0].send(('observation_space', None))
        return self.remotes[0].recv() 
    

    @property
    def action_space(self):
        self.remotes[0].send(('action_space', None))
        return self.remotes[0].recv() 
    

    @property
    def _max_episode_steps(self):
        self.remotes[0].send(('_max_episode_steps', None))
        return self.remotes[0].recv() 



# All the environments have the same parameters.
def make_mp_envs(env_id = '', n_envs = 4):
    def fn():
        env = gym.make(env_id, disable_env_checker = True)
        return env
    return SubprocVecEnv([fn for _ in range(n_envs)])


# The environments have different parameters.
def make_mp_diffenvs(env_id, params_list = [{}]):
    def fn(**kwargs):
        env = gym.make(env_id, disable_env_checker = True, **kwargs)
        return env
    return SubprocVecEnv([fn for _ in range(len(params_list))], params_list)



if __name__ == '__main__':
    env = make_mp_diffenvs('HalfCheetah-v4', [{} for _ in range(2)])
    env.reset()
    done = False
    i = 0
    while not np.all(done):
        _, _, done, info = env.step(env.sample_actions())
        i += 1
    print(i, info)