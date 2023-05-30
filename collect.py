import gym
import numpy as np
import torch
from data import Batch
from mpenv import SubprocVecEnv
from typing import Union
from basepolicy import BasePolicy
from utils import DataListLogger
from gym.wrappers.monitoring.video_recorder import VideoRecorder


def collect_from_env(
    env: Union[SubprocVecEnv, gym.Env],
    policy : BasePolicy = None,
    n_steps : int = None,
    is_eval : bool = False
):
    if n_steps is None:
        n_steps = env._max_episode_steps

    obs_list = []
    next_obs_list = []
    act_list = []
    rew_list = []
    done_list = []

    obs = env.reset()
    episode_log = DataListLogger()

    for _ in range(n_steps):
        if policy is None:
            act = env.sample_actions() if isinstance(env, SubprocVecEnv) else env.action_space.sample()
        else:
            with torch.no_grad():
                act, _ = policy.forward(obs, deterministic = is_eval)
                act = act.cpu().numpy()
                act = policy.map_action(act)

        next_obs, reward, done, infos = env.step(act)
        
        obs_list.append(obs)
        next_obs_list.append(next_obs)
        act_list.append(act)
        rew_list.append(reward)
        done_list.append(done)

        for info in infos:
            if info['terminate']:
                episode_log.add('episode_reward', info['episode_reward'])
                episode_log.add('episode_step', info['episode_step'])

        obs = np.copy(next_obs)

    batch = Batch(
        obs = np.concatenate(obs_list),
        next_obs = np.concatenate(next_obs_list),
        act = np.concatenate(act_list),
        rew = np.concatenate(rew_list),
        done = np.concatenate(done_list)
    )

    return batch, episode_log


def record_video(
    env: gym.Env,
    policy : BasePolicy = None,
    is_eval : bool = True,
    video_path = './eval.mp4'
):
    n_steps = env._max_episode_steps
    obs = env.reset()
    recorder = VideoRecorder(env, video_path, enabled = True)
    total_reward, total_step = 0, 0

    for _ in range(n_steps):
        if policy is None:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                act, _ = policy.forward(obs, deterministic = is_eval)
                act = act.cpu().numpy()
                act = policy.map_action(act)

        next_obs, reward, done, info = env.step(act)
        recorder.capture_frame()
        obs = np.copy(next_obs)
        total_reward += reward
        total_step += 1

    recorder.close()
    recorder.enabled = False
    env.close()

    print(f'Save the video to path {video_path}!')
    print(f'Rewards = {total_reward} | Steps = {total_step}')