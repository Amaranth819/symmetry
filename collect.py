import gymnasium as gym
import numpy as np
import torch
import os
from data import Batch
from mpenv import SubprocVecEnv
from typing import Union
from basepolicy import BasePolicy
from utils import DataListLogger
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.utils.save_video import save_video


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

    n_envs = env.n_envs if isinstance(env, SubprocVecEnv) else 1
    episode_rewards = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs) 

    obs, _ = env.reset()
    episode_log = DataListLogger()

    for _ in range(n_steps):
        if policy is None:
            act = env.sample_actions() if isinstance(env, SubprocVecEnv) else env.action_space.sample()
        else:
            with torch.no_grad():
                act, _ = policy.forward(obs, deterministic = is_eval)
                act = act.cpu().numpy()
                act = policy.map_action(act)

        next_obs, reward, done, truncated, infos = env.step(act)
        done = np.logical_or(done, truncated)

        obs_list.append(obs)
        next_obs_list.append(next_obs)
        act_list.append(act)
        rew_list.append(reward)
        done_list.append(done)

        episode_rewards += reward
        episode_steps += 1

        if isinstance(env, SubprocVecEnv):
            for i in range(n_envs):
                if done[i]:
                    episode_log.add('episode_reward', episode_rewards[i])
                    episode_log.add('episode_step', episode_steps[i])
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
        else:
            if done:
                episode_log.add('episode_reward', episode_rewards)
                episode_log.add('episode_step', episode_steps)
                episode_rewards = 0
                episode_steps = 0
                if is_eval:
                    break

        obs = np.copy(next_obs)

    if isinstance(env, SubprocVecEnv):
        batch = Batch(
            obs = np.concatenate(obs_list),
            next_obs = np.concatenate(next_obs_list),
            act = np.concatenate(act_list),
            rew = np.concatenate(rew_list),
            done = np.concatenate(done_list)
        )
    else:
        batch = Batch(
            obs = np.array(obs_list),
            next_obs = np.array(next_obs_list),
            act = np.array(act_list),
            rew = np.array(rew_list),
            done = np.array(done_list)
        )

    return batch, episode_log


def eval_policy(
    env : gym.Env,
    policy : BasePolicy,
    n_eval_epochs : int,
    n_steps = None
):
    if n_steps is None:
        n_steps = env._max_episode_steps
    all_eval_log = DataListLogger()
    for _ in range(n_eval_epochs):
        _, episode_log = collect_from_env(env, policy, n_steps, True)
        all_eval_log.merge_logger(episode_log)
    return all_eval_log


def record_video(
    env_id: str,
    policy : BasePolicy = None,
    is_eval : bool = True,
    video_dir = './eval/'
):
    # Need X11 forwarding if run this function on server
    env = gym.make(env_id, render_mode = 'rgb_array')
    n_steps = env._max_episode_steps
    env = RecordVideo(env, video_dir)
    obs, _ = env.reset() # In Gymnasium, reset() function returns (obs, info)
    total_reward, total_step = 0, 0

    for _ in range(n_steps):
        if policy is None:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                act, _ = policy.forward(obs, deterministic = is_eval)
                act = act.cpu().numpy()
                act = policy.map_action(act)

        next_obs, reward, done, truncated, _ = env.step(act)
        obs = np.copy(next_obs)
        total_reward += reward
        total_step += 1

        if done or truncated:
            break

    env.close()

    res_str = f'Rewards = {total_reward} | Steps = {total_step}'
    print(res_str)

    with open(os.path.join(video_dir, 'video_result.txt'), 'w') as f:
        f.write(res_str)



if __name__ == '__main__':
    env = gym.make('HalfCheetah-v4')
    # from mpenv import make_mp_diffenvs
    # env = make_mp_diffenvs('HalfCheetah-v4', [{} for _ in range(2)])
    batch, log = collect_from_env(env, None)
    print(batch)
    print(log.analysis())