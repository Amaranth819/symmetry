import gymnasium as gym
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from data import Batch
from collections import defaultdict
from typing import Union
from basepolicy import BasePolicy
from utils import DataListLogger
from gymnasium.wrappers.record_video import RecordVideo


def collect_from_env(
    env: Union[gym.vector.VectorEnv, gym.Env],
    policy : BasePolicy = None,
    n_steps : int = None,
    is_eval : bool = False
):
    is_single_env = not isinstance(env, gym.vector.VectorEnv)

    if n_steps is None:
        n_steps = env._max_episode_steps if is_single_env else 1000

    obs_list = []
    next_obs_list = []
    act_list = []
    rew_list = []
    done_list = []

    n_envs = 1 if is_single_env else env.num_envs
    episode_rewards = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs) 

    obs, _ = env.reset()
    episode_log = DataListLogger()

    for _ in range(n_steps):
        if policy is None:
            act = env.action_space.sample()
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

        if not is_single_env:
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
                episode_rewards = np.zeros(1)
                episode_steps = np.zeros(1)
                if is_eval:
                    break
                else:
                    next_obs, _ = env.reset()

        obs = np.copy(next_obs)

    if not is_single_env:
        batch_obs = np.stack(obs_list, 1)
        batch_next_obs = np.stack(next_obs_list, 1)
        batch_act = np.stack(act_list, 1)
        batch_rew = np.stack(rew_list, 1)
        batch_done = np.stack(done_list, 1)

        batch_obs = batch_obs.reshape(-1, *batch_obs.shape[2:])
        batch_next_obs = batch_next_obs.reshape(-1, *batch_next_obs.shape[2:])
        batch_act = batch_act.reshape(-1, *batch_act.shape[2:])
        batch_rew = batch_rew.reshape(-1, *batch_rew.shape[2:])
        batch_done = batch_done.reshape(-1, *batch_done.shape[2:])
    else:
        batch_obs = np.array(obs_list)
        batch_next_obs = np.array(next_obs_list)
        batch_act = np.array(act_list)
        batch_rew = np.array(rew_list)
        batch_done = np.array(done_list)

    batch = Batch(
        obs = batch_obs,
        next_obs = batch_next_obs,
        act = batch_act,
        rew = batch_rew,
        done = batch_done
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
    video_dir = './eval/',
    plot_info = False
):
    # Need X11 forwarding if run this function on server
    env = gym.make(env_id, render_mode = 'rgb_array')
    n_steps = env._max_episode_steps
    env = RecordVideo(env, video_dir)
    obs, _ = env.reset() # In Gymnasium, reset() function returns (obs, info)
    total_reward, total_step = 0, 0

    plot_info_data = defaultdict(lambda: []) if plot_info else None

    for _ in range(n_steps):
        if policy is None:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                act, _ = policy.forward(obs, deterministic = is_eval)
                act = act.cpu().numpy()
                act = policy.map_action(act)

        next_obs, reward, done, truncated, info = env.step(act)
        obs = np.copy(next_obs)
        total_reward += reward
        total_step += 1

        if plot_info_data is not None:
            plot_info_data['reward_per_time_step'].append(reward)
            for key, val in info.items():
                plot_info_data[key].append(val)

        if done or truncated:
            break

    env.close()

    res_str = f'Rewards = {total_reward} | Steps = {total_step}'
    print(res_str)

    with open(os.path.join(video_dir, 'video_result.txt'), 'w') as f:
        f.write(res_str)

    # Plotting
    time_steps = np.arange(total_step)
    n_rows = len(plot_info_data.keys())
    fig, axs = plt.subplots(n_rows, 1, figsize = (16, 3 * n_rows))
    for i, (key, val_list) in enumerate(plot_info_data.items()):
        axs[i].plot(time_steps, val_list)
        axs[i].set_title(key)
    plt.savefig(os.path.join(video_dir, 'plot_info.png'))
    plt.close()


if __name__ == '__main__':
    env = gym.vector.make('Humanoid-v4', 3)
    # env = gym.make('Humanoid-v4')
    batch, log = collect_from_env(env, None, None, False)
    print(batch)
    _, res_str = log.analysis()
    print(res_str)