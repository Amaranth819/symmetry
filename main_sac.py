import argparse
import yaml
import os
import gymnasium as gym
import torch
from sac import SACPolicy
from collect import collect_from_env, eval_policy, record_video
from data import ReplayBuffer
from mpenv import make_mp_diffenvs, SubprocVecEnv
from network import GaussianActorNet, CriticNet
from utils import SummaryLogger, DataListLogger
from register_customized_envs import register_customized_envs


def read_parser():
    parser = argparse.ArgumentParser()

    # Environment settings
    parser.add_argument('--env_id', type = str, default = 'ReducedObsSpaceHumanoidEnv-v0')
    parser.add_argument('--n_envs', type = int, default = 4)
    parser.add_argument('--buffer_capacity', type = int, default = 500000)

    # Policy settings
    parser.add_argument('--device_str', type = str, default = 'cuda', choices = ['auto', 'cpu', 'cuda'])
    parser.add_argument('--actor_hidden_dims', type = list, default = [400, 400])
    parser.add_argument('--min_logstd', type = float, default = -20)
    parser.add_argument('--max_logstd', type = float, default = 2)
    parser.add_argument('--action_bounding_func', type = str, default = '')
    parser.add_argument('--actor_lr', type = float, default = 1e-3)
    parser.add_argument('--critic_hidden_dims', type = list, default = [400, 400])
    parser.add_argument('--critic_lr', type = float, default = 1e-3)
    parser.add_argument('--alpha', type = float, default = 0.2)
    parser.add_argument('--log_alpha_lr', type = float, default = 1e-3)
    parser.add_argument('--target_entropy', type = float, default = -3)
    parser.add_argument('--tau', type = float, default = 0.005)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--pretrain_sac_path', type = str, default = None)

    # Training settings
    parser.add_argument('--num_steps', type = int, default = 500000)
    parser.add_argument('--log_path', type = str, default = 'ReducedObsSpaceHumanoidEnv-v0-alphatuning/')
    parser.add_argument('--update_frequency', type = int, default = 1)
    parser.add_argument('--eval_frequency', type = int, default = 5000)
    parser.add_argument('--n_eval_epochs', default = 5)
    parser.add_argument('--batch_size', type = int, default = 256)

    config = parser.parse_args()
    return config



def create(config):
    register_customized_envs()

    # Create environment
    env = make_mp_diffenvs(config.env_id, [{} for _ in range(config.n_envs)])
    obs_dim = env.get_env_attribute(0, 'observation_space').shape[0]
    act_dim = env.get_env_attribute(0, 'action_space').shape[0]
    eval_env = gym.make(config.env_id)

    # Create buffer
    buffer = ReplayBuffer(config.buffer_capacity)

    # Create networks
    actor = GaussianActorNet(obs_dim, act_dim, config.actor_hidden_dims, config.min_logstd, config.max_logstd)
    actor_optim = torch.optim.Adam(actor.parameters(), lr = config.actor_lr)

    critic1 = CriticNet(obs_dim, act_dim, config.critic_hidden_dims)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr = config.critic_lr)

    critic2 = CriticNet(obs_dim, act_dim, config.critic_hidden_dims)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr = config.critic_lr)

    # Policy
    policy = SACPolicy(
        actor, actor_optim,
        critic1, critic1_optim,
        critic2, critic2_optim,
        config.alpha, config.log_alpha_lr, config.target_entropy, config.gamma, config.tau,
        device = config.device_str
    )
    if config.pretrain_sac_path is not None:
        policy.load(config.pretrain_sac_path)

    return env, eval_env, buffer, policy



def main(
    env : SubprocVecEnv,
    eval_env : gym.Env, 
    buffer : ReplayBuffer, 
    policy : SACPolicy, 
    config : argparse.Namespace
):
    # Save config to file
    summary = SummaryLogger(config.log_path)
    with open(os.path.join(config.log_path, 'config.yaml'), 'w') as f:
        yaml.dump(vars(config), f)

    # Evaluation before training
    total_steps = 0
    n_steps = env._max_episode_steps
    eval_log = eval_policy(eval_env, policy, config.n_eval_epochs)
    eval_res, eval_res_str = eval_log.analysis()
    best_eval_rewards = eval_res['episode_reward_mean']
    summary.add(total_steps, eval_res, prefix = 'Eval/')
    print(f'Eval step {total_steps}: (Deterministic) {eval_res_str}')

    # Start training    
    while total_steps < config.num_steps:
        env.reset()

        # Collect simulated data
        episode_batch, episode_log = collect_from_env(env, policy, n_steps, False)
        buffer.add(episode_batch)

        # Update policy
        train_log = DataListLogger()
        total_train_batches = config.update_frequency * n_steps
        for _ in range(total_train_batches):
            one_time_train_log = policy.update(config.batch_size, buffer)
            for key, val in one_time_train_log.items():
                train_log.add(key, val)
        train_log.merge_logger(episode_log)
        train_res, _ = train_log.analysis()

        # Summary
        total_steps += n_steps
        summary.add(total_steps, train_res, prefix = 'Train/')
        print(f'Train step {total_steps}/{config.num_steps}: (Sample) {episode_log.analysis()[1]}')

        # Evaluation
        if total_steps % config.eval_frequency == 0:
            eval_log = eval_policy(eval_env, policy, config.n_eval_epochs)
            eval_res, eval_res_str = eval_log.analysis()
            summary.add(total_steps, eval_res, prefix = 'Eval/')
            print(f'Eval step {total_steps}: (Deterministic) {eval_res_str}')
            if eval_res['episode_reward_mean'] > best_eval_rewards:
                best_eval_rewards = eval_res['episode_reward_mean']
                print('Got a better model!')
                # Save the best model
                policy.save(os.path.join(config.log_path, 'best_policy.pkl'))
            else:
                print('Did not get a better model!')


    policy.save(os.path.join(config.log_path, 'final_policy.pkl'))
    env.close()



def record_video_with_policy(root_path):
    # Load configuration file
    with open(os.path.join(root_path, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        config = argparse.Namespace(**config)

    video_root_path = os.path.join(root_path, 'video')
    if not os.path.exists(video_root_path):
        os.makedirs(video_root_path)

    # Record video, currently not work on server
    _, _, _, policy = create(config)
    policy.load(os.path.join(root_path, 'final_policy.pkl'))
    record_video(config.env_id, policy, is_eval = True, video_dir = os.path.join(video_root_path, 'final_policy'))
    policy.load(os.path.join(root_path, 'best_policy.pkl'))
    record_video(config.env_id, policy, is_eval = True, video_dir = os.path.join(video_root_path, 'best_policy'))


if __name__ == '__main__':
    # config = read_parser()
    # env, eval_env, buffer, policy = create(config)
    # main(env, eval_env, buffer, policy, config)
    record_video_with_policy('./exps/SAC/Hopper-v4-noalphatuning-2/')