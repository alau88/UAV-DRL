import itertools
import numpy as np
import torch
from core.dqn import DQN
from core.replay_buffer import ReplayBuffer
from core.uav_env import UAVEnv
from core.train import train_dqn
from configs.config import Config
from configs.hyperparams_grid import param_grid
from checkpoints.check_point import save_best_model
from visualization.plotting import plot_training_evaluation_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())


def evaluate_config(config):
    env = UAVEnv(num_users=20, num_uavs=3, area_size=(100, 100))
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.shape[0] * env.action_space.shape[1]

    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_buffer_capacity)

    total_reward_per_episode, eval_reward_per_interval, best_model = train_dqn(env, policy_net,
                                                                               target_net, optimizer,
                                                                               replay_buffer, config)

    mean_reward = np.mean(total_reward_per_episode[-100:])

    return mean_reward, best_model, total_reward_per_episode, eval_reward_per_interval


def run_grid_search():
    best_config = None
    best_reward = -float('inf')
    best_model_overall = None
    best_total_reward_per_episode = None
    best_eval_reward_per_interval = None

    for params in param_combinations:
        config = Config(**dict(zip(param_names, params)))
        mean_reward, best_model, total_reward_per_episode, eval_reward_per_interval = evaluate_config(config)

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_config = config
            best_model_overall = best_model
            best_total_reward_per_episode = total_reward_per_episode
            best_eval_reward_per_interval = eval_reward_per_interval

        print(f"Config: {params}, Mean Reward: {mean_reward}")

    if best_model_overall:
        save_best_model(best_model_overall, file_name="best_model_overall.pth.tar")
        plot_training_evaluation_rewards(best_total_reward_per_episode, best_eval_reward_per_interval,
                                         best_config.evaluation_interval, best_config)

    print(f"Best Config: {vars(best_config)}, Best Mean Reward: {best_reward}")


if __name__ == "__main__":
    run_grid_search()