import itertools
import numpy as np
import torch
from core.dqn import DQN, DuelingDQN
from core.replay_buffer import ReplayBuffer
from core.uav_env import UAVEnv
from core.train import train_dqn
from configs.config import Config
from configs.hyperparams_grid import param_grid
from checkpoints.check_point import save_best_model, set_current_output_directory
from visualization.plotting import plot_training_evaluation_rewards, plot_training_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())


def evaluate_config(config,network):
    env = UAVEnv(num_users=20, num_uavs=3, area_size=(100, 100))
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.shape[0] * env.action_space.shape[1]
    if network == "DQN":
        policy_net = DQN(state_size, action_size).to(device)
        target_net = DQN(state_size, action_size).to(device)
    elif network == "DuelingDQN":
        policy_net = DuelingDQN(state_size, action_size).to(device)
        target_net = DuelingDQN(state_size, action_size).to(device)
    else:
        raise Exception("Please choose valid DQN class from ./core/dqn.py")
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_buffer_capacity)

    total_reward_per_episode, eval_reward_per_interval, best_model, avg_losses_per_episode = train_dqn(env, policy_net,
                                                                               target_net, optimizer,
                                                                               replay_buffer, config)
    plot_training_losses(avg_losses_per_episode)

    plot_training_evaluation_rewards(total_reward_per_episode, eval_reward_per_interval,
                                     config.evaluation_interval, config)

    mean_reward = np.mean(total_reward_per_episode[-100:])

    return mean_reward, best_model, total_reward_per_episode, eval_reward_per_interval


def run_grid_search():
    count = 0
    best_config = None
    best_reward = -float('inf')
    best_model_overall = None
    best_total_reward_per_episode = None
    best_eval_reward_per_interval = None

    for params in param_combinations:
        set_current_output_directory(params)
        config = Config(**dict(zip(param_names, params)))
        mean_reward, best_model, total_reward_per_episode, eval_reward_per_interval = evaluate_config(config)

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_config = config
            best_model_overall = best_model
            best_total_reward_per_episode = total_reward_per_episode
            best_eval_reward_per_interval = eval_reward_per_interval

        print(f"Config: {params}, Mean Reward: {mean_reward}")

        if count % 10 == 0:
            save_best_model(best_model_overall, file_name="best_model_overall.pth.tar", best_overall=True)
            plot_training_evaluation_rewards(best_total_reward_per_episode, best_eval_reward_per_interval,
                                         best_config.evaluation_interval, best_config, best_overall=True)
            print(f"Count:{count}, Best Config: {vars(best_config)}, Best Mean Reward: {best_reward}")

        count = count + 1

    if best_model_overall:
        save_best_model(best_model_overall, file_name="best_model_overall.pth.tar", best_overall=True)
        plot_training_evaluation_rewards(best_total_reward_per_episode, best_eval_reward_per_interval,
                                         best_config.evaluation_interval, best_config, best_overall=True)

    print(f"Best Config: {vars(best_config)}, Best Mean Reward: {best_reward}")


if __name__ == "__main__":
    #run_grid_search()
    evaluate_config(Config)
