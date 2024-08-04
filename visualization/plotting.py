import matplotlib.pyplot as plt
import numpy as np
from checkpoints.check_point import save_plot


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def plot_training_evaluation_rewards(train_reward, eval_reward, eval_interval, config, best_overall=False):
    params = config.to_dict()
    smoothed_train_reward = moving_average(train_reward, window=50)

    plt.figure(figsize=(12, 6))
    plt.title("Training and Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.plot(smoothed_train_reward, label="Training Rewards per Episode")

    eval_episodes = list(range(eval_interval, eval_interval * (len(eval_reward) + 1), eval_interval))
    plt.plot(eval_episodes, eval_reward, 'ro-', label="Evaluation Rewards")

    param_text = "\n".join([f"{key}: {value}" for key, value in params.items()])
    plt.figtext(0.87, 0.24, param_text, ha="right", va="bottom",
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

    plt.legend()
    plt.grid()
    save_plot(plt, "Training and Evaluation Rewards.png", best_overall=best_overall)
    plt.show()


def plot_training_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Time')

    save_plot(plt, "Training Loss over Time.png")
    plt.show()


def plot_average_movement(average_movements):
    plt.figure(figsize=(10, 6))
    plt.plot(average_movements, label='Average Movement per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Movement')
    plt.title('Average Movement of UAVs Over Episodes.png')
    plt.legend()
    plt.grid(True)

    save_plot(plt, "Average Movement of UAVs Over Episodes")
    plt.show()


def plot_total_reward_and_moving_average(total_reward_per_episode):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plotting total reward
    ax1.plot(total_reward_per_episode, color='gray', alpha=0.5, label='Total Reward')

    window_size = 50
    moving_avg_rewards = moving_average(total_reward_per_episode, window_size)

    # Calculate the standard deviation for the shaded area
    std_rewards = np.std(total_reward_per_episode[:len(moving_avg_rewards)])
    lower_bound = moving_avg_rewards - std_rewards
    upper_bound = moving_avg_rewards + std_rewards

    # Padding for the moving average
    padding = (len(total_reward_per_episode) - len(moving_avg_rewards)) // 2
    moving_avg_rewards_padded = np.pad(moving_avg_rewards, (padding, padding), 'edge')
    lower_bound_padded = np.pad(lower_bound, (padding, padding), 'edge')
    upper_bound_padded = np.pad(upper_bound, (padding, padding), 'edge')

    # Ensure all padded arrays are the same length as total_reward_per_episode
    if len(moving_avg_rewards_padded) < len(total_reward_per_episode):
        diff = len(total_reward_per_episode) - len(moving_avg_rewards_padded)
        moving_avg_rewards_padded = np.pad(moving_avg_rewards_padded, (0, diff), 'edge')
        lower_bound_padded = np.pad(lower_bound_padded, (0, diff), 'edge')
        upper_bound_padded = np.pad(upper_bound_padded, (0, diff), 'edge')
    else:
        moving_avg_rewards_padded = moving_avg_rewards_padded[:len(total_reward_per_episode)]
        lower_bound_padded = lower_bound_padded[:len(total_reward_per_episode)]
        upper_bound_padded = upper_bound_padded[:len(total_reward_per_episode)]

    # Plotting the moving average with shaded area
    ax1.plot(moving_avg_rewards_padded, color='red', label='Moving Average')
    ax1.fill_between(range(len(total_reward_per_episode)), lower_bound_padded, upper_bound_padded, color='red', alpha=0.2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward and Moving Average Over Episodes')
    ax1.legend()

    plt.grid(True)
    save_plot(plt, "Total Reward and Moving Average Over Episodes.png")
    plt.show()