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

    offset = window_size // 2
    ax1.plot(range(offset, len(moving_avg_rewards) + offset), moving_avg_rewards,
             color='red', label='Moving Average')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward and Moving Average Over Episodes')
    ax1.legend()

    plt.grid(True)
    save_plot(plt, "Total Reward and Moving Average Over Episodes.png")
    plt.show()