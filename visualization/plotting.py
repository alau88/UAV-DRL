import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def plot_training_evaluation_rewards(train_reward, eval_reward, eval_interval, config):
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
    # plt.savefig(f"training_evaluation_rewards.png")
    plt.show()
