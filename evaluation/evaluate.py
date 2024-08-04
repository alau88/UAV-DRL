import numpy as np
import torch
from evaluation.UVA_performance_metrics import UAVPerformanceMetrics, output_metrics
from checkpoints.check_point import save_evaluation_metrics


def evaluate_policy(env, policy_net, num_episodes=1, device='cpu'):
    performance_metrics = []
    total_rewards = []
    all_user_positions = []
    all_uav_positions = []

    policy_net.to(device)
    policy_net.eval()

    for episode in range(num_episodes):
        performance_metrics_per_episode = UAVPerformanceMetrics()
        state = env.reset().flatten().unsqueeze(0).to(device)
        total_reward = 0
        done = False
        user_positions_history = []
        uav_positions_history = []

        while not done:
            actions = []
            for uav in range(env.num_uavs):
                with torch.no_grad():
                    q_values = policy_net(state)
                    action_index = q_values[0, uav].argmax(dim=0).item()
                    actions.append(action_index)
            action = torch.tensor(actions, dtype=torch.long, device=state.device)

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten().clone().detach().unsqueeze(0).to(device)
            next_state_np = next_state.cpu().numpy().flatten()
            user_positions = next_state_np[:env.num_users * 2].reshape(env.num_users, 2)
            uav_positions = next_state_np[env.num_users * 2:].reshape(env.num_uavs, 2)

            user_positions_history.append(user_positions)
            uav_positions_history.append(uav_positions)

            state = next_state.clone().detach()
            total_reward += reward

            performance_metrics_per_episode.update(uav_positions, user_positions, total_reward)

        total_rewards.append(total_reward)
        all_user_positions.append(user_positions_history)
        all_uav_positions.append(uav_positions_history)
        metrics = performance_metrics_per_episode.get_metrics()
        performance_metrics.append(metrics)
        output_metrics(metrics, episode)

    save_evaluation_metrics(performance_metrics)
    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')

    return avg_reward, all_user_positions, all_uav_positions
