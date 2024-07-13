import numpy as np
import torch


def evaluate_policy(env, policy_net, num_episodes=1, device='cpu'):
    total_rewards = []
    all_user_positions = []
    all_uav_positions = []

    policy_net.to(device)
    policy_net.eval()

    for episode in range(num_episodes):
        state = env.reset().flatten().unsqueeze(0).to(device)
        total_reward = 0
        done = False
        user_positions_history = []
        uav_positions_history = []

        while not done:
            with torch.no_grad():
                action = policy_net(state).cpu().numpy().flatten()
            
            next_state, reward, done, _ = env.step(action.reshape(env.num_uavs, 2))
            next_state = torch.tensor(next_state.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

            next_state_np = next_state.cpu().numpy().flatten()
            user_positions = next_state_np[:env.num_users * 2].reshape(env.num_users, 2)
            uav_positions = next_state_np[env.num_users * 2:].reshape(env.num_uavs, 2)
            
            user_positions_history.append(user_positions)
            uav_positions_history.append(uav_positions)
            
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        all_user_positions.append(user_positions_history)
        all_uav_positions.append(uav_positions_history)
    
    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    return avg_reward, all_user_positions, all_uav_positions
