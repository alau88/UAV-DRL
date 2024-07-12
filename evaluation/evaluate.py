import torch
import numpy as np
from uav_env import UAVEnv

def evaluate_policy(env, policy_net, num_episodes=1):
    total_rewards = []
    all_user_positions = []
    all_uav_positions = []

    for episode in range(num_episodes):
        state = env.reset().flatten()
        total_reward = 0
        user_positions_history = []
        uav_positions_history = []

        for t in range(200):
            action = policy_net(state)
            
            next_state, reward, done, _ = env.step(action.reshape(env.num_uavs, 2))
            next_state = next_state.flatten()
            
            user_positions = next_state[:env.num_users * 2].reshape(env.num_users, 2)
            uav_positions = next_state[env.num_users * 2:].reshape(env.num_uavs, 2)
            
            user_positions_history.append(user_positions)
            uav_positions_history.append(uav_positions)
            
            state = next_state
            total_reward += reward
            
            if done:
                break

        total_rewards.append(total_reward)
        all_user_positions.append(user_positions_history)
        all_uav_positions.append(uav_positions_history)
    
    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    return avg_reward, all_user_positions, all_uav_positions
