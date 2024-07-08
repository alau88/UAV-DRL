import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from uav_env import UAVEnv

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_dqn(env, num_episodes=100, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.shape[0] * env.action_space.shape[1]
    
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters())
    memory = []
    epsilon = epsilon_start
    
    def select_action(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.uniform(-1, 1, env.action_space.shape).flatten()
        else:
            with torch.no_grad():
                return policy_net(torch.FloatTensor(state)).numpy()
    
    for episode in range(num_episodes):
        state = env.reset().flatten()
        total_reward = 0
        
        for t in range(200):
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action.reshape(env.num_uavs, 2))
            next_state = next_state.flatten()
            
            memory.append((state, action, reward, next_state, done))
            if len(memory) > 10000:
                memory.pop(0)
            
            state = next_state
            total_reward += reward
            
            if len(memory) >= batch_size:
                batch = np.random.choice(len(memory), batch_size, replace=False)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*[memory[i] for i in batch])
                
                state_batch = torch.FloatTensor(state_batch)
                action_batch = torch.FloatTensor(action_batch)
                reward_batch = torch.FloatTensor(reward_batch)
                next_state_batch = torch.FloatTensor(next_state_batch)
                done_batch = torch.FloatTensor(done_batch)
                
                q_values = policy_net(state_batch).gather(1, torch.argmax(action_batch, dim=1, keepdim=True)).squeeze()
                next_q_values = target_net(next_state_batch).max(1)[0]
                target_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))
                
                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        
        print(f'Episode {episode}, Total Reward: {total_reward}')
    
    return policy_net

def evaluate_dqn(env, policy_net, num_episodes=10):
    total_rewards = []
    all_user_positions = []
    all_uav_positions = []

    for episode in range(num_episodes):
        state = env.reset().flatten()
        total_reward = 0
        user_positions_history = []
        uav_positions_history = []

        for t in range(200):
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).numpy()
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

def visualize_uav_positions(all_user_positions, all_uav_positions):
    fig = go.Figure()
    num_steps = len(all_user_positions[0])

    for step in range(num_steps):
        user_positions = all_user_positions[0][step]
        uav_positions = all_uav_positions[0][step]
        
        fig.add_trace(go.Scatter(
            x=user_positions[:, 0],
            y=user_positions[:, 1],
            mode='markers',
            marker=dict(color='blue'),
            name=f'Users (Step {step+1})',
            visible=True if step == 0 else 'legendonly'
        ))

        fig.add_trace(go.Scatter(
            x=uav_positions[:, 0],
            y=uav_positions[:, 1],
            mode='markers',
            marker=dict(color='red'),
            name=f'UAVs (Step {step+1})',
            visible=True if step == 0 else 'legendonly'
        ))

    # Create frames for animation
    frames = []
    for step in range(num_steps):
        user_positions = all_user_positions[0][step]
        uav_positions = all_uav_positions[0][step]
        frames.append(go.Frame(data=[
            go.Scatter(
                x=user_positions[:, 0],
                y=user_positions[:, 1],
                mode='markers',
                marker=dict(color='blue'),
                name='Users'
            ),
            go.Scatter(
                x=uav_positions[:, 0],
                y=uav_positions[:, 1],
                mode='markers',
                marker=dict(color='red'),
                name='UAVs'
            )
        ], name=str(step)))

    fig.frames = frames

    # Add slider
    sliders = [dict(
        steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=300, redraw=True), transition=dict(duration=0))], label=str(k)) for k in range(num_steps)],
        transition=dict(duration=0),
        x=0.1,
        xanchor='left',
        y=0,
        yanchor='top'
    )]

    # Update layout with slider
    fig.update_layout(
        title='UAV and User Positions Over Time',
        xaxis_title='X',
        yaxis_title='Y',
        sliders=sliders,
        updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True, transition=dict(duration=0))]), dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])])]
    )

    fig.show()

if __name__ == "__main__":
    env = UAVEnv(num_users=20, num_uavs=3)
    trained_policy = train_dqn(env)

    # Evaluate the trained policy
    avg_reward, all_user_positions, all_uav_positions = evaluate_dqn(env, trained_policy)

    # Visualize the UAV positions
    visualize_uav_positions(all_user_positions, all_uav_positions)
