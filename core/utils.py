import random
import numpy as np
import torch
import torch.nn as nn
import logging


def select_action(state, policy_net, epsilon, action_space, num_uavs, device='cpu'):
    state = state.view(1, -1).to(device)
    actions = []

    for uav in range(num_uavs):
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = policy_net(state)
                action_index = q_values[0, uav].argmax(dim=0).item()
                actions.append(action_index)
                logging.info(f'Epsilon: {epsilon}, action selected by policy exploitation: {actions}')
        else:
            action_index = random.randint(0, action_space.n - 1)
            actions.append(action_index)
            logging.info(f'Epsilon: {epsilon}, action selected by random exploration: {actions}')

    return torch.tensor(actions, dtype=torch.long, device=state.device)


def process_batch_element(batch, policy_net, device):
    states, actions, rewards, next_states, dones = zip(*batch)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    states = torch.stack([s.view(-1) for s in states]).to(device)
    next_states = torch.stack([ns.view(-1) for ns in next_states]).to(device)

    actions = torch.stack([torch.tensor(a, dtype=torch.long) for a in actions]).to(device)
    batch_size, num_uavs = actions.shape

    rewards = rewards.unsqueeze(1).expand(batch_size, num_uavs)
    dones = dones.unsqueeze(1).expand(batch_size, num_uavs)

    actions = actions.unsqueeze(2)

    policy_net_output = policy_net(states)
    current_q_values = policy_net_output.gather(2, actions).squeeze(2)

    return next_states, rewards, dones, current_q_values


def train_batch(policy_net, target_net, optimizer, batch, gamma, device='cpu'):
    next_states, rewards, dones, current_q_values = process_batch_element(batch, policy_net, device)

    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(2)[0]

    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_double_dqn_batch(policy_net, target_net, optimizer, batch, gamma, device='cpu'):
    next_states, rewards, dones, current_q_values = process_batch_element(batch, policy_net, device)

    with torch.no_grad():
        next_state_actions = policy_net(next_states).argmax(dim=2, keepdim=True)
        max_next_q_values = target_net(next_states).gather(2, next_state_actions).squeeze(2)

    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def calculate_average_movement(uav_positions_history):
    average_movements = []

    for episode_positions in uav_positions_history:
        movements = []
        for i in range(1, len(episode_positions)):
            movement = np.linalg.norm(np.array(episode_positions[i]) - np.array(episode_positions[i - 1]), axis=1)
            movements.append(np.mean(movement))
        average_movements.append(np.mean(movements))

    return average_movements
