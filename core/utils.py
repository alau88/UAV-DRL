import random
import torch
import torch.nn as nn
import logging


def select_action(state, policy_net, epsilon, action_space, device='cpu'):
    if random.random() > epsilon:
        with torch.no_grad():
            state = state.to(device)
            action = policy_net(state).view(action_space.shape)
            action = torch.clamp(action, -1, 1)
            logging.info(f'Epsilon: {epsilon}, action selected by policy exploitation: {action}')
            return action
    else:
        action = torch.tensor(action_space.sample(), dtype=torch.float32, device=device)
        logging.info(f'Epsilon: {epsilon}, action selected by random exploration: {action}')
        return action


def train_batch(policy_net, target_net, optimizer, batch, gamma, device='cpu'):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.cat(next_states).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    current_q_values = policy_net(states)

    max_next_q_values = target_net(next_states).max(1)[0].detach()
    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
