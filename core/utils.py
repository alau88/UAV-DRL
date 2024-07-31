import random
import numpy as np
import torch
import torch.nn as nn

def select_action(state, policy_net, epsilon, action_space, device='cpu'):
    if random.random() > epsilon:
        # Exploit: choose the action with the highest Q-value
        with torch.no_grad():
            state = state.to(device)
            q_values = policy_net(state)
            action = q_values.argmax(dim=1).view(-1, 1)
            return action
    else:
        # Explore: choose a random action from the discrete action space
        random_action = torch.tensor([[random.randrange(action_space.n)]], dtype=torch.long, device=device)
        return random_action


def train_batch(policy_net, target_net, optimizer, batch, gamma, device='cpu'):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device).long()
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.cat(next_states).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    current_q_values = policy_net(states).gather(1, actions)
    print(f"Current Q-values: {current_q_values}")
    max_next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
