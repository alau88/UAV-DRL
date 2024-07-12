import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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


def select_action(state, policy_net, epsilon, action_space):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(-1, 1)
    else:
        return torch.tensor([[random.choice(range(action_space.shape[0]))]], dtype=torch.long)


def train_batch(policy_net, target_net, optimizer, batch, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.cat(states)
    actions = torch.cat(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    current_q_values = policy_net(states).gather(1, actions)
    max_next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
