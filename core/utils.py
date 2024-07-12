import random
import torch
import torch.nn as nn


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
