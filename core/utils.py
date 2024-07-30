import random
import torch
import torch.nn as nn
import logging


def select_action(state, policy_net, epsilon, action_space, device='cpu'):
    state = state.view(1, -1).to(device)

    if random.random() > epsilon:
        with torch.no_grad():
            q_values = policy_net(state)
            action_index = q_values.argmax(dim=1).item()
            actions = [action_index]
            logging.info(f'Epsilon: {epsilon}, action selected by policy exploitation: {actions}')
    else:
        action_index = random.randint(0, action_space.n - 1)
        actions = [action_index]
        logging.info(f'Epsilon: {epsilon}, action selected by random exploration: {actions}')

    return torch.tensor(actions, dtype=torch.long, device=state.device)


def train_batch(policy_net, target_net, optimizer, batch, gamma, device='cpu'):
    states, actions, rewards, next_states, dones = zip(*batch)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    states = torch.stack([s.view(-1) for s in states]).to(device)
    next_states = torch.stack([ns.view(-1) for ns in next_states]).to(device)

    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)

    current_q_values = policy_net(states).gather(1, actions)
    max_next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
