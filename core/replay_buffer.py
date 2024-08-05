import numpy as np
import torch
from collections import deque, namedtuple
import random


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedMultiStepReplayBuffer:
    def __init__(self, max_size, n_steps, alpha=0.6, beta_start=0.4, beta_frames=10000, gamma=0.99):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.n_steps = n_steps
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)

    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_steps:
            reward, next_state, done = self._get_n_step_info()
            state, action = self.n_step_buffer[0][:2]
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max(self.priorities, default=1.0))

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = random.choices(range(len(self.buffer)), k=batch_size, weights=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame / self.beta_frames)
        self.frame += 1

        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        batch = Experience(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.mean().item()

    def __len__(self):
        return len(self.buffer)

# class ReplayBuffer:
#     def __init__(self, max_size):
#         self.buffer = deque(maxlen=max_size)
#
#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)
#
#     def __len__(self):
#         return len(self.buffer)
