import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import logging

class UAVEnv(gym.Env):
    def __init__(self, num_users=10, num_uavs=3, area_size=(100, 100), max_steps=500):
        super(UAVEnv, self).__init__()

        self.uav_positions = None
        self.user_positions = None
        self.num_users = num_users
        self.num_uavs = num_uavs
        self.area_size = area_size
        self.max_steps = max_steps
        self.step_count = 0
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=max(area_size), shape=(num_users + num_uavs, 2), dtype=np.float32)

        self.reset()

    def set_positions(self, user_positions, uav_positions):
        self.user_positions = np.array(user_positions)
        self.uav_positions = np.array(uav_positions)

    def reset(self):
        self.step_count = 0
        self.user_positions = np.random.rand(self.num_users, 2) * self.area_size if self.user_positions is None else self.user_positions
        self.uav_positions = np.random.rand(self.num_uavs, 2) * self.area_size if self.uav_positions is None else self.uav_positions
        self.uav_positions = self.uav_positions.astype(np.float64)
        self.uav_positions_history = [self.uav_positions.copy()]
        return self._get_obs()

    def _get_obs(self):
        return torch.tensor(np.vstack((self.user_positions, self.uav_positions)), dtype=torch.float32)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        logging.info(f"Step {self.step_count + 1}: Received action: {action}")

        base_step_size = 0.5
        max_distance = np.sqrt((self.area_size[0] ** 2) + (self.area_size[1] ** 2))

        for i in range(self.num_uavs):
            distances = np.linalg.norm(self.user_positions - self.uav_positions[i], axis=1)
            step_size = base_step_size + 0.5 * (distances.max() / max_distance)
            action_set = [
                (0, 0),
                (0, step_size),  # Up
                (0, -step_size),  # Down
                (step_size, 0),  # Right
                (-step_size, 0)  # Left
            ]

            scaled_action = action_set[action[i]]
            logging.info(f"Step {self.step_count + 1}: Scaled action: {scaled_action}")
            self.uav_positions[i] += scaled_action

        self.uav_positions = np.clip(self.uav_positions, [0, 0], self.area_size)
        self.uav_positions_history.append(self.uav_positions.copy())

        # Update user positions
        self._move_users()

        reward = self._compute_reward()
        self.step_count += 1

        logging.info(f"Step {self.step_count}: UAV positions: {self.uav_positions}")
        logging.info(f"Step {self.step_count}: User positions: {self.user_positions}")
        logging.info(f"Step {self.step_count}: Reward: {reward}")

        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, {}

    def _move_users(self):
        # Independent movement for each user
        move_range = 1.0  # Define the maximum movement distance per step
        for i in range(self.num_users):
            movement = (np.random.rand(2) - 0.5) * 2 * move_range
            self.user_positions[i] += movement
            # Clip user positions to stay within the area
            self.user_positions[i] = np.clip(self.user_positions[i], [0, 0], self.area_size)

    def _compute_reward(self):
        distances = np.linalg.norm(self.user_positions[:, np.newaxis] - self.uav_positions, axis=2)
        min_distances = np.min(distances, axis=1)
        max_possible_distance = np.sqrt((self.area_size[0] ** 2) + (self.area_size[1] ** 2))

        distance_reward = -np.sum(min_distances) / (max_possible_distance * self.num_users)
        close_reward = np.sum(min_distances < 5)

        total_reward = distance_reward + close_reward

        return total_reward
