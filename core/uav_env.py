import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class UAVEnv(gym.Env):
    def __init__(self, num_users=10, num_uavs=3, area_size=(100, 100), max_steps=1000):
        super(UAVEnv, self).__init__()

        self.uav_positions = None
        self.previous_uav_positions = None
        self.user_positions = None
        self.num_users = num_users
        self.num_uavs = num_uavs
        self.area_size = area_size
        self.max_steps = max_steps
        self.step_count = 0
        self.step_size = 1
        # Define action and observation space
        # Define discrete action space including no movement
        # Actions: 0 - No movement, 1 - Up, 2 - Down, 3 - Right, 4 - Left
        self.action_space = spaces.Discrete(5)

        # Observations: Positions of users and UAVs
        self.observation_space = spaces.Box(low=0, high=max(area_size), shape=(num_users + num_uavs, 2),
                                            dtype=np.float32)

        self.reset()

    def reset(self):
        # Randomly place users in the area
        self.user_positions = np.random.rand(self.num_users, 2) * self.area_size

        # Initialize UAV positions
        self.uav_positions = np.random.rand(self.num_uavs, 2) * self.area_size

        self.step_count = 0

        return self._get_obs()

    def _get_obs(self):
        # Concatenate user and UAV positions for the observation
        return torch.tensor(np.vstack((self.user_positions, self.uav_positions)), dtype=torch.float32)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        # Initialize movement with zero (no movement)
        movement = np.zeros((self.num_uavs, 2))

        # Define movement based on the action
        if action == 1:  # Up
            movement[:, 1] = 1
        elif action == 2:  # Down
            movement[:, 1] = -1
        elif action == 3:  # Right
            movement[:, 0] = 1
        elif action == 4:  # Left
            movement[:, 0] = -1

        # Apply step size
        movement *= self.step_size
        print(movement)
        # Update UAV positions based on the movement
        self.uav_positions += movement

        # Clip UAV positions to stay within the area boundaries
        self.uav_positions = np.clip(self.uav_positions, 0, self.area_size)

        reward = self._compute_reward()
        self.step_count += 1

        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, {}


    def _compute_reward(self):
        # Compute distances between users and UAVs
        distances = np.linalg.norm(self.user_positions[:, np.newaxis] - self.uav_positions, axis=2)
        min_distances = np.min(distances, axis=1)
        total_distance = np.sum(min_distances)

        # Normalize total distance
        max_possible_distance = np.sqrt((self.area_size[0] ** 2) + (self.area_size[1] ** 2))
        normalized_total_distance = total_distance / max_possible_distance

        # Base reward is negative normalized distance
        reward = -normalized_total_distance

            # Penalize proximity to boundaries (example assuming a 2D area)
        boundary_penalty = 0
        for pos in self.uav_positions:
            boundary_penalty += np.exp(-np.min([pos[0], self.area_size[0] - pos[0], pos[1], self.area_size[1] - pos[1]]))

        reward -= boundary_penalty  # Apply penalty for being near the boundary

        self.previous_uav_positions = np.copy(self.uav_positions)

        return reward
