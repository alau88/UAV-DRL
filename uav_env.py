import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UAVEnv(gym.Env):
    def __init__(self, num_users=10, num_uavs=3, area_size=(100, 100)):
        super(UAVEnv, self).__init__()
        
        self.num_users = num_users
        self.num_uavs = num_uavs
        self.area_size = area_size
        
        # Define action and observation space
        # Actions: Move UAVs in x and y directions
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_uavs, 2), dtype=np.float32)
        
        # Observations: Positions of users and UAVs
        self.observation_space = spaces.Box(low=0, high=max(area_size), shape=(num_users + num_uavs, 2), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        # Randomly place users in the area
        self.user_positions = np.random.rand(self.num_users, 2) * self.area_size
        
        # Initialize UAV positions
        self.uav_positions = np.random.rand(self.num_uavs, 2) * self.area_size
        
        return self._get_obs()
    
    def _get_obs(self):
        # Concatenate user and UAV positions for the observation
        return np.vstack((self.user_positions, self.uav_positions))
    
    def step(self, action):
        # Move UAVs based on the action
        self.uav_positions += action
        
        # Clip UAV positions to stay within the area
        self.uav_positions = np.clip(self.uav_positions, 0, self.area_size)
        
        reward = self._compute_reward()
        
        return self._get_obs(), reward, False, {}
    
    def _compute_reward(self):
        # Compute the reward based on the distance between users and their nearest UAV
        distances = np.linalg.norm(self.user_positions[:, np.newaxis] - self.uav_positions, axis=2)
        min_distances = np.min(distances, axis=1)
        reward = -np.sum(min_distances)
        return reward

