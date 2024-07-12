import matplotlib.pyplot as plt
from uav_env import UAVEnv
import numpy as np

# Initialize environment
env = UAVEnv(num_users=20, num_uavs=3)

# Reset environment
obs = env.reset()

# Define a simple movement policy for demonstration (random movements)
num_steps = 50
for step in range(num_steps):
    action = np.random.uniform(-1, 1, (env.num_uavs, 2))
    obs, reward, done, info = env.step(action)

    # Extract user and UAV positions
    user_positions = obs[:env.num_users]
    uav_positions = obs[env.num_users:]
    
    # Plot positions
    plt.scatter(user_positions[:, 0], user_positions[:, 1], c='blue', label='Users')
    plt.scatter(uav_positions[:, 0], uav_positions[:, 1], c='red', label='UAVs')
    plt.xlim(0, env.area_size[0])
    plt.ylim(0, env.area_size[1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Step {step+1}')
    plt.show()
    plt.pause(0.5)
    plt.clf()

plt.close()
