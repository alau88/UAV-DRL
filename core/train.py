import torch
from core.utils import select_action, train_batch
from checkpoints.check_point import save_checkpoint, save_best_model
# for main training
from configs.config import Config
from core.dqn import DQN
from core.replay_buffer import ReplayBuffer
from core.uav_env import UAVEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_dqn(env, policy_net, target_net, optimizer, replay_buffer, config,
              start_episode=0, best_total_reward=-float('inf')):
    epsilon = config.epsilon_start
    # try:
    for episode in range(start_episode, config.num_episodes):
        state = env.reset().flatten().unsqueeze(0).to(device)
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, policy_net, epsilon, env.action_space)
            next_state, reward, done, _ = env.step(action.detach().cpu().numpy())
            next_state = torch.tensor(next_state.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            total_reward += reward

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) > config.batch_size:
                batch = replay_buffer.sample(config.batch_size)
                train_batch(policy_net, target_net, optimizer, batch, config.gamma)

            epsilon = max(config.epsilon_end, config.epsilon_decay * epsilon)

        if episode % config.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % config.checkpoint_interval == 0:
            checkpoint = {
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'best_total_reward': best_total_reward
            }
            save_checkpoint(checkpoint, file_name=f"checkpoint_{episode}.pth.tar")

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_model = {
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'best_total_reward': best_total_reward,
            }
            save_best_model(best_model)

        print(f'Episode {episode}, Total Reward: {total_reward}')

    # except Exception as e:
    #     print(f"Error during training: {e}")


# TODO: removed after test finished
if __name__ == "__main__":
    config = Config()
    # Initialize networks and optimizer
    env = UAVEnv(num_users=20, num_uavs=3, area_size=(100, 100))
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.shape[0] * env.action_space.shape[1]

    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters())
    replay_buffer = ReplayBuffer(config.replay_buffer_capacity)

    train_dqn(env, policy_net, target_net, optimizer, replay_buffer, config)
