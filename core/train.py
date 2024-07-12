import torch
from core.utils import select_action, train_batch
from checkpoints.check_point import save_checkpoint, save_best_model


def train_dqn(env, policy_net, target_net, optimizer, replay_buffer, config,
              start_episode=0, best_total_reward=-float('inf')):
    epsilon = config.epsilon_start
    # try:
    for episode in range(start_episode, config.num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, policy_net, epsilon, env.action_space)
            next_state, reward, done, _ = env.step(action.numpy())
            next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
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
