import torch
import numpy as np
from core.utils import select_action, train_batch
from checkpoints.check_point import save_checkpoint, save_best_model
from evaluation.evaluate import evaluate_policy
import logging

logging.basicConfig(filename='training_debug.log', level=logging.INFO)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_dqn(env, policy_net, target_net, optimizer, replay_buffer, config,
              start_episode=0, best_total_reward=-float('inf')):
    epsilon = config.epsilon_start
    total_reward_per_episode = []
    evaluation_rewards_per_interval = []
    best_model = None
    episode_losses = []

    for episode in range(start_episode, config.num_episodes):
        total_reward, epsilon, losses = run_episode(env, policy_net, target_net, optimizer,
                                            replay_buffer, config, epsilon)
        total_reward_per_episode.append(total_reward)
        episode_losses.append(np.mean(losses))

        if episode > 0 and episode % config.target_update == 0:
            update_target_net(policy_net, target_net)

        if episode > 0 and episode % config.checkpoint_interval == 0:
            save_checkpoint_at_interval(episode, policy_net, target_net,
                                        optimizer, epsilon, best_total_reward)

        if total_reward > best_total_reward:
            best_total_reward, best_model = save_best_model_if_improved(episode, policy_net, target_net,
                                                            optimizer, epsilon, total_reward)

        print(f'Episode {episode}, Total Reward: {total_reward}')
        logging.info(f'Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}')

        if episode > 0 and episode % config.evaluation_interval == 0:
            eval_reward, _, _ = evaluate_policy(env, policy_net, num_episodes=10, device=device)
            evaluation_rewards_per_interval.append(eval_reward)

    return total_reward_per_episode, evaluation_rewards_per_interval, best_model, episode_losses


def run_episode(env, policy_net, target_net, optimizer, replay_buffer, config, epsilon):
    state = env.reset().to(device)
    total_reward = 0
    done = False
    episode_losses = []

    while not done:
        action = select_action(state, policy_net, epsilon, env.action_space, device)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.to(device)

        total_reward += reward

        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer) > config.batch_size:
            batch = replay_buffer.sample(config.batch_size)
            loss = train_batch(policy_net, target_net, optimizer, batch, config.gamma, device)
            episode_losses.append(loss) 

        epsilon = max(config.epsilon_end, config.epsilon_decay * epsilon)

    return total_reward, epsilon, episode_losses


def update_target_net(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())


def save_checkpoint_at_interval(episode, policy_net, target_net, optimizer, epsilon, best_total_reward):
    checkpoint = {
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'best_total_reward': best_total_reward
    }
    save_checkpoint(checkpoint, file_name=f"checkpoint_{episode}.pth.tar")


def save_best_model_if_improved(episode, policy_net, target_net, optimizer, epsilon, total_reward):
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

    return best_total_reward, best_model
