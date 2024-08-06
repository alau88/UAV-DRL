import os
import torch
import pandas as pd

BASE_OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'output')
BEST_OVERALL_MODEL_DIRECTORY = os.path.join(BASE_OUTPUT_DIRECTORY, 'best_overall_model')

current_output_directory = None
check_point_directory = None
best_model_directory = None


def set_current_output_directory(network):
    global current_output_directory, check_point_directory, best_model_directory
    current_output_directory = os.path.join(BASE_OUTPUT_DIRECTORY, network)
    if not os.path.exists(current_output_directory):
        os.makedirs(current_output_directory)

    check_point_directory = os.path.join(current_output_directory, 'checkpoints_history')
    best_model_directory = os.path.join(current_output_directory, 'best_model')

    print(f"Current output directory set to: {current_output_directory}")


def return_path(directory, file_name):
    return os.path.join(directory, file_name)


def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    if not os.path.exists(check_point_directory):
        os.makedirs(check_point_directory)

    file_path = return_path(check_point_directory, file_name)

    torch.save(state, file_path)
    print(f'Checkpoint saved to {file_path}')


def load_checkpoint(file_name='checkpoint.pth.tar'):
    file_path = return_path(check_point_directory, file_name)
    if os.path.exists(file_path):
        state = torch.load(file_path)
        print(f'Checkpoint loaded from {file_path}')
        return state
    else:
        print(f'No checkpoint found at {file_path}')
        return None


def save_best_model(state, file_name='best_model.pth.tar', best_overall=False):
    if best_overall:
        if not os.path.exists(BEST_OVERALL_MODEL_DIRECTORY):
            os.makedirs(BEST_OVERALL_MODEL_DIRECTORY)
        file_path = os.path.join(BEST_OVERALL_MODEL_DIRECTORY, file_name)
        torch.save(state, file_path)
        return

    if not os.path.exists(best_model_directory):
        os.makedirs(best_model_directory)
    file_path = return_path(best_model_directory, file_name)
    torch.save(state, file_path)
    print(f'Best model saved to {file_path}')


def save_evaluation_metrics(performance_metrics, file_name='uav_performance_metrics.csv'):
    df_metrics = pd.DataFrame(performance_metrics)
    csv_file_path = return_path(current_output_directory, file_name)
    df_metrics.to_csv(csv_file_path, index=False)


def save_plot(plt, file_name, best_overall=False):
    if best_overall:
        if not os.path.exists(BEST_OVERALL_MODEL_DIRECTORY):
            os.makedirs(BEST_OVERALL_MODEL_DIRECTORY)
        plt_file_path = os.path.join(BEST_OVERALL_MODEL_DIRECTORY, file_name)
        plt.savefig(plt_file_path)
        return

    plt_file_path = return_path(current_output_directory, file_name)
    plt.savefig(plt_file_path)


def save_train_metrics(total_reward_per_episode, avg_losses_per_episode,
                       epsilon_values, training_time, file_name='training_metrics.csv'):

    train_metrics = {
        'Total Reward': total_reward_per_episode,
        'Epsilon': epsilon_values,
        'Average Loss': avg_losses_per_episode,
        'Training Time': training_time
    }
    df_metrics = pd.DataFrame(train_metrics)
    csv_file_path = return_path(current_output_directory, file_name)
    df_metrics.to_csv(csv_file_path, index=False)
