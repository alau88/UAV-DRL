import os
import torch

BASE_OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'output')
BEST_OVERALL_MODEL_DIRECTORY = os.path.join(BASE_OUTPUT_DIRECTORY, 'best_overall_model')

current_output_directory = None
check_point_directory = None
best_model_directory = None


def set_current_output_directory(config):
    global current_output_directory, check_point_directory, best_model_directory
    config = f"{config}"
    config_dir_name = config.replace(' ', '').replace('(', '').replace(')', '').replace(',', '_')
    current_output_directory = os.path.join(BASE_OUTPUT_DIRECTORY, config_dir_name)
    if not os.path.exists(current_output_directory):
        os.makedirs(current_output_directory)

    check_point_directory = os.path.join(current_output_directory, 'checkpoints_history')
    best_model_directory = os.path.join(current_output_directory, 'best_model')

    print(f"Current output directory set to: {current_output_directory}")


def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    if not os.path.exists(check_point_directory):
        os.makedirs(check_point_directory)
    file_path = os.path.join(check_point_directory, file_name)

    torch.save(state, file_path)
    print(f'Checkpoint saved to {file_path}')


def load_checkpoint(file_name='checkpoint.pth.tar'):
    file_path = os.path.join(check_point_directory, file_name)
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
    file_path = os.path.join(best_model_directory, file_name)
    torch.save(state, file_path)
    print(f'Best model saved to {file_path}')


def save_plot(plt, file_name="training_evaluation_rewards.png", best_overall=False):
    if best_overall:
        if not os.path.exists(BEST_OVERALL_MODEL_DIRECTORY):
            os.makedirs(BEST_OVERALL_MODEL_DIRECTORY)
        plt_file_path = os.path.join(BEST_OVERALL_MODEL_DIRECTORY, file_name)
        plt.savefig(plt_file_path)
        return

    plt_file_path = os.path.join(current_output_directory, file_name)
    plt.savefig(plt_file_path)