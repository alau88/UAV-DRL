import os
import torch

BASE_OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'output')
CHECK_POINT_DIRECTORY = os.path.join(BASE_OUTPUT_DIRECTORY, 'checkpoints_history')
BEST_MODEL_DIRECTORY = os.path.join(BASE_OUTPUT_DIRECTORY, 'best_model')


def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    if not os.path.exists(CHECK_POINT_DIRECTORY):
        os.makedirs(CHECK_POINT_DIRECTORY)
    file_path = os.path.join(CHECK_POINT_DIRECTORY, file_name)
    torch.save(state, file_path)
    print(f'Checkpoint saved to {file_path}')


def load_checkpoint(file_name='checkpoint.pth.tar'):
    file_path = os.path.join(CHECK_POINT_DIRECTORY, file_name)
    if os.path.exists(file_path):
        state = torch.load(file_path)
        print(f'Checkpoint loaded from {file_path}')
        return state
    else:
        print(f'No checkpoint found at {file_path}')
        return None


def save_best_model(state, file_name='best_model.pth.tar'):
    if not os.path.exists(BEST_MODEL_DIRECTORY):
        os.makedirs(BEST_MODEL_DIRECTORY)
    file_path = os.path.join(BEST_MODEL_DIRECTORY, file_name)
    torch.save(state, file_path)
    print(f'Best model saved to {file_path}')
