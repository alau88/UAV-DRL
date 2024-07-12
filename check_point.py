import os
import torch

DIRECTORY = 'checkpoints'


def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    file_path = os.path.join(DIRECTORY, file_name)
    torch.save(state, file_path)
    print(f'Checkpoint saved to {file_path}')


def load_checkpoint(state, file_name='checkpoint.pth.tar'):
    file_path = os.path.join(DIRECTORY, file_name)
    if os.path.exists(file_path):
        state = torch.load(file_path)
        print(f'Checkpoint loaded from {file_path}')
        return state
    else:
        print(f'No checkpoint found at {file_path}')
        return None
