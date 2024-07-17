param_grid = {
    'learning_rate': [0.001, 0.0005],
    'gamma': [0.99, 0.95],
    'epsilon_start': [1.0],
    'epsilon_end': [0.1],
    'epsilon_decay': [0.995, 0.99],
    'batch_size': [64, 32],
    'target_update': [10, 20],
    'replay_buffer_capacity': [10000, 5000]
}


