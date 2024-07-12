class Config:
    def __init__(self):
        self.num_episodes = 1000
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.checkpoint_interval = 50
        self.replay_buffer_capacity = 10000
