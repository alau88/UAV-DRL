class Config:
    def __init__(self,
                 num_episodes=1000,
                 batch_size=64,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 target_update=10,
                 checkpoint_interval=50,
                 replay_buffer_capacity=10000,
                 learning_rate=0.001):
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_update = target_update
        self.checkpoint_interval = checkpoint_interval

        self.replay_buffer_capacity = replay_buffer_capacity
