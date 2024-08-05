class Config:
    def __init__(self,
                 network="DQN",
                 num_episodes=51,
                 batch_size=64,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 target_update=10,
                 checkpoint_interval=50,
                 replay_buffer_capacity=5000,
                 learning_rate=0.0001,
                 evaluation_interval=50):
        self.network = network
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

        self.evaluation_interval = evaluation_interval

    def to_dict(self):
        return {
            "network": self.network,
            "num_episodes": self.num_episodes,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "target_update": self.target_update,
            "checkpoint_interval": self.checkpoint_interval,
            "evaluation_interval": self.evaluation_interval,
            "replay_buffer_capacity": self.replay_buffer_capacity
        }