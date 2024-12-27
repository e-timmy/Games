class Config:
    def __init__(self):
        # Game settings
        self.screen_width = 800
        self.screen_height = 600
        self.fps = 60

        # RL settings
        self.memory_size = 10000
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.save_file = 'rl_model.pth'