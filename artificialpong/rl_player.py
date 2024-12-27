import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os


class GameState:
    def __init__(self, left_paddle, right_paddle, ball):
        self.left_paddle = left_paddle
        self.right_paddle = right_paddle
        self.ball = ball

    def get_state(self):
        return [
            self.left_paddle.rect.centery / 600,
            self.ball.rect.centerx / 800,
            self.ball.rect.centery / 600,
            self.ball.dx / 10,
            self.ball.dy / 10
        ]


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class RLPlayer:
    def __init__(self, paddle, memory_size=10000, batch_size=64, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, save_file='rl_model.pth'):
        self.paddle = paddle
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.save_file = save_file

        self.model = DQN(5, 3)  # 5 state inputs, 3 actions
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Load model if exists
        self.load()

    def update(self, game_state, action, next_game_state, reward, done):
        state = game_state.get_state()
        next_state = next_game_state.get_state() if next_game_state else [0] * 5

        self.remember(state, action, reward, next_state, done)

        if done:
            self.train()

    def calculate_reward(self, game_state, action, ball_hit, game_over, won):
        # Simple reward structure
        if ball_hit:
            return 0.1
        elif game_over:
            return 1 if won else -1
        else:
            # Small reward for moving towards the ball
            moved_towards_ball = (
                    (action == 0 and game_state.ball.rect.centery < self.paddle.rect.centery) or
                    (action == 2 and game_state.ball.rect.centery > self.paddle.rect.centery)
            )
            return 0.01 if moved_towards_ball else -0.01

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, game_state):
        state = game_state.get_state()
        if random.random() <= self.epsilon:
            return random.randrange(3)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, self.save_file)

    def load(self):
        if os.path.exists(self.save_file):
            checkpoint = torch.load(self.save_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Loaded model. Epsilon: {self.epsilon:.2f}")
        else:
            print("No saved model found. Starting from scratch.")