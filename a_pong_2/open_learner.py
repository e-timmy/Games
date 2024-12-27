import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from player import Player


class AdvancedDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(AdvancedDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class OpenLearner(Player):
    def __init__(self, paddle, config):
        super().__init__(paddle)
        self.memory = deque(maxlen=config.memory_size)
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.save_file = 'open_learner.pth'

        # More complex state representation
        self.state_size = 10  # Expanded state representation
        self.action_size = 3  # No change: up, stay, down

        self.model = AdvancedDQN(self.state_size, self.action_size)
        self.target_model = AdvancedDQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.update_target_every = 10
        self.game_count = 0

        self.load()

    def get_state(self, game_state):
        """More complex state representation"""
        return [
            self.paddle.rect.centery / 600,  # Normalized paddle y position
            game_state.ball.rect.centerx / 800,  # Normalized ball x
            game_state.ball.rect.centery / 600,  # Normalized ball y
            game_state.ball.dx / 10,  # Normalized ball x velocity
            game_state.ball.dy / 10,  # Normalized ball y velocity
            game_state.ball.speed / 10,  # Normalized ball speed
            (game_state.ball.rect.centerx - self.paddle.rect.centerx) / 800,  # Relative x distance
            (game_state.ball.rect.centery - self.paddle.rect.centery) / 600,  # Relative y distance
            game_state.left_score / 10,  # Normalized score
            game_state.right_score / 10,  # Normalized opponent score
        ]

    def act(self, game_state):
        state = self.get_state(game_state)
        if random.random() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.model(state_tensor)
                action = q_values.argmax().item()

        # Execute action
        self.paddle.move(action == 0, action == 2)
        return action

    def update(self, game_state, action, next_state, reward, done):
        current_state = self.get_state(game_state)
        next_state_data = self.get_state(next_state) if next_state else [0] * self.state_size

        self.remember(current_state, action, reward, next_state_data, done)

        self.train()

        if done:
            self.game_count += 1
            if self.game_count % self.update_target_every == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if game_state.games_played % 100 == 0:
                self.save()
                win_rate = game_state.left_score / game_state.games_played
                print(f"Games: {game_state.games_played}, Win rate: {win_rate:.2f}")

    def calculate_reward(self, game_state, action, ball_hit, game_over, won):
        reward = 0

        # Reward for hitting the ball
        if ball_hit:
            reward += 0.5

        # Reward for winning/losing
        if game_over:
            reward += 1 if won else -1

        # Small reward for moving towards the ball
        paddle_to_ball = game_state.ball.rect.centery - self.paddle.rect.centery
        if (action == 0 and paddle_to_ball < 0) or (action == 2 and paddle_to_ball > 0):
            reward += 0.1

        # Penalty for moving away from the ball
        if (action == 0 and paddle_to_ball > 0) or (action == 2 and paddle_to_ball < 0):
            reward -= 0.05

        return reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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

        # Use target network for next state Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'game_count': self.game_count
        }, self.save_file)

    def load(self):
        if os.path.exists(self.save_file):
            checkpoint = torch.load(self.save_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.game_count = checkpoint['game_count']
            print(f"Loaded model. Epsilon: {self.epsilon:.2f}, Games played: {self.game_count}")
        else:
            print("No saved model found. Starting from scratch.")