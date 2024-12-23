import numpy as np
import random
import os
from collections import deque


class RLPlayer:
    def __init__(self, paddle):
        self.paddle = paddle
        self.state_size = 10  # Extended state: current + previous state
        self.action_size = 3  # up, stay, down
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self.create_model()
        self.load_model()
        self.previous_state = np.zeros(5)
        self.training_stage = 0  # For curriculum learning

    def create_model(self):
        model = {
            'w1': np.random.randn(self.state_size, 32) / np.sqrt(self.state_size),
            'b1': np.zeros((32,)),
            'w2': np.random.randn(32, 32) / np.sqrt(32),
            'b2': np.zeros((32,)),
            'w3': np.random.randn(32, self.action_size) / np.sqrt(32),
            'b3': np.zeros((self.action_size,))
        }
        return model

    def save_model(self):
        np.savez('rl_model.npz', w1=self.model['w1'], b1=self.model['b1'],
                 w2=self.model['w2'], b2=self.model['b2'],
                 w3=self.model['w3'], b3=self.model['b3'],
                 epsilon=self.epsilon, training_stage=self.training_stage)

    def load_model(self):
        if os.path.exists('rl_model.npz'):
            data = np.load('rl_model.npz')
            self.model['w1'] = data['w1']
            self.model['b1'] = data['b1']
            self.model['w2'] = data['w2']
            self.model['b2'] = data['b2']
            self.model['w3'] = data['w3']
            self.model['b3'] = data['b3']
            self.epsilon = data['epsilon']
            self.training_stage = data['training_stage']

    def get_state(self, ball, opponent):
        current_state = np.array([
            (ball.rect.x - self.paddle.rect.x) / 800,
            (ball.rect.y - self.paddle.rect.y) / 600,
            ball.dx / 10,
            ball.dy / 10,
            (self.paddle.rect.y - 300) / 300  # Relative to center of screen
        ])
        state = np.concatenate((current_state, self.previous_state))
        self.previous_state = current_state
        return state

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.forward(state))

    def forward(self, state):
        h1 = np.maximum(0, np.dot(state, self.model['w1']) + self.model['b1'])  # ReLU
        h1 = h1 * (np.random.rand(*h1.shape) > 0.5)  # Dropout for regularization
        h2 = np.maximum(0, np.dot(h1, self.model['w2']) + self.model['b2'])  # ReLU
        h2 = h2 * (np.random.rand(*h2.shape) > 0.5)  # Dropout for regularization
        q = np.dot(h2, self.model['w3']) + self.model['b3']
        return q

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.forward(next_state))
            target_f = self.forward(state)
            target_f[action] = target
            self.backpropagation(state, target_f)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def move(self, ball, opponent):
        state = self.get_state(ball, opponent)
        action = self.act(state)
        if action == 0:
            return True, False
        elif action == 2:
            return False, True
        else:
            return False, False

    def calculate_reward(self, hit_ball, missed, lost, won, moved_towards_ball, ball):
        reward = 0

        # Stage 0: Just focus on hitting the ball
        if self.training_stage == 0:
            if hit_ball:
                reward += 1
            vertical_distance = abs(self.paddle.rect.centery - ball.rect.centery)
            reward += max(0, 1 - vertical_distance / 300)

        # Stage 1: Hitting ball and court positioning
        elif self.training_stage == 1:
            if hit_ball:
                reward += 1
            vertical_distance = abs(self.paddle.rect.centery - ball.rect.centery)
            reward += max(0, 1 - vertical_distance / 300)

            # Encourage middle court positioning
            center_distance = abs(self.paddle.rect.centery - 300)
            reward += max(0, 1 - center_distance / 300)

        # Stage 2: Full game strategy
        else:
            if hit_ball:
                reward += 2
            if missed:
                reward -= 2
            if lost:
                reward -= 5
            if won:
                reward += 5

            vertical_distance = abs(self.paddle.rect.centery - ball.rect.centery)
            reward += max(0, 1 - vertical_distance / 300)

            anticipated_y = ball.rect.centery + (self.paddle.rect.centerx - ball.rect.centerx) * ball.dy / ball.dx
            positioning_quality = 1 - abs(self.paddle.rect.centery - anticipated_y) / 300
            reward += positioning_quality

            # Penalize being at the edges
            if self.paddle.rect.top < 50 or self.paddle.rect.bottom > 550:
                reward -= 1

        return reward

    def update_curriculum(self, performance):
        if self.training_stage < 2 and performance > 0.7:  # If win rate > 70%
            self.training_stage += 1
            self.epsilon = 1.0

    def backpropagation(self, state, target):
        # Forward pass
        h1 = np.maximum(0, np.dot(state, self.model['w1']) + self.model['b1'])
        h2 = np.maximum(0, np.dot(h1, self.model['w2']) + self.model['b2'])
        q = np.dot(h2, self.model['w3']) + self.model['b3']

        # Backward pass
        d_loss = q - target
        d_w3 = np.outer(h2, d_loss)
        d_b3 = d_loss
        d_h2 = np.dot(d_loss, self.model['w3'].T)
        d_h2[h2 <= 0] = 0
        d_w2 = np.outer(h1, d_h2)
        d_b2 = d_h2
        d_h1 = np.dot(d_h2, self.model['w2'].T)
        d_h1[h1 <= 0] = 0
        d_w1 = np.outer(state, d_h1)
        d_b1 = d_h1

        # Update weights and biases
        self.model['w3'] -= self.learning_rate * d_w3
        self.model['b3'] -= self.learning_rate * d_b3
        self.model['w2'] -= self.learning_rate * d_w2
        self.model['b2'] -= self.learning_rate * d_b2
        self.model['w1'] -= self.learning_rate * d_w1
        self.model['b1'] -= self.learning_rate * d_b1
