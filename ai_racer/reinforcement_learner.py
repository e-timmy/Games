import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from base_car import BaseCar
import pygame
import math


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReinforcementLearner(BaseCar):
    def __init__(self, pos, color, number, track, waypoint_visualizer):
        super().__init__(pos, color, number)
        self.track = track
        self.waypoint_visualizer = waypoint_visualizer

        # DQN parameters
        self.state_size = 7  # pos_x, pos_y, vel_x, vel_y, angle, progress, on_track
        self.action_size = 5  # accelerate, decelerate, turn_left, turn_right, no_op
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999  # Slower decay
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Track center for directional guidance
        self.track_center = self.track.outer_rect.center

        self.model = DQN(self.state_size, self.action_size).to(self.device)
        self.target_model = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_model()

        # For action visualization
        self.action_keys = {0: 'W', 1: 'S', 2: 'A', 3: 'D', 4: '-'}
        self.last_action = None
        self.last_action_time = 0
        self.action_display_time = 500

        # Progress tracking
        self.last_progress = 0
        self.last_lap_count = 0
        self.stuck_timer = 0
        self.last_position = self.pos[:]

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self):
        # Get normalized position relative to track center
        center_x, center_y = self.track_center
        norm_pos_x = (self.pos[0] - center_x) / (self.track.outer_rect.width / 2)
        norm_pos_y = (self.pos[1] - center_y) / (self.track.outer_rect.height / 2)

        # Normalize velocities
        norm_vel_x = self.vel[0] / self.max_speed
        norm_vel_y = self.vel[1] / self.max_speed

        # Normalize angle to [-1, 1]
        norm_angle = (self.angle % 360) / 180 - 1

        # Get progress and track status
        progress = self.waypoint_visualizer.get_progress(self.pos)
        on_track = 1.0 if self.track.is_on_track(self.pos, self.radius) else 0.0

        return np.array([
            norm_pos_x,
            norm_pos_y,
            norm_vel_x,
            norm_vel_y,
            norm_angle,
            progress,
            on_track
        ])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Next Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def handle_input(self, game_state):
        if not game_state.race_started:
            return

        current_state = self.get_state()
        action = self.act(current_state)
        self.last_action = action
        self.last_action_time = pygame.time.get_ticks()

        # Execute action
        if action == 0:  # accelerate
            self.accelerate()
        elif action == 1:  # decelerate
            self.decelerate()
        elif action == 2:  # turn left
            self.turn(-self.turn_speed)
        elif action == 3:  # turn right
            self.turn(self.turn_speed)
        # action 4 is no_op

        # Check if stuck
        current_speed = math.hypot(*self.vel)
        if current_speed < 0.1:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0

        # Get next state and calculate reward
        next_state = self.get_state()
        done = self.stuck_timer > 100  # Consider episode done if stuck for too long
        reward = self.get_reward(current_state, next_state)

        # Store experience
        self.remember(current_state, action, reward, next_state, done)

        # Learn from experience
        if len(self.memory) > 32:
            self.replay(32)

        # Update progress tracking
        self.last_progress = next_state[5]
        self.last_position = self.pos.copy()
        self.last_lap_count = self.laps

    def get_reward(self, state, next_state):
        # Progress reward
        progress_diff = next_state[5] - state[5]
        if progress_diff < -0.5:  # Crossed finish line forward
            progress_reward = 10.0
        elif progress_diff < -0.1:  # Going backwards
            progress_reward = -5.0
        else:
            progress_reward = progress_diff * 20.0

        # Speed reward: encourage maintaining speed while on track
        speed = math.hypot(self.vel[0], self.vel[1])
        speed_reward = (speed / self.max_speed) * next_state[6] * 2  # Increased weight

        # Track adherence reward
        track_reward = 2.0 if next_state[6] > 0 else -10.0  # Increased penalties

        # Lap completion reward
        lap_reward = 200.0 if self.laps > self.last_lap_count else 0.0

        # Stuck penalty
        stuck_penalty = -2.0 if self.stuck_timer > 50 else 0.0

        return progress_reward + speed_reward + track_reward + lap_reward + stuck_penalty

    def draw_action(self, screen):
        font = pygame.font.Font(None, 36)
        key_size = 40
        base_x = screen.get_width() - 160
        base_y = screen.get_height() - 110

        current_time = pygame.time.get_ticks()
        is_active = current_time - self.last_action_time < self.action_display_time

        for i, (action, key) in enumerate(self.action_keys.items()):
            x = base_x + (i % 3) * 50
            y = base_y + (i // 3) * 50

            is_current = self.last_action == action and is_active

            key_rect = pygame.Rect(x, y, key_size, key_size)
            if is_current:
                pygame.draw.rect(screen, (255, 255, 255), key_rect)
                text_color = (0, 0, 0)
            else:
                pygame.draw.rect(screen, (100, 100, 100), key_rect)
                pygame.draw.rect(screen, (150, 150, 150), key_rect, 2)
                text_color = (255, 255, 255)

            text = font.render(key, True, text_color)
            text_rect = text.get_rect(center=(x + key_size / 2, y + key_size / 2))
            screen.blit(text, text_rect)

    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.model.eval()