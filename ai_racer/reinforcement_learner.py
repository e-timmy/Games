import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math
from base_car import BaseCar
import pygame


class ReinforcementLearner(BaseCar):
    def __init__(self, pos, color, number, track):
        super().__init__(pos, color, number)
        self.track = track
        self.waypoints, self.waypoint_rows = self._generate_track_waypoints()
        self.total_waypoints = len(self.waypoints)
        self.waypoint_status = [False] * self.total_waypoints  # Track passed waypoints
        self.flash_waypoints = set()  # Waypoints that should flash this frame
        self.flash_colors = {}  # Colors for flashing waypoints
        self.activation_radius = 100  # Area of activation for waypoints

        # Neural Network
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 actions: accelerate, turn left, turn right
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Replay memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.99  # discount factor

        # Epsilon-greedy exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.total_reward = 0
        self._initialize_waypoint_status()

    def _generate_track_waypoints(self):
        waypoints = []
        waypoint_rows = []
        spacing = 30  # Distance between waypoints

        # Calculate track boundaries
        left = self.track.outer_rect.left + self.track.inset // 2
        right = self.track.outer_rect.right - self.track.inset // 2
        top = self.track.outer_rect.top + self.track.inset // 2
        bottom = self.track.outer_rect.bottom - self.track.inset // 2

        # Generate waypoints across the entire width of the track
        for x in range(left, right + 1, spacing):
            row = []
            for y in range(top, bottom + 1, spacing):
                # Check if the point is on the track
                if self.track.is_on_track((x, y), 0):
                    # Calculate reward based on distance to center line
                    center_line_x = (left + right) // 2
                    distance_to_center = abs(x - center_line_x)
                    max_distance = (right - left) // 2
                    reward_multiplier = max(0.5, 2 - (distance_to_center / max_distance))
                    waypoints.append(((x, y), reward_multiplier))
                    row.append(len(waypoints) - 1)
            if row:
                waypoint_rows.append(row)

        return waypoints, waypoint_rows

    def _initialize_waypoint_status(self):
        start_y = self.track.start_line.centery
        self.waypoint_status = []
        for i in range(len(self.waypoints)):
            wp_y = self.waypoints[i][0][1]
            self.waypoint_status.append(wp_y < start_y)  # True (yellow) if behind start line

    def _get_state(self):
        # Normalize position
        norm_x = (self.pos[0] - self.track.outer_rect.left) / self.track.outer_rect.width
        norm_y = (self.pos[1] - self.track.outer_rect.top) / self.track.outer_rect.height
        return np.array([norm_x, norm_y], dtype=np.float32)

    def handle_input(self, game_state):
        if not game_state.race_started:
            return

        state = self._get_state()
        state_tensor = torch.FloatTensor(state)

        # Clear flashing waypoints from previous frame
        self.flash_waypoints.clear()

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = q_values.argmax().item()

        # Perform action
        if action == 0:
            self.accelerate()
        elif action == 1:
            self.turn(-self.turn_speed)
        else:
            self.turn(self.turn_speed)

        # Calculate reward and update waypoints
        reward = self._calculate_reward()

        # Store experience
        next_state = self._get_state()
        self.memory.append((state, action, reward, next_state, game_state.race_started))

        # Train model
        if len(self.memory) > self.batch_size:
            self._train()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _calculate_reward(self):
        reward = 0
        activated_row = None

        # Check for activated waypoint rows
        for row in self.waypoint_rows:
            for wp_index in row:
                distance = math.hypot(
                    self.waypoints[wp_index][0][0] - self.pos[0],
                    self.waypoints[wp_index][0][1] - self.pos[1]
                )
                if distance < self.activation_radius:
                    activated_row = row
                    break
            if activated_row:
                break

        if activated_row:
            for wp_index in activated_row:
                if not self.waypoint_status[wp_index]:
                    # Positive reward based on proximity and multiplier
                    distance = math.hypot(
                        self.waypoints[wp_index][0][0] - self.pos[0],
                        self.waypoints[wp_index][0][1] - self.pos[1]
                    )
                    reward += max(0, (self.activation_radius - distance) / 10) * self.waypoints[wp_index][1]
                    self.waypoint_status[wp_index] = True
                    self.flash_waypoints.add(wp_index)
                    self.flash_colors[wp_index] = (0, 255, 0)  # Green for positive
                else:
                    # Negative reward for revisiting
                    reward -= 2
                    self.flash_waypoints.add(wp_index)
                    self.flash_colors[wp_index] = (255, 0, 0)  # Red for negative

        # Small negative reward to encourage forward motion
        reward -= 0.1

        self.total_reward += reward
        return reward

    def _train(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * dones)

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def draw(self, screen):
        super().draw(screen)
        # Draw waypoints
        for i, (wp_pos, _) in enumerate(self.waypoints):
            if i in self.flash_waypoints:
                color = self.flash_colors[i]
            elif self.waypoint_status[i]:
                color = (255, 255, 0)  # Yellow for negative reward potential
            else:
                color = (0, 0, 255)  # Blue for positive reward potential
            pygame.draw.circle(screen, color, (int(wp_pos[0]), int(wp_pos[1])), 3)

        # Draw activation radius (for debugging)
        pygame.draw.circle(screen, (255, 255, 255), (int(self.pos[0]), int(self.pos[1])),
                           self.activation_radius, 1)