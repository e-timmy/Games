import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from base_car import BaseCar
import math
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)


class ReinforcementLearner(BaseCar):
    def __init__(self, pos, color, number, track):
        super().__init__(pos, color, number)
        print("Initializing Reinforcement Learner")
        self.track = track

        # State space definition
        self.state_size = 12
        self.action_size = 9

        # Initialize device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Adjusted exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Increased minimum exploration
        self.epsilon_decay = 0.9999  # Slower decay
        self.gamma = 0.99

        # Additional training metrics
        self.episode_steps = 0
        self.episode_collisions = 0
        self.episode_checkpoints = 0
        self.best_episode_reward = float('-inf')
        self.episode_history = []
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'checkpoints_per_episode': [],
            'collisions_per_episode': [],
            'exploration_rates': []
        }

        # Logging frequency
        self.log_frequency = 100  # steps
        self.last_log_step = 0

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # Training metrics
        self.total_reward = 0
        self.episode_reward = 0
        self.episodes = 0
        self.steps = 0

        # Objective tracking
        self.last_checkpoint_time = 0
        self.last_position = list(pos)  # Ensure it's a list
        self.last_checkpoint = None
        self.checkpoint_times = []
        self.stuck_counter = 0
        self.last_progress = 0

        # Ray casting parameters
        self.ray_step = 5
        self.max_ray_distance = 300

        # Enhanced statistics tracking
        self.stats = {
            'current_episode': {
                'steps': 0,
                'reward': 0,
                'checkpoints_hit': 0,
                'collisions': 0,
                'laps_completed': 0,
                'last_checkpoint': None,
                'running_reward': []  # Last N step rewards
            },
            'history': {
                'episode_rewards': [],
                'episode_steps': [],
                'episode_checkpoints': [],
                'episode_collisions': [],
                'episode_laps': [],
                'running_rewards': []  # Average reward per N steps
            },
            'window_size': 100,  # For running averages
            'log_frequency': 100  # Steps between logs
        }

    def get_distance_to_wall(self, angle):
        """Cast a ray to find distance to wall"""
        absolute_angle = (self.angle + angle) % 360
        rad_angle = math.radians(absolute_angle)

        dx = math.cos(rad_angle) * self.ray_step
        dy = math.sin(rad_angle) * self.ray_step

        current_x = float(self.pos[0])
        current_y = float(self.pos[1])
        distance = 0

        while distance < self.max_ray_distance:
            current_x += dx
            current_y += dy
            distance += self.ray_step

            if not self.track.is_on_track((int(current_x), int(current_y)), 1):
                return distance

        return self.max_ray_distance

    def get_state(self):
        try:
            # Get distances to walls in 8 directions
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            distances = []
            for angle in angles:
                distance = self.get_distance_to_wall(angle)
                distances.append(min(distance / self.max_ray_distance, 1.0))

            # Current speed (normalized)
            speed = math.hypot(*self.vel) / self.max_speed

            # Current angle (normalized)
            angle_norm = (self.angle % 360) / 360.0

            # Distance and angle to next checkpoint
            if self.next_checkpoint < len(self.track.checkpoints):
                next_cp = self.track.checkpoints[self.next_checkpoint]
                dx = next_cp.centerx - self.pos[0]
                dy = next_cp.centery - self.pos[1]
                distance_to_cp = math.hypot(dx, dy)
                angle_to_cp = math.degrees(math.atan2(dy, dx)) - self.angle
                angle_to_cp = (angle_to_cp + 180) % 360 - 180
            else:
                distance_to_cp = self.max_ray_distance
                angle_to_cp = 0

            state = distances + [
                speed,
                angle_norm,
                min(distance_to_cp / 500.0, 1.0),
                angle_to_cp / 180.0
            ]
            return torch.FloatTensor(state).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error in get_state: {e}")
            return torch.zeros((1, self.state_size)).to(self.device)

    def get_reward(self):
        try:
            reward = 0

            # Adjusted reward weights
            speed = math.hypot(*self.vel)
            speed_reward = (speed / self.max_speed) * 0.2  # Increased speed reward
            reward += speed_reward

            # Progress reward
            if hasattr(self, 'last_position'):
                dx = self.pos[0] - self.last_position[0]
                dy = self.pos[1] - self.last_position[1]
                movement = math.hypot(dx, dy)

                if self.next_checkpoint < len(self.track.checkpoints):
                    next_cp = self.track.checkpoints[self.next_checkpoint]
                    cp_dx = next_cp.centerx - self.last_position[0]
                    cp_dy = next_cp.centery - self.last_position[1]

                    if movement > 0 and math.hypot(cp_dx, cp_dy) > 0:
                        dot_product = (dx * cp_dx + dy * cp_dy) / (movement * math.hypot(cp_dx, cp_dy))
                        progress_reward = movement * max(0, dot_product) * 0.05  # Increased progress reward
                        reward += progress_reward

            # Checkpoint reward
            if self.current_checkpoint != self.last_checkpoint:
                if self.current_checkpoint == self.next_checkpoint:
                    reward += 20.0  # Increased checkpoint reward
                    self.episode_checkpoints += 1
                    self.checkpoint_times.append(self.steps - self.last_checkpoint_time)
                    self.last_checkpoint_time = self.steps
                else:
                    reward -= 10.0
                self.last_checkpoint = self.current_checkpoint

            # Lap completion reward
            if self.lap_valid and self.current_checkpoint == 0:
                reward += 100.0  # Increased lap reward
                self._log_episode_completion()

            # Collision penalty
            if self.track.get_reflection_vector(self.pos, self.radius) != (0, 0):
                reward -= 2.0  # Increased collision penalty
                self.episode_collisions += 1

            # Stuck penalty
            if movement < 0.1:
                self.stuck_counter += 1
                if self.stuck_counter > 60:
                    reward -= 10.0  # Increased stuck penalty
            else:
                self.stuck_counter = 0

            self.last_position = list(self.pos)
            return reward
        except Exception as e:
            print(f"Error in get_reward: {e}")
            return 0.0

    def handle_input(self, game_state):
        if not game_state.race_started:
            return

        self.steps += 1
        self.episode_steps += 1

        # Periodic logging
        if self.steps - self.last_log_step >= self.log_frequency:
            self._log_training_progress()
            self.last_log_step = self.steps

        state = self.get_state()

        # Epsilon-greedy with annealed exploration
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                action = self.model(state).argmax().item()

        self._execute_action(action)

        reward = self.get_reward()
        next_state = self.get_state()

        self.memory.append((state, action, reward, next_state))

        if len(self.memory) >= self.batch_size:
            self._learn()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_reward += reward
        self.total_reward += reward

    def _execute_action(self, action):
        try:
            if action == 0:  # No action
                pass
            elif action == 1:  # Accelerate
                self.accelerate()
                self.accelerate()
            elif action == 2:  # Brake
                self.decelerate()
                self.decelerate()
            elif action == 3:  # Turn Left
                self.turn(-self.turn_speed * 1.5)
            elif action == 4:  # Turn Right
                self.turn(self.turn_speed * 1.5)
            elif action == 5:  # Accelerate + Turn Left
                self.accelerate()
                self.turn(-self.turn_speed * 1.5)
            elif action == 6:  # Accelerate + Turn Right
                self.accelerate()
                self.turn(self.turn_speed * 1.5)
            elif action == 7:  # Brake + Turn Left
                self.decelerate()
                self.turn(-self.turn_speed * 1.5)
            elif action == 8:  # Brake + Turn Right
                self.decelerate()
                self.turn(self.turn_speed * 1.5)
        except Exception as e:
            print(f"Error in _execute_action: {e}")

    def _learn(self):
        try:
            if len(self.memory) < self.batch_size:
                return

            batch = random.sample(self.memory, self.batch_size)
            states = torch.cat([s[0] for s in batch])
            actions = torch.tensor([s[1] for s in batch], device=self.device)
            rewards = torch.tensor([s[2] for s in batch], device=self.device, dtype=torch.float32)
            next_states = torch.cat([s[3] for s in batch])

            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.model(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values)

            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            print(f"Error in _learn: {e}")

    def load_model(self):
        try:
            if os.path.exists('reinforcement_learner.pth'):
                self.model.load_state_dict(torch.load('reinforcement_learner.pth', map_location=self.device))
                self.epsilon = self.epsilon_min
                print("Loaded existing model")
        except Exception as e:
            print(f"Error loading model: {e}")

    def _log_episode_completion(self):
        self.episodes += 1
        self.best_episode_reward = max(self.best_episode_reward, self.episode_reward)

        # Store episode statistics
        self.training_stats['episode_rewards'].append(self.episode_reward)
        self.training_stats['episode_lengths'].append(self.episode_steps)
        self.training_stats['checkpoints_per_episode'].append(self.episode_checkpoints)
        self.training_stats['collisions_per_episode'].append(self.episode_collisions)
        self.training_stats['exploration_rates'].append(self.epsilon)

        print(f"\nEpisode {self.episodes} completed:")
        print(f"Steps: {self.episode_steps}")
        print(f"Reward: {self.episode_reward:.2f}")
        print(f"Checkpoints: {self.episode_checkpoints}")
        print(f"Collisions: {self.episode_collisions}")
        print(f"Exploration rate: {self.epsilon:.3f}")

        # Reset episode-specific counters
        self.episode_steps = 0
        self.episode_reward = 0
        self.episode_checkpoints = 0
        self.episode_collisions = 0

    def _log_training_progress(self):
        if self.episodes > 0:
            avg_reward = sum(self.training_stats['episode_rewards'][-10:]) / min(10, len(
                self.training_stats['episode_rewards']))
            avg_checkpoints = sum(self.training_stats['checkpoints_per_episode'][-10:]) / min(10, len(
                self.training_stats['checkpoints_per_episode']))

            print(f"\nTraining Progress (Step {self.steps}):")
            print(f"Last 10 episodes avg reward: {avg_reward:.2f}")
            print(f"Last 10 episodes avg checkpoints: {avg_checkpoints:.1f}")
            print(f"Current exploration rate: {self.epsilon:.3f}")

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), 'reinforcement_learner.pth')

            print("\nTraining Summary:")
            print(f"Total episodes: {self.episodes}")
            print(f"Total steps: {self.steps}")
            print(f"Total reward: {self.total_reward:.2f}")
            print(f"Best episode reward: {self.best_episode_reward:.2f}")

            if self.episodes > 0:
                print("\nAverages per episode:")
                print(f"Steps: {sum(self.training_stats['episode_lengths']) / self.episodes:.1f}")
                print(f"Reward: {sum(self.training_stats['episode_rewards']) / self.episodes:.2f}")
                print(f"Checkpoints: {sum(self.training_stats['checkpoints_per_episode']) / self.episodes:.1f}")
                print(f"Collisions: {sum(self.training_stats['collisions_per_episode']) / self.episodes:.1f}")

                print("\nLearning progression:")
                episodes_chunks = min(5, self.episodes)
                chunk_size = max(1, self.episodes // episodes_chunks)
                for i in range(episodes_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, self.episodes)
                    avg_reward = sum(self.training_stats['episode_rewards'][start_idx:end_idx]) / (end_idx - start_idx)
                    print(f"Episodes {start_idx}-{end_idx - 1}: Avg reward = {avg_reward:.2f}")

        except Exception as e:
            print(f"Error saving model: {e}")