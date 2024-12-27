# rl_player.py
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from player import Player


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


class RLPlayer(Player):
    def __init__(self, paddle, config):
        super().__init__(paddle)
        self.memory = deque(maxlen=config.memory_size)
        self.success_memory = deque(maxlen=config.memory_size // 2)
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.save_file = config.save_file

        self.model = DQN(5, 3)
        self.advanced_model = None  # Only created when needed
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.advanced_mode = False
        self.transition_phase = False
        self.win_rates = deque(maxlen=100)
        self.transition_progress = 0.0

        self.load()

    def act(self, game_state):
        state = torch.FloatTensor(game_state.get_state())

        if self.advanced_mode:
            explore_rate = 0.3  # Higher exploration in advanced mode
        else:
            explore_rate = self.epsilon

        if random.random() <= explore_rate:
            return random.randrange(3)

        with torch.no_grad():
            if self.advanced_mode:
                q_values = self.advanced_model(state)
            else:
                q_values = self.model(state)

            action = q_values.argmax().item()

        self.paddle.move(action == 0, action == 2)
        return action

    def calculate_reward(self, game_state, action, ball_hit, game_over, won):
        if self.advanced_mode:
            if game_over:
                return 1 if won else -1
            return 0
        else:
            if ball_hit:
                return 0.1
            elif game_over:
                return 1 if won else -1
            else:
                moved_towards_ball = (
                        (action == 0 and game_state.ball.rect.centery < self.paddle.rect.centery) or
                        (action == 2 and game_state.ball.rect.centery > self.paddle.rect.centery)
                )
                return 0.01 if moved_towards_ball else -0.01

    def update(self, game_state, action, next_state, reward, done):
        current_state = game_state.get_state()
        next_state_data = next_state.get_state() if next_state else [0] * 5

        self.memory.append((current_state, action, reward, next_state_data, done))

        if done:
            game_won = game_state.left_score > game_state.right_score
            self.win_rates.append(1 if game_won else 0)

            if len(self.win_rates) == self.win_rates.maxlen:
                current_win_rate = sum(self.win_rates) / len(self.win_rates)

                if current_win_rate >= 0.6 and not self.advanced_mode:
                    print("Switching to advanced mode!")
                    self.advanced_mode = True
                    self.advanced_model = DQN(5, 3)
                    self.advanced_model.load_state_dict(self.model.state_dict())
                    self.optimizer = optim.Adam(self.advanced_model.parameters(), lr=0.0005)  # Lower learning rate

                    # Store successful experiences for later use
                    recent_experiences = list(self.memory)[-1000:]
                    successful_experiences = [(s, a, r, ns, d) for s, a, r, ns, d in recent_experiences
                                              if r > 0]
                    self.success_memory.extend(successful_experiences)

            self.train()

            if game_state.games_played % 100 == 0:
                self.save()
                win_rate = game_state.left_score / game_state.games_played
                mode = "Advanced" if self.advanced_mode else "Basic"
                print(f"Games: {game_state.games_played}, Win rate: {win_rate:.2f}, Mode: {mode}")

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        if self.advanced_mode:
            # Mix regular and successful experiences in advanced mode
            if len(self.success_memory) > self.batch_size // 2:
                regular_batch = random.sample(self.memory, self.batch_size // 2)
                success_batch = random.sample(self.success_memory, self.batch_size // 2)
                minibatch = regular_batch + success_batch
            else:
                minibatch = random.sample(self.memory, self.batch_size)
        else:
            minibatch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_model = self.advanced_model if self.advanced_mode else self.model

        current_q_values = current_model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = current_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not self.advanced_mode:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'advanced_mode': self.advanced_mode,
        }
        if self.advanced_mode:
            save_dict['advanced_model_state_dict'] = self.advanced_model.state_dict()
        torch.save(save_dict, self.save_file)

    def load(self):
        if os.path.exists(self.save_file):
            checkpoint = torch.load(self.save_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.advanced_mode = checkpoint.get('advanced_mode', False)
            if self.advanced_mode:
                self.advanced_model = DQN(5, 3)
                self.advanced_model.load_state_dict(checkpoint['advanced_model_state_dict'])
            print(f"Loaded model. Epsilon: {self.epsilon:.2f}, Mode: {'Advanced' if self.advanced_mode else 'Basic'}")
        else:
            print("No saved model found. Starting from scratch.")