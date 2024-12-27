import pygame
import torch
import torch.nn as nn
import random
import copy

from ball import Ball
from game_state import GameState
from paddle import Paddle
from player import Player


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def mutate(self, mutation_power=0.1):
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * mutation_power
                param.add_(noise)


class EvolutionaryLearner(Player):
    def __init__(self, paddle):
        super().__init__(paddle)
        self.brain = SimpleNet()
        self.games_won = 0
        self.total_games = 0

    def act(self, game_state):
        state = torch.FloatTensor(game_state.get_state())
        with torch.no_grad():
            outputs = self.brain(state)
            action = outputs.argmax().item()

        self.paddle.move(action == 0, action == 2)
        return action

    def update(self, game_state, action, next_state, reward, done):
        if done:
            if reward > 0:  # Won the game
                self.games_won += 1
            self.total_games += 1

    def create_offspring(self):
        child = EvolutionaryLearner(self.paddle)
        child.brain = copy.deepcopy(self.brain)
        child.brain.mutate()
        return child


class EvolutionaryGame:
    def __init__(self, config):
        self.config = config
        self.left_paddle = Paddle(20, 300, 'left')
        self.right_paddle = Paddle(760, 300, 'right')
        self.ball = Ball(400, 300)
        self.game_state = GameState(self.left_paddle, self.right_paddle, self.ball)

        self.population_size = 4
        self.population = [EvolutionaryLearner(self.left_paddle) for _ in range(self.population_size)]
        self.current_left = 0
        self.current_right = 1
        self.generation = 0

    def handle_events(self, event):
        pass

    def update(self):
        left_player = self.population[self.current_left]
        right_player = self.population[self.current_right]

        # Both players act
        left_action = left_player.act(self.game_state)
        right_action = right_player.act(self.game_state)

        # Update game state
        ball_hit, game_over, left_won = self.game_state.update()

        # If game is over, update players and move to next matchup
        if game_over:
            reward = 1 if left_won else -1
            left_player.update(self.game_state, left_action, None, reward, True)
            right_player.update(self.game_state, right_action, None, -reward, True)

            self.game_state.reset()
            self.next_matchup()

    def next_matchup(self):
        self.current_right += 1
        if self.current_right >= self.population_size:
            self.current_left += 1
            self.current_right = self.current_left + 1

        if self.current_left >= self.population_size - 1:
            self.evolve_population()

    def evolve_population(self):
        self.generation += 1
        # Sort population by win rate
        self.population.sort(key=lambda x: x.games_won / max(1, x.total_games), reverse=True)

        # Keep top half, replace bottom half with mutated versions of top half
        half = self.population_size // 2
        for i in range(half):
            self.population[i + half] = self.population[i].create_offspring()

        # Reset stats for next generation
        for player in self.population:
            player.games_won = 0
            player.total_games = 0

        self.current_left = 0
        self.current_right = 1

        print(f"Generation {self.generation} complete")

    def render(self, screen):
        screen.fill((0, 0, 0))
        self.left_paddle.draw(screen)
        self.right_paddle.draw(screen)
        self.ball.draw(screen)

        font = pygame.font.Font(None, 36)
        gen_text = font.render(f"Generation: {self.generation}", True, (255, 255, 255))
        matchup_text = font.render(f"Matchup: {self.current_left} vs {self.current_right}", True, (255, 255, 255))

        screen.blit(gen_text, (50, 50))
        screen.blit(matchup_text, (50, 100))