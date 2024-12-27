import pygame

from open_learner import OpenLearner
from paddle import Paddle
from ball import Ball
from rl_player import RLPlayer
from ai_player import AIPlayer
from game_state import GameState


class InputBox:
    def __init__(self, x, y, w, h, text='1'):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = pygame.Color('lightskyblue3')
        self.text = text
        self.font = pygame.font.Font(None, 32)
        self.txt_surface = self.font.render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = pygame.Color('dodgerblue2') if self.active else pygame.Color('lightskyblue3')
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    return True
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, self.color)
        return False

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        pygame.draw.rect(screen, self.color, self.rect, 2)


class Button:
    def __init__(self, x, y, w, h, text='Apply'):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = pygame.Color('lightskyblue3')
        self.text = text
        self.font = pygame.font.Font(None, 32)
        self.txt_surface = self.font.render(text, True, self.color)

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        pygame.draw.rect(screen, self.color, self.rect, 2)


class PongGame:
    def __init__(self, difficulty, config):
        self.config = config
        self.left_paddle = Paddle(20, 300, 'left')
        self.right_paddle = Paddle(760, 300, 'right')
        self.ball = Ball(400, 300)
        self.game_state = GameState(self.left_paddle, self.right_paddle, self.ball)

        self.rl_player = RLPlayer(self.left_paddle, config)
        self.ai_player = AIPlayer(self.right_paddle, "Easy")

        self.speed_multiplier = 1
        self.input_box = InputBox(300, 550, 140, 32)
        self.apply_button = Button(450, 550, 80, 32)

    def handle_events(self, event):
        if self.input_box.handle_event(event):
            try:
                self.speed_multiplier = float(self.input_box.text)
            except ValueError:
                pass
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.apply_button.rect.collidepoint(event.pos):
                try:
                    self.speed_multiplier = float(self.input_box.text)
                except ValueError:
                    pass

    def update(self):
        for _ in range(int(self.speed_multiplier)):
            # RL player action
            action = self.rl_player.act(self.game_state)

            # AI player action
            self.ai_player.act(self.game_state)

            # Update game state
            ball_hit, game_over, won = self.game_state.update()

            # Calculate reward and update RL player
            next_state = None if game_over else self.game_state
            reward = self.rl_player.calculate_reward(self.game_state, action, ball_hit, game_over, won)
            self.rl_player.update(self.game_state, action, next_state, reward, game_over)

            if game_over:
                self.game_state.reset()

    def render(self, screen):
        screen.fill((0, 0, 0))
        self.left_paddle.draw(screen)
        self.right_paddle.draw(screen)
        self.ball.draw(screen)

        font = pygame.font.Font(None, 36)
        left_score_text = font.render(f"RL Wins: {self.game_state.left_score}", True, (255, 255, 255))
        right_score_text = font.render(f"AI Wins: {self.game_state.right_score}", True, (255, 255, 255))
        games_text = font.render(f"Games: {self.game_state.games_played}", True, (255, 255, 255))
        speed_text = font.render(f"Speed Multiplier:", True, (255, 255, 255))

        screen.blit(left_score_text, (50, 50))
        screen.blit(right_score_text, (650, 50))
        screen.blit(games_text, (350, 50))
        screen.blit(speed_text, (100, 550))

        self.input_box.draw(screen)
        self.apply_button.draw(screen)