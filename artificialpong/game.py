import pygame
from paddle import Paddle
from ball import Ball
from ai_player import AIPlayer
from rl_player import RLPlayer, GameState


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


class Game:
    def __init__(self, difficulty):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.left_paddle = Paddle(20, 300, 'left')
        self.right_paddle = Paddle(760, 300, 'right')
        self.ball = Ball(400, 300)
        self.rl_player = RLPlayer(self.left_paddle)
        self.ai_player = AIPlayer(self.right_paddle, difficulty)
        self.running = True
        self.left_wins = 0
        self.right_wins = 0
        self.games_played = 0
        self.speed_multiplier = 1

        # Speed control UI
        self.input_box = InputBox(300, 550, 140, 32)
        self.apply_button = Button(450, 550, 80, 32)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
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
        # Get current game state
        current_state = GameState(self.left_paddle, self.right_paddle, self.ball)

        # RL agent action
        action = self.rl_player.act(current_state)
        self.rl_player.paddle.move(action == 0, action == 2)

        # AI movement
        up, down = self.ai_player.move(self.ball)
        self.right_paddle.move(up, down)

        # Ball movement and collision detection
        self.ball.move()
        ball_hit = self.ball.check_collision(self.left_paddle, self.right_paddle) == 'left'

        game_over = False
        won = False

        # Score and game counting
        if self.ball.rect.right >= 800:
            self.left_wins += 1
            self.games_played += 1
            game_over = True
            won = True
        elif self.ball.rect.left <= 0:
            self.right_wins += 1
            self.games_played += 1
            game_over = True
            won = False

        next_state = GameState(self.left_paddle, self.right_paddle, self.ball) if not game_over else None
        reward = self.rl_player.calculate_reward(current_state, action, ball_hit, game_over, won)
        self.rl_player.update(current_state, action, next_state, reward, game_over)

        if game_over:
            self.ball.reset()

            # Save model every 100 games and print progress
            if self.games_played % 100 == 0:
                self.rl_player.save()
                win_rate = self.left_wins / self.games_played
                print(f"Games: {self.games_played}, Win rate: {win_rate:.2f}")

    def run(self):
        while self.running:
            self.handle_events()
            for _ in range(int(self.speed_multiplier)):
                self.update()
            self.draw()
            self.clock.tick(60)

    def __del__(self):
        # Save the model when the game closes
        self.rl_player.save()

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.left_paddle.draw(self.screen)
        self.right_paddle.draw(self.screen)
        self.ball.draw(self.screen)

        font = pygame.font.Font(None, 36)
        left_score_text = font.render(f"RL Wins: {self.left_wins}", True, (255, 255, 255))
        right_score_text = font.render(f"AI Wins: {self.right_wins}", True, (255, 255, 255))
        games_text = font.render(f"Games: {self.games_played}", True, (255, 255, 255))
        speed_text = font.render(f"Speed Multiplier:", True, (255, 255, 255))

        self.screen.blit(left_score_text, (50, 50))
        self.screen.blit(right_score_text, (650, 50))
        self.screen.blit(games_text, (350, 50))
        self.screen.blit(speed_text, (100, 550))

        self.input_box.draw(self.screen)
        self.apply_button.draw(self.screen)

        pygame.display.flip()