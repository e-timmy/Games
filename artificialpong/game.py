import pygame
from paddle import Paddle
from ball import Ball
from ai_player import AIPlayer
from reinforcement_ai import RLPlayer


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
                    print(self.text)
                    self.active = False
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, self.color)

    def update(self):
        width = max(200, self.txt_surface.get_width() + 10)
        self.rect.w = width

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
        self.ai_player = AIPlayer(self.right_paddle, difficulty)
        self.rl_player = RLPlayer(self.left_paddle)
        self.running = True
        self.last_hit_by_ai = False
        self.rl_score = 0
        self.ai_score = 0
        self.last_ball_y = self.ball.rect.y
        self.ball_just_reset = True
        self.speed_multiplier = 1
        self.input_box = InputBox(550, 550, 140, 32)
        self.apply_button = Button(700, 550, 80, 32)

        self.games_played = 0
        self.rl_wins = 0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            self.input_box.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.apply_button.rect.collidepoint(event.pos):
                    try:
                        self.speed_multiplier = float(self.input_box.text)
                        print(f"Speed multiplier set to: {self.speed_multiplier}")
                    except ValueError:
                        print("Invalid input. Speed multiplier not changed.")

    def update(self):
        print(f"Ball position: ({self.ball.rect.x}, {self.ball.rect.y})")
        print(f"Ball just reset: {self.ball_just_reset}")
        print(f"Last hit by AI: {self.last_hit_by_ai}")

        state = self.rl_player.get_state(self.ball, self.right_paddle)

        up, down = self.rl_player.move(self.ball, self.right_paddle)
        moved_towards_ball = (up and self.ball.rect.y < self.left_paddle.rect.centery) or \
                             (down and self.ball.rect.y > self.left_paddle.rect.centery)
        self.left_paddle.move(up, down)

        if self.ball_just_reset:
            print("AI using reset_position() due to ball reset")
            up, down = self.ai_player.reset_position()
            self.ball_just_reset = False
        elif self.last_hit_by_ai:
            print("AI using reset_position() due to last hit")
            up, down = self.ai_player.reset_position()
        else:
            print("AI using regular move()")
            up, down = self.ai_player.move(self.ball)
        self.right_paddle.move(up, down)

        self.ball.move()

        scored = False
        old_rl_score = self.rl_score
        old_ai_score = self.ai_score

        if self.ball.rect.right >= 800:
            print("Ball passed right boundary")
            self.rl_score += 1
            scored = True
            print(f"Score update - RL: {old_rl_score} -> {self.rl_score}, AI: {old_ai_score} -> {self.ai_score}")
        elif self.ball.rect.left <= 0:
            print("Ball passed left boundary")
            self.ai_score += 1
            scored = True
            print(f"Score update - RL: {old_rl_score} -> {self.rl_score}, AI: {old_ai_score} -> {self.ai_score}")

        collision = self.ball.check_collision(self.left_paddle, self.right_paddle)

        hit_ball = collision == 'left'
        missed = self.ball.rect.left <= 0
        lost = missed
        won = self.ball.rect.right >= 800

        if collision == 'right':
            self.last_hit_by_ai = True
            print("Ball hit AI paddle")
        elif collision == 'left':
            self.last_hit_by_ai = False
            print("Ball hit RL paddle")

        reward = self.rl_player.calculate_reward(hit_ball, missed, lost, won, moved_towards_ball, self.ball)
        next_state = self.rl_player.get_state(self.ball, self.right_paddle)
        self.rl_player.remember(state, up - down + 1, reward, next_state, lost or won)
        self.rl_player.train()

        if scored:
            print("Resetting ball due to score")
            self.ball.reset()
            self.ball_just_reset = True
            self.last_hit_by_ai = False
            self.games_played += 1
            if won:
                self.rl_wins += 1

            # Update curriculum every 100 games
            if self.games_played % 100 == 0:
                performance = self.rl_wins / 100
                self.rl_player.update_curriculum(performance)
                self.rl_wins = 0
                print(
                    f"Games played: {self.games_played}, Performance: {performance:.2f}, Training stage: {self.rl_player.training_stage}")

        self.last_ball_y = self.ball.rect.y
        self.input_box.update()

        if scored:
            print("Resetting ball due to score")
            self.ball.reset()
            self.ball_just_reset = True
            self.last_hit_by_ai = False

        self.last_ball_y = self.ball.rect.y
        self.input_box.update()

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.left_paddle.draw(self.screen)
        self.right_paddle.draw(self.screen)
        self.ball.draw(self.screen)

        font = pygame.font.Font(None, 36)
        rl_score_text = font.render(f"RL: {self.rl_score}", True, (255, 255, 255))
        ai_score_text = font.render(f"AI: {self.ai_score}", True, (255, 255, 255))
        self.screen.blit(rl_score_text, (50, 50))
        self.screen.blit(ai_score_text, (700, 50))

        epsilon_text = font.render(f"Epsilon: {self.rl_player.epsilon:.4f}", True, (255, 255, 255))
        self.screen.blit(epsilon_text, (300, 50))

        speed_text = font.render(f"Speed: {self.speed_multiplier}x", True, (255, 255, 255))
        self.screen.blit(speed_text, (50, 550))

        self.input_box.draw(self.screen)
        self.apply_button.draw(self.screen)

        # Draw training stage and games played
        stage_text = font.render(f"Stage: {self.rl_player.training_stage}", True, (255, 255, 255))
        games_text = font.render(f"Games: {self.games_played}", True, (255, 255, 255))
        self.screen.blit(stage_text, (300, 10))
        self.screen.blit(games_text, (300, 90))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            for _ in range(int(self.speed_multiplier)):
                self.update()
            self.draw()
            self.clock.tick(60)
            print("---")

        print(f"Final Score - RL: {self.rl_score}, AI: {self.ai_score}")