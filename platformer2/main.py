# main.py
import pygame
import pymunk
import sys
from player import Player
from camera import Camera
from level_generator import LevelGenerator
from game_state import GameState
from constants import *


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Infinite Platformer")

        # Initialize Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)

        # Initialize game components
        self.game_state = GameState()
        self.player = Player(self.space, START_X, START_Y)
        self.camera = Camera()
        self.level_generator = LevelGenerator(self.space)

        self.clock = pygame.time.Clock()
        self.game_over = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.player.jump()
                elif event.key == pygame.K_DOWN:
                    self.player.duck()
                elif event.key == pygame.K_r and self.game_over:
                    self.reset_game()
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    self.player.stand()

        # Continuous movement
        if not self.game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.player.move_left()
            elif keys[pygame.K_RIGHT]:
                self.player.move_right()
            else:
                self.player.stop_horizontal_movement()

        return True

    def update(self):
        if not self.game_over:
            self.space.step(1 / FPS)
            self.player.update()
            self.camera.update(self.player.get_position())
            self.level_generator.update(self.player.get_position()[0])
            self.game_state.update_score(self.player.get_position()[0])

            # Check for game over condition
            if self.player.get_position()[1] > SCREEN_HEIGHT + 100:
                self.game_over = True

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)

        # Apply camera offset to all drawn objects
        camera_offset = self.camera.get_offset()

        # Draw level
        self.level_generator.draw(self.screen, camera_offset)

        # Draw player
        self.player.draw(self.screen, camera_offset)

        # Draw score
        score_text = f"Score: {self.game_state.get_score()}"
        font = pygame.font.Font(None, 36)
        text_surface = font.render(score_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            game_over_font = pygame.font.Font(None, 72)
            game_over_text = game_over_font.render("GAME OVER", True, (255, 0, 0))
            restart_text = font.render("Press 'R' to restart", True, (255, 255, 255))
            self.screen.blit(game_over_text,
                             (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 50))

        pygame.display.flip()

    def reset_game(self):
        self.game_over = False
        self.game_state = GameState()
        self.player = Player(self.space, START_X, START_Y)
        self.camera = Camera()
        self.level_generator = LevelGenerator(self.space)

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()