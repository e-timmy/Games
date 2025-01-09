import pygame
import pymunk
import sys

from falling_platform import FallingPlatform
from player import Player
from camera import Camera
from level_generator import LevelGenerator
from game_state import GameState
from background import Background
from constants import *

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Infinite Platformer")

        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)

        self.game_state = GameState()
        self.player = Player(self.space, START_X, START_Y)
        self.camera = Camera()
        self.level_generator = LevelGenerator(self.space)
        self.background = Background()

        self.clock = pygame.time.Clock()
        self.game_over = False

        self.setup_collision_handlers()

        self.level_generator.player = self.player  # Pass player reference to level generator


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
            dt = 1 / FPS
            self.space.step(dt)
            self.player.update()
            self.camera.update(self.player.get_position())
            self.level_generator.update(self.player.get_position()[0], dt)
            self.game_state.update_score(self.player.get_position()[0])

            if self.player.get_position()[1] > SCREEN_HEIGHT + 100:
                self.game_over = True

        for enemy in self.level_generator.enemies:
            result = enemy.check_player_collision(self.player)
            if result == "player_death":
                self.game_over = True
            elif result == "enemy_destroyed":
                # You might want to add a score bonus here
                self.game_state.score += 100


    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)

        camera_offset = self.camera.get_offset()

        self.background.draw(self.screen, camera_offset)
        self.level_generator.draw(self.screen, camera_offset)
        self.player.draw(self.screen, camera_offset)

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
        # Remove all bodies and shapes from the space
        for body in self.space.bodies:
            self.space.remove(body)
        for shape in self.space.shapes:
            self.space.remove(shape)

        # Reset game state
        self.game_over = False
        self.game_state = GameState()
        self.player = Player(self.space, START_X, START_Y)
        self.camera = Camera()
        self.level_generator = LevelGenerator(self.space)  # This will regenerate enemies
        self.background = Background()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    def setup_collision_handlers(self):
        handler = self.space.add_collision_handler(COLLISION_TYPE_PLAYER, COLLISION_TYPE_PLATFORM)
        handler.begin = self.handle_collision

    def handle_collision(self, arbiter, space, data):
        player_shape, platform_shape = arbiter.shapes
        player = player_shape.body
        platform = platform_shape.body

        # Allow player to pass through platform from below
        if player.position.y > platform.position.y and player.velocity.y < 0:
            return False

        # Activate falling platforms when landed on
        if isinstance(platform_shape.parent, FallingPlatform):
            platform_shape.parent.activate()

        return True


if __name__ == "__main__":
    game = Game()
    game.run()