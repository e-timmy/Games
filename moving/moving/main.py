import math

import pygame
import pymunk
import sys

from camera import Camera
from constants import *
from player import Player
from game_objects import Item, PowerUpType, PlayerState
from level_manager import LevelManager


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = (0, 981)

        self.camera = Camera(WINDOW_WIDTH, WINDOW_WIDTH)
        self.player = Player(self.space, (WINDOW_WIDTH // 2, 50))
        self.level_manager = LevelManager(self.space)

        self.setup_collisions()

    def setup_collisions(self):
        def collect_item(arbiter, space, data):
            item_shape = arbiter.shapes[1]
            item = self.level_manager.get_item_from_shape(item_shape)
            if item and not item.collected:
                self.player.powerup_manager.add_powerup(item.powerup_type)
                item.collected = True
            return True

        def bullet_collect_item(arbiter, space, data):
            item_shape = arbiter.shapes[1]
            item = self.level_manager.get_item_from_shape(item_shape)
            if item and not item.collected:
                self.player.powerup_manager.add_powerup(item.powerup_type)
                item.collected = True
            return True

        # Player collecting item
        handler = self.space.add_collision_handler(1, 2)  # Player: 1, Item: 2
        handler.begin = collect_item

        # Bullet collecting item
        bullet_handler = self.space.add_collision_handler(4, 2)  # Bullet: 4, Item: 2
        bullet_handler.begin = bullet_collect_item

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.player.start_aiming()
                    elif event.key == pygame.K_j:
                        self.player.jump()
                    elif event.key == pygame.K_s:
                        self.player.shoot()
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        self.state = PlayerState.FALLING

            self.space.step(1 / FPS)
            self.player.update_aim()
            self.player.update_state()
            self.player.update_bullets()

            # Check for first landing in starting level
            self.level_manager.check_first_landing(self.player.state.name)

            # Check for level completion and camera transition
            if self.level_manager.check_level_complete(self.player.body.position):
                self.camera.start_transition()

            # Update level transition and wall descent
            self.level_manager.update_transition()
            self.camera.update()

            # Draw
            self.screen.fill(BLACK)
            self.player.draw(self.screen, self.camera)
            self.level_manager.draw_current_level(self.screen, self.camera)

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    game = Game()
    game.run()