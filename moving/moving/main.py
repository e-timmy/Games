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
        self.level_manager = LevelManager(self.space)
        self.player = Player(self.space, (WINDOW_WIDTH // 2, 50))

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

        def bullet_wall_collision(arbiter, space, data):
            return True  # Return True to allow for bouncing

        # Player collecting item
        handler = self.space.add_collision_handler(1, 2)  # Player: 1, Item: 2
        handler.begin = collect_item

        # Bullet collecting item
        bullet_handler = self.space.add_collision_handler(4, 2)  # Bullet: 4, Item: 2
        bullet_handler.begin = bullet_collect_item

        # Bullet wall collision
        bullet_wall_handler = self.space.add_collision_handler(4, 3)  # Bullet: 4, Wall: 3
        bullet_wall_handler.begin = bullet_wall_collision

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_j:
                        self.player.jump()
                    elif event.key == pygame.K_s:
                        self.player.shoot()
                    elif event.key == pygame.K_g:
                        self.player.toggle_gravity()

            self.space.step(1 / FPS)
            self.player.update_aim()
            self.player.update_state()
            self.update_bullets()  # New method to manage bullets

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
            self.draw_debug()  # Add this line to enable debug drawing

            pygame.display.flip()
            self.clock.tick(FPS)

    def draw_debug(self):
        if not DEBUG_MODE:
            return

        # Draw all physics shapes
        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Segment):
                start = shape.a + shape.body.position
                end = shape.b + shape.body.position
                start_screen = self.camera.apply(start.x, start.y)
                end_screen = self.camera.apply(end.x, end.y)
                if start_screen[0] > -1000 and end_screen[0] > -1000:
                    pygame.draw.line(self.screen, RED, start_screen, end_screen, 1)

            elif isinstance(shape, pymunk.Circle):
                pos = self.camera.apply(shape.body.position.x, shape.body.position.y)
                if pos[0] > -1000:
                    pygame.draw.circle(self.screen, RED,
                                       (int(pos[0]), int(pos[1])),
                                       int(shape.radius), 1)

            elif isinstance(shape, pymunk.Poly):
                vertices = [self.camera.apply(v.x + shape.body.position.x,
                                              v.y + shape.body.position.y)
                            for v in shape.get_vertices()]
                if any(v[0] > -1000 for v in vertices):
                    pygame.draw.polygon(self.screen, RED, vertices, 1)

    def update_bullets(self):
        current_level_offset = self.level_manager.current_level_number * WINDOW_WIDTH
        min_x = current_level_offset - WINDOW_WIDTH
        max_x = current_level_offset + WINDOW_WIDTH * 2

        # Remove bullets that are out of bounds
        self.player.bullets = [
            bullet for bullet in self.player.bullets
            if (min_x <= bullet.body.position.x <= max_x and
                -WINDOW_HEIGHT <= bullet.body.position.y <= WINDOW_HEIGHT * 2)
        ]


if __name__ == "__main__":
    game = Game()
    game.run()