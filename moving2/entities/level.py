import pygame
import pymunk
from entities.wall import Wall
from entities.item import Item
from entities.platform import Platform
from systems.powerup_system import PowerUpType
from constants.game_constants import *


class Level:
    def __init__(self, space, level_number):
        self.space = space
        self.level_number = level_number
        self.x_offset = level_number * WINDOW_WIDTH

        self.walls = []
        self.items = []
        self.platforms = []
        self.blocking_wall = None
        self.descending_wall = None

        self.wall_height = 0
        self.target_wall_height = WINDOW_HEIGHT
        self.wall_descent_speed = 5
        self.wall_descending = False
        self.wall_ascending = False
        self.current_wall_height = WINDOW_HEIGHT

        self.create_boundaries()
        self.generate_items()
        self.create_blocking_wall()
        self.create_platforms()

        # Add this new condition for Level 0
        if self.level_number == 0:
            self.create_initial_wall()

    def create_initial_wall(self):
        self.descending_wall = Wall(
            self.space,
            (self.x_offset, 0),  # At the left edge of Level 0
            (self.x_offset, WINDOW_HEIGHT)  # Full height
        )


    def create_boundaries(self):
        floor = Wall(self.space,
                     (self.x_offset, WINDOW_HEIGHT - WALL_THICKNESS / 2),
                     (self.x_offset + WINDOW_WIDTH, WINDOW_HEIGHT - WALL_THICKNESS / 2))
        ceiling = Wall(self.space,
                       (self.x_offset, WALL_THICKNESS / 2),
                       (self.x_offset + WINDOW_WIDTH, WALL_THICKNESS / 2))

        self.walls.extend([floor, ceiling])

    def generate_items(self):
        default_pos = (self.x_offset + WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

        if self.level_number == 0:
            self.items.append(Item(self.space, default_pos, PowerUpType.JUMP))
        elif self.level_number == 1:
            self.items.append(Item(self.space, default_pos, PowerUpType.SHOOT))
        elif self.level_number == 2:
            self.items.append(Item(self.space,
                                   (self.x_offset + WINDOW_WIDTH - 100, 100),
                                   PowerUpType.GRAVITY))
        elif self.level_number == 3:
            self.items.append(Item(self.space, default_pos, PowerUpType.AOE))
        elif self.level_number == 4:
            self.items.append(Item(self.space,
                                   (self.x_offset + WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100),
                                   PowerUpType.JUMP))
        else:
            self.items.append(Item(self.space, default_pos, PowerUpType.JUMP))

    def create_platforms(self):
        if self.level_number == 4:
            platform = Platform(self.space,
                                (self.x_offset + WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2),
                                200, 20)
            self.platforms.append(platform)

    def create_blocking_wall(self):
        if self.blocking_wall:
            self.space.remove(self.blocking_wall.shape, self.blocking_wall.body)

        self.blocking_wall = Wall(
            self.space,
            (self.x_offset + WINDOW_WIDTH, 0),
            (self.x_offset + WINDOW_WIDTH, WINDOW_HEIGHT)
        )

    def start_wall_ascent(self):
        self.wall_ascending = True
        self.wall_descending = False
        self.current_wall_height = WINDOW_HEIGHT

    def start_wall_descent(self):
        self.wall_descending = True
        self.wall_ascending = False
        self.wall_height = 0

    def update_descending_wall(self):
        if self.descending_wall:
            self.space.remove(self.descending_wall.shape, self.descending_wall.body)

        self.descending_wall = Wall(
            self.space,
            (self.x_offset + WINDOW_WIDTH, 0),
            (self.x_offset + WINDOW_WIDTH, self.wall_height)
        )

    def update(self):
        if self.wall_descending and self.wall_height < self.target_wall_height:
            self.wall_height += self.wall_descent_speed
            self.update_descending_wall()

        if self.wall_ascending and self.current_wall_height > 0:
            self.current_wall_height -= self.wall_descent_speed
            if self.current_wall_height <= 0:
                if self.blocking_wall:
                    self.space.remove(self.blocking_wall.shape, self.blocking_wall.body)
                    self.blocking_wall = None
                self.wall_ascending = False
            self.update_blocking_wall()

    def cleanup(self, space):
        for wall in self.walls:
            space.remove(wall.shape, wall.body)
        for item in self.items:
            if not item.collected:
                space.remove(item.shape, item.body)
        if self.blocking_wall:
            space.remove(self.blocking_wall.shape, self.blocking_wall.body)
        if self.descending_wall:
            space.remove(self.descending_wall.shape, self.descending_wall.body)
        for platform in self.platforms:
            space.remove(platform.shape, platform.body)

    def create_descending_wall(self):
        if self.descending_wall:
            self.space.remove(self.descending_wall.shape, self.descending_wall.body)

        self.descending_wall = Wall(
            self.space,
            (self.x_offset, 0),
            (self.x_offset, self.wall_height)
        )

    def update_blocking_wall(self):
        if self.blocking_wall:
            self.space.remove(self.blocking_wall.shape, self.blocking_wall.body)

        if self.current_wall_height > 0:
            self.blocking_wall = Wall(
                self.space,
                (self.x_offset + WINDOW_WIDTH, 0),
                (self.x_offset + WINDOW_WIDTH, self.current_wall_height)
            )

    def draw(self, screen, camera):
        for wall in self.walls:
            wall.draw(screen, camera)

        for item in self.items:
            if not item.collected:
                item.draw(screen, camera)

        for platform in self.platforms:
            platform.draw(screen, camera)

        if self.blocking_wall:
            self.blocking_wall.draw(screen, camera)
        if self.descending_wall:
            self.descending_wall.draw(screen, camera)

        font = pygame.font.Font(None, 36)
        level_text = font.render(f"Level {self.level_number}", True, WHITE)
        text_pos = camera.apply(self.x_offset + 50, 50)
        if text_pos[0] > -1000:
            screen.blit(level_text, text_pos)