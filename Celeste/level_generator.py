import pygame
import random
from settings import *


class LevelGenerator:
    def __init__(self):
        self.min_platform_length = 100
        self.max_platform_length = 250
        self.platform_height = 20
        self.vertical_spacing = 120
        self.horizontal_spacing = 200
        self.difficulty = 1
        self.max_difficulty = 5

    def generate_level(self):
        platforms = []

        # Ground
        platforms.append(pygame.Rect(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40))

        # Left wall with opening at bottom (constant)
        platforms.append(pygame.Rect(0, 0, 40, SCREEN_HEIGHT - 140))

        # Right wall with opening at top (constant)
        platforms.append(pygame.Rect(SCREEN_WIDTH - 40, 100, 40, SCREEN_HEIGHT - 100))

        # Generate main path platforms
        current_height = SCREEN_HEIGHT - 200
        current_x = 200
        path_platforms = []

        while current_height > 200:
            # Platform length decreases with difficulty
            platform_length = random.randint(
                max(80, self.min_platform_length - self.difficulty * 15),
                max(120, self.max_platform_length - self.difficulty * 20)
            )

            # Ensure platform doesn't exceed screen bounds
            if current_x + platform_length > SCREEN_WIDTH - 100:
                platform_length = SCREEN_WIDTH - current_x - 100

            # Add platform
            new_platform = pygame.Rect(
                current_x,
                current_height,
                platform_length,
                self.platform_height
            )
            path_platforms.append(new_platform)

            # Move up and to the side for next platform
            current_height -= random.randint(
                self.vertical_spacing,
                self.vertical_spacing + self.difficulty * 10
            )

            # Alternate between left and right sides of the screen
            if current_x < SCREEN_WIDTH / 2:
                current_x += random.randint(
                    self.horizontal_spacing,
                    self.horizontal_spacing + self.difficulty * 20
                )
            else:
                current_x = random.randint(100, SCREEN_WIDTH // 2 - 100)

        platforms.extend(path_platforms)

        # Add challenging elements based on difficulty
        self.add_challenging_elements(platforms, path_platforms)

        return platforms

    def add_challenging_elements(self, platforms, path_platforms):
        # Add moving platforms
        num_moving_platforms = min(self.difficulty, 3)
        for _ in range(num_moving_platforms):
            self.add_moving_platform(platforms, path_platforms)

        # Add some smaller platforms for advanced jumping
        num_small_platforms = self.difficulty * 2
        for _ in range(num_small_platforms):
            x = random.randint(100, SCREEN_WIDTH - 150)
            y = random.randint(200, SCREEN_HEIGHT - 200)
            width = random.randint(40, 80)
            small_platform = pygame.Rect(x, y, width, self.platform_height)
            if not any(small_platform.colliderect(p) for p in platforms):
                platforms.append(small_platform)

        # Add some vertical walls for wall jumping (if that mechanic exists)
        num_walls = min(self.difficulty, 3)
        for _ in range(num_walls):
            x = random.randint(100, SCREEN_WIDTH - 100)
            height = random.randint(100, 200)
            y = random.randint(SCREEN_HEIGHT - 400, SCREEN_HEIGHT - height)
            wall = pygame.Rect(x, y, 20, height)
            if not any(wall.colliderect(p) for p in platforms):
                platforms.append(wall)

    def add_moving_platform(self, platforms, path_platforms):
        # Choose a random path platform to replace with a moving platform
        if path_platforms:
            platform_to_replace = random.choice(path_platforms)
            platforms.remove(platform_to_replace)
            path_platforms.remove(platform_to_replace)

            # Create a moving platform with a wider range
            move_range = random.randint(100, 200)
            start_x = max(50, platform_to_replace.x - move_range // 2)
            end_x = min(SCREEN_WIDTH - 50 - platform_to_replace.width, start_x + move_range)

            moving_platform = MovingPlatform(
                start_x, platform_to_replace.y,
                platform_to_replace.width, self.platform_height,
                start_x, end_x
            )
            platforms.append(moving_platform)

    def increase_difficulty(self):
        self.difficulty += 1
        self.difficulty = min(self.difficulty, self.max_difficulty)


class MovingPlatform(pygame.Rect):
    def __init__(self, x, y, width, height, start_x, end_x):
        super().__init__(x, y, width, height)
        self.start_x = start_x
        self.end_x = end_x
        self.speed = random.randint(1, 3)
        self.direction = 1

    def update(self):
        self.x += self.speed * self.direction
        if self.x <= self.start_x or self.x >= self.end_x:
            self.direction *= -1