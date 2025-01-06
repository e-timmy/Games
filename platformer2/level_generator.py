import pygame
import pymunk
import random
from constants import *


class LevelGenerator:
    def __init__(self, space):
        self.space = space
        self.chunks = {}  # Dictionary to store generated chunks
        self.last_chunk_x = 0
        self.static_body = space.static_body
        self.last_platform_end = 0

    def generate_chunk(self, chunk_x):
        if chunk_x in self.chunks:
            return

        platforms = []
        x = max(chunk_x * CHUNK_WIDTH, self.last_platform_end)

        while x < (chunk_x + 1) * CHUNK_WIDTH:
            # Generate ground or gap
            if random.random() < 0.3:  # 30% chance of gap
                gap_width = random.randint(MIN_GAP_WIDTH, min(MAX_GAP_WIDTH, int(PLAYER_SPEED * 0.5)))
                x += gap_width
            else:
                platform_width = random.randint(MIN_PLATFORM_WIDTH, MAX_PLATFORM_WIDTH)

                # Create ground segment
                ground = pymunk.Segment(
                    self.static_body,
                    (x, SCREEN_HEIGHT - GROUND_HEIGHT),
                    (x + platform_width, SCREEN_HEIGHT - GROUND_HEIGHT),
                    PLATFORM_THICKNESS
                )
                ground.friction = 1.0
                self.space.add(ground)
                platforms.append(ground)

                # Possibly add elevated platform
                if random.random() < 0.3:  # 30% chance of elevated platform
                    platform_y = random.randint(
                        SCREEN_HEIGHT - GROUND_HEIGHT - MAX_PLATFORM_HEIGHT,
                        SCREEN_HEIGHT - GROUND_HEIGHT - MIN_PLATFORM_HEIGHT
                    )
                    platform = pymunk.Segment(
                        self.static_body,
                        (x, platform_y),
                        (x + platform_width / 2, platform_y),
                        PLATFORM_THICKNESS
                    )
                    platform.friction = 1.0
                    self.space.add(platform)
                    platforms.append(platform)

                x += platform_width

        self.chunks[chunk_x] = platforms
        self.last_platform_end = x

    def update(self, player_x):
        # Generate chunks ahead of the player
        current_chunk = int(player_x // CHUNK_WIDTH)
        for i in range(current_chunk - 1, current_chunk + 3):
            self.generate_chunk(i)

        # Remove chunks far behind the player
        for chunk_x in list(self.chunks.keys()):
            if chunk_x < current_chunk - 2:
                for platform in self.chunks[chunk_x]:
                    self.space.remove(platform)
                del self.chunks[chunk_x]

    def draw(self, screen, camera_offset):
        for platforms in self.chunks.values():
            for platform in platforms:
                pygame.draw.line(
                    screen,
                    GROUND_COLOR,
                    (platform.a.x + camera_offset[0], platform.a.y + camera_offset[1]),
                    (platform.b.x + camera_offset[0], platform.b.y + camera_offset[1]),
                    int(platform.radius * 2)
                )