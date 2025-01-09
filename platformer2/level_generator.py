import random
from constants import *
from enemy import Enemy
from pform import Platform
from falling_platform import FallingPlatform
from moving_platform import MovingPlatform

class LevelGenerator:
    def __init__(self, space):
        self.space = space
        self.chunks = {}
        self.last_chunk_x = 0
        self.last_platform_end = 0
        self.enemies = []

    def generate_chunk(self, chunk_x):
        if chunk_x in self.chunks:
            return

        platforms = []
        x = max(chunk_x * CHUNK_WIDTH, self.last_platform_end)

        # Ensure the first platform is standard and covers the screen
        if chunk_x == 0:
            initial_platform = Platform(self.space, SCREEN_WIDTH / 2, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH)
            platforms.append(initial_platform)
            x = SCREEN_WIDTH

        while x < (chunk_x + 1) * CHUNK_WIDTH:
            gap_width = random.randint(MIN_GAP_WIDTH, MAX_GAP_WIDTH)
            platform_width = random.randint(MIN_PLATFORM_WIDTH, MAX_PLATFORM_WIDTH)
            platform_y = random.randint(
                SCREEN_HEIGHT - MAX_PLATFORM_HEIGHT,
                SCREEN_HEIGHT - MIN_PLATFORM_HEIGHT
            )

            platform = self.create_platform(x + gap_width, platform_y, platform_width)
            platforms.append(platform)

            x += gap_width + platform_width

        self.chunks[chunk_x] = platforms
        self.last_platform_end = x

        # After generating platforms, add enemies:
        for platform in platforms:
            if random.random() < ENEMY_SPAWN_PROBABILITY:  # Add this constant to constants.py
                enemy = Enemy(platform)
                self.enemies.append(enemy)

    def create_platform(self, x, y, width):
        platform_type = random.choices([PLATFORM_REGULAR, PLATFORM_FALLING, PLATFORM_MOVING],
                                       weights=[0.7, 0.15, 0.15])[0]

        if platform_type == PLATFORM_REGULAR:
            return Platform(self.space, x + width / 2, y, width)
        elif platform_type == PLATFORM_FALLING:
            return FallingPlatform(self.space, x + width / 2, y, width)
        else:  # PLATFORM_MOVING
            return MovingPlatform(self.space, x + width / 2, y, width)

    def update(self, player_x, dt):
        current_chunk = int(player_x // CHUNK_WIDTH)
        for i in range(current_chunk - 1, current_chunk + 3):
            self.generate_chunk(i)

        for chunk_x in list(self.chunks.keys()):
            if chunk_x < current_chunk - 2:
                for platform in self.chunks[chunk_x]:
                    platform.remove(self.space)
                del self.chunks[chunk_x]

        for platforms in self.chunks.values():
            for platform in platforms:
                platform.update(dt)

        # Update enemies
        player_pos = (player_x, self.space.bodies[0].position.y)  # Assuming first body is player
        self.enemies = [e for e in self.enemies if not e.destroyed]
        for enemy in self.enemies:
            enemy.update(player_pos, dt)

    def draw(self, screen, camera_offset):
        for platforms in self.chunks.values():
            for platform in platforms:
                platform.draw(screen, camera_offset)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(screen, camera_offset)