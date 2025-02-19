import pygame
from settings import *
from level_generator import LevelGenerator, MovingPlatform

class Level:
    def __init__(self):
        self.generator = LevelGenerator()
        self.platforms = []
        self.moving_platforms = []
        self.generate_new_level()

    def generate_new_level(self):
        self.platforms = self.generator.generate_level()
        self.moving_platforms = [p for p in self.platforms if isinstance(p, MovingPlatform)]
        self.generator.increase_difficulty()

    def update(self):
        for platform in self.moving_platforms:
            platform.update()

    def get_platforms(self):
        return self.platforms

    def draw(self, screen):
        for platform in self.platforms:
            # Platform base
            pygame.draw.rect(screen, PLATFORM_COLOR, platform)
            # Platform glow
            pygame.draw.rect(screen, (120, 100, 180), platform, 2)