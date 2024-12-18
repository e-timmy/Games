import pygame
from config import *


class Animation(pygame.sprite.Sprite):
    def __init__(self, centerx, centery, animation_type):
        super().__init__()
        self.animation_type = animation_type
        self.frame = 0
        self.last_update = pygame.time.get_ticks()
        self.frame_rate = 50  # milliseconds

        if animation_type == "enemy_explosion":
            self.frames = self.create_enemy_explosion_frames()
        elif animation_type == "player_death":
            self.frames = self.create_player_death_frames()

        self.image = self.frames[0]
        self.rect = self.image.get_rect()
        self.rect.centerx = centerx
        self.rect.centery = centery

    def create_enemy_explosion_frames(self):
        frames = []
        sizes = [20, 25, 30, 25, 20, 15]
        colors = [(255, 255, 0), (255, 165, 0), (255, 69, 0), (255, 0, 0), (139, 0, 0), (69, 0, 0)]

        for size, color in zip(sizes, colors):
            surf = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (20, 20), size)
            frames.append(surf)

        return frames

    def create_player_death_frames(self):
        frames = []
        sizes = [30, 40, 50, 60, 70, 60, 50, 40, 30]
        colors = [(255, 255, 255), (255, 255, 0), (255, 165, 0), (255, 69, 0),
                  (255, 0, 0), (255, 69, 0), (255, 165, 0), (139, 0, 0), (69, 0, 0)]

        for size, color in zip(sizes, colors):
            surf = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (50, 50), size)
            for i in range(4):  # Draw "spikes" of explosion
                angle = i * 90
                end_x = 50 + size * pygame.math.Vector2(1, 0).rotate(angle).x
                end_y = 50 + size * pygame.math.Vector2(1, 0).rotate(angle).y
                pygame.draw.line(surf, color, (50, 50), (end_x, end_y), 3)
            frames.append(surf)

        return frames

    def update(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.frame_rate:
            self.last_update = now
            self.frame += 1
            if self.frame == len(self.frames):
                self.kill()
            else:
                self.image = self.frames[self.frame]