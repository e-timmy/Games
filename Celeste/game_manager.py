import pygame
import random
from settings import *

class GameManager:
    def __init__(self):
        self.sparkles = [(random.randint(0, SCREEN_WIDTH),
                          random.randint(0, SCREEN_HEIGHT))
                         for _ in range(50)]
        self.win = False
        self.win_zone = pygame.Rect(SCREEN_WIDTH - 80, 0, 80, 100)

    def check_win_condition(self, player):
        if self.win_zone.colliderect(player.rect):
            self.win = True
            return True
        return False

    def draw_effects(self, screen):
        # Update and draw sparkles
        for i, (x, y) in enumerate(self.sparkles):
            # Sparkle movement
            self.sparkles[i] = ((x + random.randint(-1, 1)) % SCREEN_WIDTH,
                                (y + random.randint(-1, 1)) % SCREEN_HEIGHT)

            # Draw sparkle
            size = random.randint(1, 3)
            alpha = random.randint(50, 200)
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 255, 255, alpha), (size, size), size)
            screen.blit(surf, (x - size, y - size))

        # Draw win zone with magical effect
        if not self.win:
            pygame.draw.rect(screen, (147, 0, 211, 100), self.win_zone)
            # Add pulsing effect
            glow = abs(pygame.time.get_ticks() % 1000 - 500) / 500
            pygame.draw.rect(screen, (200, 100, 255), self.win_zone,
                             max(1, int(glow * 4)))