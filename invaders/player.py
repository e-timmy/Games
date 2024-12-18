import pygame
import random
from config import *
from bullet import Bullet


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = self.create_player_image()
        self.image = self.original_image
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = PLAYER_START_Y
        self.speed = PLAYER_SPEED
        self.is_respawning = False
        self.respawn_timer = 0
        self.last_blink = 0
        self.visible = True

    def create_player_image(self):
        # Create a more detailed player ship
        surf = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT), pygame.SRCALPHA)

        # Main body (triangle shape)
        pygame.draw.polygon(surf, GREEN, [
            (PLAYER_WIDTH // 2, 0),  # Top point
            (0, PLAYER_HEIGHT - 5),  # Bottom left
            (PLAYER_WIDTH, PLAYER_HEIGHT - 5)  # Bottom right
        ])

        # Cockpit
        pygame.draw.circle(surf, (0, 200, 0), (PLAYER_WIDTH // 2, PLAYER_HEIGHT // 2 - 5), 4)

        # Wings
        pygame.draw.polygon(surf, GREEN, [
            (5, PLAYER_HEIGHT - 15),
            (0, PLAYER_HEIGHT - 5),
            (15, PLAYER_HEIGHT - 5)
        ])
        pygame.draw.polygon(surf, GREEN, [
            (PLAYER_WIDTH - 5, PLAYER_HEIGHT - 15),
            (PLAYER_WIDTH, PLAYER_HEIGHT - 5),
            (PLAYER_WIDTH - 15, PLAYER_HEIGHT - 5)
        ])

        # Engine glow
        pygame.draw.rect(surf, (0, 200, 0),
                         (PLAYER_WIDTH // 2 - 5, PLAYER_HEIGHT - 5, 10, 5))

        return surf

    def draw_ship(self):
        # Draw the ship body
        pygame.draw.rect(self.image, GREEN, (0, PLAYER_HEIGHT // 2, PLAYER_WIDTH, PLAYER_HEIGHT // 2))

        # Draw the ship point
        pygame.draw.polygon(self.image, GREEN, [
            (0, PLAYER_HEIGHT // 2),
            (PLAYER_WIDTH // 2, 0),
            (PLAYER_WIDTH, PLAYER_HEIGHT // 2)
        ])

    def respawn(self):
        pos = random.choice(RESPAWN_POSITIONS)
        self.rect.centerx = pos[0]
        self.rect.bottom = PLAYER_START_Y  # Update this line
        self.is_respawning = True
        self.respawn_timer = pygame.time.get_ticks()
        self.visible = True

    def update(self):
        current_time = pygame.time.get_ticks()

        # Handle respawn blinking
        if self.is_respawning:
            if current_time - self.respawn_timer > RESPAWN_DURATION:
                self.is_respawning = False
                self.visible = True
            elif current_time - self.last_blink > RESPAWN_BLINK_TIME:
                self.visible = not self.visible
                self.last_blink = current_time

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < SCREEN_WIDTH:
            self.rect.x += self.speed

    def draw(self, surface):
        if self.visible:
            surface.blit(self.image, self.rect)

    def shoot(self):
        return Bullet(self.rect.centerx, self.rect.top)