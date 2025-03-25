import pygame
import random
from constants import *

class Player:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH // 2 - PLAYER_SIZE // 2, HEIGHT - GROUND_HEIGHT - PLAYER_SIZE, PLAYER_SIZE,
                                PLAYER_SIZE)
        self.velocity_y = 0
        self.velocity_x = 0
        self.jumping = False
        log_debug(f"Player initialized at y={self.rect.y}")

    def update(self):
        self.velocity_y += GRAVITY
        self.rect.y += self.velocity_y
        self.rect.x += self.velocity_x

        if self.rect.left < 0:
            self.rect.left = 0
        elif self.rect.right > WIDTH:
            self.rect.right = WIDTH

        if self.velocity_y > 0:
            self.jumping = False

    def jump(self):
        self.velocity_y = JUMP_POWER
        self.jumping = True
        log_debug(f"Player jumped. y={self.rect.y}, vy={self.velocity_y}")

    def move_left(self):
        self.velocity_x = -HORIZONTAL_SPEED

    def move_right(self):
        self.velocity_x = HORIZONTAL_SPEED

    def stop_horizontal(self):
        self.velocity_x = 0


class Platform:
    def __init__(self, x, y, width, platform_type=NORMAL_PLATFORM):
        self.rect = pygame.Rect(x, y, width, PLATFORM_HEIGHT)
        self.platform_type = platform_type
        self.color = PLATFORM_COLORS[platform_type]
        self.is_broken = False

        # For moving platforms
        self.direction = random.choice([-1, 1])  # -1 for left, 1 for right
        self.original_x = x

        log_debug(f"Platform created at y={y}, width={width}, type={platform_type}")

    def update(self):
        if self.platform_type == MOVING_PLATFORM and not self.is_broken:
            self.rect.x += self.direction * MOVING_PLATFORM_SPEED

            # Reverse direction when hitting screen edge
            if self.rect.left <= 0:
                self.rect.left = 0
                self.direction = 1
            elif self.rect.right >= WIDTH:
                self.rect.right = WIDTH
                self.direction = -1

    def break_platform(self):
        if self.platform_type == BREAKABLE_PLATFORM:
            self.is_broken = True
            log_debug(f"Platform broken at y={self.rect.y}")