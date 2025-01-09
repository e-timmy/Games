import pygame
import random
from constants import *


class Enemy:
    def __init__(self, platform):
        self.platform = platform
        self.x = platform.body.position.x + random.randint(-platform.width // 4, platform.width // 4)
        self.y = platform.body.position.y - ENEMY_HEIGHT
        self.width = ENEMY_WIDTH
        self.height = ENEMY_HEIGHT
        self.speed = ENEMY_PATROL_SPEED
        self.direction = 1 if random.random() > 0.5 else -1
        self.state = "patrolling"
        self.charging_speed = ENEMY_CHARGE_SPEED
        self.destroyed = False
        self.falling = False
        self.last_charge_direction = 0  # To remember direction after player leaves

    def update(self, player_pos, dt):
        # Update position based on platform
        self.y = self.platform.body.position.y - ENEMY_HEIGHT

        # If the platform is falling, the enemy falls with it
        if hasattr(self.platform, 'falling') and self.platform.falling:
            self.falling = True
            return

        # If enemy is falling, just update y position
        if self.falling:
            self.y += GRAVITY * dt
            if self.y > SCREEN_HEIGHT + 100:  # Destroy if off screen
                self.destroyed = True
            return

        platform_left = self.platform.body.position.x - self.platform.width / 2
        platform_right = self.platform.body.position.x + self.platform.width / 2

        if self.state == "patrolling":
            new_x = self.x + self.speed * self.direction * dt

            # Ensure enemy stays on platform
            if new_x - self.width / 2 <= platform_left:
                self.x = platform_left + self.width / 2
                self.direction *= -1
            elif new_x + self.width / 2 >= platform_right:
                self.x = platform_right - self.width / 2
                self.direction *= -1
            else:
                self.x = new_x

            # Check if player is on the platform
            if (abs(player_pos[1] - self.platform.body.position.y) < PLAYER_HEIGHT / 2 + 5 and
                    platform_left < player_pos[0] < platform_right):
                self.state = "charging"
                self.last_charge_direction = 1 if player_pos[0] > self.x else -1

        elif self.state == "charging":
            # Check if player is still on the platform
            player_on_platform = (abs(player_pos[1] - self.platform.body.position.y) < PLAYER_HEIGHT / 2 + 5 and
                                  platform_left < player_pos[0] < platform_right)

            if player_on_platform:
                self.last_charge_direction = 1 if player_pos[0] > self.x else -1

            new_x = self.x + self.charging_speed * self.last_charge_direction * dt

            # If we've reached the end of the platform, switch back to patrolling
            if (self.last_charge_direction == 1 and new_x + self.width / 2 > platform_right) or \
                    (self.last_charge_direction == -1 and new_x - self.width / 2 < platform_left):
                self.x = platform_right - self.width / 2 if self.last_charge_direction == 1 else platform_left + self.width / 2
                self.state = "patrolling"
                self.direction = -self.last_charge_direction  # Turn around when reaching the end
            else:
                self.x = new_x

    def check_player_collision(self, player):
        player_rect = pygame.Rect(
            player.body.position.x - PLAYER_WIDTH / 2,
            player.body.position.y - (PLAYER_DUCKED_HEIGHT if player.ducked else PLAYER_HEIGHT) / 2,
            PLAYER_WIDTH,
            PLAYER_DUCKED_HEIGHT if player.ducked else PLAYER_HEIGHT
        )

        enemy_rect = pygame.Rect(
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.width,
            self.height
        )

        if player_rect.colliderect(enemy_rect):
            if player.ducked and player.body.velocity.y > 0:  # Player is ducking and falling
                self.destroyed = True
                return "enemy_destroyed"
            else:
                return "player_death"
        return None

    def draw(self, screen, camera_offset):
        rect = pygame.Rect(
            self.x - self.width / 2 + camera_offset[0],
            self.y - self.height / 2 + camera_offset[1],
            self.width,
            self.height
        )
        pygame.draw.rect(screen, ENEMY_COLOR, rect)

        # Draw enemy "eye"
        eye_size = min(self.width, self.height) // 3
        eye_rect = pygame.Rect(
            rect.centerx - eye_size // 2,
            rect.centery - eye_size // 2,
            eye_size,
            eye_size
        )
        pygame.draw.rect(screen, ENEMY_EYE_COLOR, eye_rect)

        # Draw glowing outline
        pygame.draw.rect(screen, ENEMY_GLOW_COLOR, rect, 2)