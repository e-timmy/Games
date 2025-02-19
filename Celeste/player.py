import pygame
import math
from settings import *


class Player:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 40, 60)
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.on_wall = False
        self.is_climbing = False
        self.is_boosting = False
        self.boost_timer = 0
        self.climb_timer = 0
        self.boost_direction_x = 0
        self.boost_direction_y = 0
        self.trail = []  # For visual effects
        self.facing_right = True

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and (self.on_ground or self.on_wall):
                self.jump()
            elif event.key == pygame.K_x and not self.is_boosting:
                self.start_boost()

    def update(self, platforms):
        keys = pygame.key.get_pressed()

        # Horizontal movement
        if not self.is_boosting and not self.is_climbing:
            self.vel_x = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * MOVE_SPEED
            if self.vel_x > 0:
                self.facing_right = True
            elif self.vel_x < 0:
                self.facing_right = False

        # Wall climbing
        if keys[pygame.K_c] and self.on_wall and self.climb_timer < CLIMB_TIME_LIMIT:
            self.is_climbing = True
            self.vel_y = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * CLIMB_SPEED
            self.vel_x = 0  # Stop horizontal movement while climbing
            self.climb_timer += 1
        else:
            self.is_climbing = False
            if self.on_ground:
                self.climb_timer = 0  # Reset climb timer when on ground

        # Update boost
        if self.is_boosting:
            self.boost_timer -= 1
            if self.boost_timer <= 0:
                self.is_boosting = False
            else:
                self.vel_x = self.boost_direction_x * BOOST_SPEED
                self.vel_y = self.boost_direction_y * BOOST_SPEED

        # Apply gravity if not climbing or boosting
        if not self.is_climbing and not self.is_boosting:
            self.vel_y += GRAVITY

        # Update position and check collisions
        self.rect.x += self.vel_x
        self.check_collision(platforms, True)

        self.rect.y += self.vel_y
        self.check_collision(platforms, False)

        # Keep player in bounds
        self.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

        # Update trail for visual effect
        self.trail.append((self.rect.centerx, self.rect.centery))
        if len(self.trail) > 10:
            self.trail.pop(0)

    def check_collision(self, platforms, horizontal):
        self.on_wall = False
        for platform in platforms:
            if self.rect.colliderect(platform):
                if horizontal:
                    if self.vel_x > 0:
                        self.rect.right = platform.left
                        self.on_wall = True
                    elif self.vel_x < 0:
                        self.rect.left = platform.right
                        self.on_wall = True
                    self.vel_x = 0
                else:
                    if self.vel_y > 0:
                        self.rect.bottom = platform.top
                        self.on_ground = True
                        self.vel_y = 0
                    elif self.vel_y < 0:
                        self.rect.top = platform.bottom
                        self.vel_y = 0

        # Check if player is touching a wall when not moving horizontally
        if self.vel_x == 0:
            for platform in platforms:
                if self.rect.right == platform.left or self.rect.left == platform.right:
                    self.on_wall = True
                    break

        # Reset on_ground if not colliding with any platform
        if not horizontal:
            self.on_ground = any(self.rect.bottom == p.top for p in platforms)

    def jump(self):
        self.vel_y = JUMP_SPEED
        self.on_ground = False
        self.is_climbing = False

    def start_boost(self):
        keys = pygame.key.get_pressed()

        # Get directional input
        dx = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        dy = keys[pygame.K_DOWN] - keys[pygame.K_UP]

        # If no direction pressed, boost in facing direction or upward
        if dx == 0 and dy == 0:
            dx = 1 if self.facing_right else -1

        # Normalize the direction vector
        length = math.sqrt(dx * dx + dy * dy)
        if length != 0:
            self.boost_direction_x = dx / length
            self.boost_direction_y = dy / length

        self.is_boosting = True
        self.boost_timer = BOOST_DURATION

    def draw(self, screen):
        # Draw trail
        for i, pos in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)))
            surf = pygame.Surface((10, 10), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*PLAYER_COLOR, alpha), (5, 5), 5)
            screen.blit(surf, (pos[0] - 5, pos[1] - 5))

        # Draw player with different colors based on state
        if self.is_boosting:
            color = BOOST_COLOR
        elif self.is_climbing:
            color = CLIMB_COLOR
        else:
            color = PLAYER_COLOR

        pygame.draw.rect(screen, color, self.rect)
        # Add a glow effect
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)