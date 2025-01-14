import random

import pygame
import math
from bubble import Bubble
from bullet_bubble import BulletBubble, PropulsionBubble
from settings import (PLAYER_ACCELERATION, PLAYER_MAX_SPEED, PLAYER_FRICTION,
                      SCREEN_WIDTH, SCREEN_HEIGHT, BULLET_SIZE, SHOOT_COOLDOWN, PROPULSION_MASS_RATIO, MIN_PLAYER_SIZE,
                      PROPULSION_BUBBLE_COUNT, MIN_PROPULSION_SIZE, MAX_PROPULSION_SIZE, MIN_BULLET_SIZE,
                      MAX_BULLET_SIZE, BULLET_SIZE_VARIANCE, SHOT_MASS_COST_RATIO)


class Player(Bubble):
    def __init__(self, x, y, size):
        super().__init__(x, y, size, 0, 0)
        self.acceleration = 0
        self.rotation = 0
        self.color = (0, 255, 0)  # Green color for the player
        self.last_shot_time = 0

    def thrust(self):
        angle = math.radians(self.rotation)
        # Apply thrust
        self.dx += PLAYER_ACCELERATION * math.cos(angle)
        self.dy += PLAYER_ACCELERATION * math.sin(angle)

        # Limit speed
        speed = math.sqrt(self.dx ** 2 + self.dy ** 2)
        if speed > PLAYER_MAX_SPEED:
            self.dx = (self.dx / speed) * PLAYER_MAX_SPEED
            self.dy = (self.dy / speed) * PLAYER_MAX_SPEED

        # Create propulsion bubbles
        propulsion_bubbles = []
        total_propulsion_mass = self.size * PROPULSION_MASS_RATIO

        if self.size - total_propulsion_mass > MIN_PLAYER_SIZE:
            self.size -= total_propulsion_mass
            mass_per_bubble = total_propulsion_mass / PROPULSION_BUBBLE_COUNT

            for _ in range(PROPULSION_BUBBLE_COUNT):
                size = random.uniform(MIN_PROPULSION_SIZE, MAX_PROPULSION_SIZE)
                # Position slightly behind player
                start_x = self.x - (self.size + size) * math.cos(angle)
                start_y = self.y - (self.size + size) * math.sin(angle)

                bubble = PropulsionBubble(start_x, start_y, size,
                                          self.rotation, self.dx, self.dy)
                propulsion_bubbles.append(bubble)

        return propulsion_bubbles

    def rotate(self, angle):
        self.rotation += angle
        self.rotation %= 360

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time >= SHOOT_COOLDOWN:
            # Calculate bullet size with some variance
            base_bullet_size = random.uniform(MIN_BULLET_SIZE, MAX_BULLET_SIZE)
            bullet_size = base_bullet_size * (1 + random.uniform(-BULLET_SIZE_VARIANCE, BULLET_SIZE_VARIANCE))

            # Calculate mass cost to player
            mass_cost = bullet_size * SHOT_MASS_COST_RATIO

            # Check if player has enough mass
            if self.size - mass_cost > MIN_PLAYER_SIZE:
                # Reduce player size
                self.size -= mass_cost

                # Calculate bullet starting position at edge of player
                angle = math.radians(self.rotation)
                start_x = self.x + (self.size + bullet_size) * math.cos(angle)
                start_y = self.y + (self.size + bullet_size) * math.sin(angle)

                # Create new bullet
                bullet = BulletBubble(start_x, start_y, bullet_size, self.rotation, (255, 255, 0))
                self.last_shot_time = current_time
                return bullet
        return None

    def update(self):
        super().update()
        # Apply friction
        self.dx *= (1 - PLAYER_FRICTION)
        self.dy *= (1 - PLAYER_FRICTION)
        # Wrap around screen edges
        self.x %= SCREEN_WIDTH
        self.y %= SCREEN_HEIGHT

    def draw(self, screen):
        super().draw(screen)
        # Draw a direction indicator
        angle = math.radians(self.rotation)
        end_x = self.x + self.size * math.cos(angle)
        end_y = self.y + self.size * math.sin(angle)
        pygame.draw.line(screen, (255, 255, 255), (self.x, self.y), (end_x, end_y), 2)