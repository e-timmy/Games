import pygame
import math
from bubble import Bubble
from settings import PLAYER_ACCELERATION, PLAYER_MAX_SPEED, PLAYER_FRICTION, SCREEN_WIDTH, SCREEN_HEIGHT

class Player(Bubble):
    def __init__(self, x, y, size):
        super().__init__(x, y, size, 0, 0)
        self.acceleration = 0
        self.rotation = 0
        self.color = (0, 255, 0)  # Green color for the player

    def rotate(self, angle):
        self.rotation += angle
        self.rotation %= 360

    def thrust(self):
        # Convert rotation to radians
        angle = math.radians(self.rotation)
        # Calculate acceleration vector
        self.dx += PLAYER_ACCELERATION * math.cos(angle)
        self.dy += PLAYER_ACCELERATION * math.sin(angle)
        # Limit speed
        speed = math.sqrt(self.dx**2 + self.dy**2)
        if speed > PLAYER_MAX_SPEED:
            self.dx = (self.dx / speed) * PLAYER_MAX_SPEED
            self.dy = (self.dy / speed) * PLAYER_MAX_SPEED

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