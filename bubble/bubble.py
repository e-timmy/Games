import pygame
import random
import math
from colors import BUBBLE_COLORS
from settings import (SCREEN_WIDTH, SCREEN_HEIGHT, WOBBLE_AMOUNT,
                      MAX_BUBBLE_SIZE, ABSORPTION_RATE)

class Bubble:
    def __init__(self, x, y, size, dx, dy):
        self.x = x
        self.y = y
        self.size = size
        self.dx = dx
        self.dy = dy
        self.color = random.choice(BUBBLE_COLORS)
        self.wobble_offset = random.uniform(0, 2 * math.pi)
        self.wobble_speed = random.uniform(1, 2)

    def update(self):
        # Add slight wobble to movement
        time = pygame.time.get_ticks() / 1000
        wobble = math.sin(time * self.wobble_speed + self.wobble_offset) * WOBBLE_AMOUNT

        # Update position with bouncing off walls
        self.x += self.dx + wobble
        self.y += self.dy + wobble

        # Bounce off walls
        if self.x - self.size <= 0 or self.x + self.size >= SCREEN_WIDTH:
            self.dx *= -1
        if self.y - self.size <= 0 or self.y + self.size >= SCREEN_HEIGHT:
            self.dy *= -1

        # Keep bubble in bounds
        self.x = max(self.size, min(SCREEN_WIDTH - self.size, self.x))
        self.y = max(self.size, min(SCREEN_HEIGHT - self.size, self.y))

    def collides_with(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < (self.size + other.size)

    def absorb(self, other):
        if self.size > other.size:
            size_diff = self.size - other.size
            absorption_amount = min(ABSORPTION_RATE, size_diff, other.size)
            self.size += absorption_amount
            other.size -= absorption_amount
            self.size = min(self.size, MAX_BUBBLE_SIZE)  # Limit maximum size

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.size))
        # Add a slight highlight for depth
        highlight_pos = (int(self.x - self.size * 0.3), int(self.y - self.size * 0.3))
        highlight_size = int(self.size * 0.2)
        pygame.draw.circle(screen, (255, 255, 255), highlight_pos, highlight_size)