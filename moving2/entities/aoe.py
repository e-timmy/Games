import pygame
import pymunk
import math
from constants.game_constants import *


class AOEEffect:
    def __init__(self, space, center_pos, max_radius=200, duration=60):
        self.space = space
        self.center_pos = center_pos
        self.max_radius = max_radius
        self.current_radius = 0
        self.duration = duration
        self.current_frame = 0
        self.ripples = []

    def update(self):
        self.current_frame += 1
        self.current_radius = self.max_radius * (self.current_frame / self.duration)

        if self.current_frame % 5 == 0:  # Create a new ripple every 5 frames
            self.ripples.append((self.current_radius, 1.0))  # Add with full opacity

        # Update ripples and remove those that have faded out
        self.ripples = [(r, max(0, o - 0.05)) for r, o in self.ripples if o > 0]

    def is_finished(self):
        return self.current_frame >= self.duration

    def draw(self, screen, camera):
        center = camera.apply(self.center_pos.x, self.center_pos.y)

        for ripple, opacity in self.ripples:
            color = (0, 0, 255, int(opacity * 255))  # Blue color with fading opacity
            pygame.draw.circle(screen, color, center, int(ripple), 2)

    def check_collision(self, point):
        distance = math.sqrt((point[0] - self.center_pos.x) ** 2 + (point[1] - self.center_pos.y) ** 2)
        return distance <= self.current_radius