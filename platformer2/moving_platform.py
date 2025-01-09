import pygame

from pform import Platform
from constants import *

class MovingPlatform(Platform):
    def __init__(self, space, x, y, width):
        super().__init__(space, x, y, width)
        self.start_x = x
        self.direction = 1
        self.speed = MOVING_PLATFORM_SPEED

    def update(self, dt):
        new_x = self.body.position.x + self.direction * self.speed * dt
        if abs(new_x - self.start_x) > MOVING_PLATFORM_RANGE / 2:
            self.direction *= -1
        self.body.velocity = (self.direction * self.speed, 0)

    def draw(self, screen, camera_offset):
        start = (self.body.position.x - self.width/2 + camera_offset[0], self.body.position.y + camera_offset[1])
        end = (self.body.position.x + self.width/2 + camera_offset[0], self.body.position.y + camera_offset[1])
        pygame.draw.line(screen, MOVING_PLATFORM_COLOR, start, end, int(PLATFORM_THICKNESS * 2))