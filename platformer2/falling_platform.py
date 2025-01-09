import pygame

from pform import Platform
from constants import *

class FallingPlatform(Platform):
    def __init__(self, space, x, y, width):
        super().__init__(space, x, y, width)
        self.timer = 0
        self.falling = False
        self.activated = False

    def activate(self):
        self.activated = True

    def update(self, dt):
        if self.activated and not self.falling:
            self.timer += dt
            if self.timer >= FALLING_PLATFORM_DELAY:
                self.falling = True
        if self.falling:
            self.body.velocity = (0, FALLING_PLATFORM_SPEED)

    def draw(self, screen, camera_offset):
        start = (self.body.position.x - self.width/2 + camera_offset[0], self.body.position.y + camera_offset[1])
        end = (self.body.position.x + self.width/2 + camera_offset[0], self.body.position.y + camera_offset[1])
        pygame.draw.line(screen, FALLING_PLATFORM_COLOR, start, end, int(PLATFORM_THICKNESS * 2))