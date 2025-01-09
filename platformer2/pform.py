import pygame
import pymunk
from constants import *

class Platform:
    def __init__(self, space, x, y, width):
        self.width = width
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = (x, y)
        self.shape = pymunk.Segment(self.body, (-width/2, 0), (width/2, 0), PLATFORM_THICKNESS)
        self.shape.collision_type = COLLISION_TYPE_PLATFORM
        self.shape.friction = 1.0
        self.shape.parent = self
        space.add(self.body, self.shape)

    def update(self, dt):
        pass

    def draw(self, screen, camera_offset):
        start = (self.body.position.x - self.width/2 + camera_offset[0], self.body.position.y + camera_offset[1])
        end = (self.body.position.x + self.width/2 + camera_offset[0], self.body.position.y + camera_offset[1])
        pygame.draw.line(screen, PLATFORM_COLOR, start, end, int(PLATFORM_THICKNESS * 2))
        pygame.draw.line(screen, GROUND_COLOR, start, end, 2)

    def remove(self, space):
        space.remove(self.shape, self.body)