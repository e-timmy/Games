import pymunk
import pygame
from constants.game_constants import *


class Wall:
    def __init__(self, space, start_pos, end_pos, thickness=WALL_THICKNESS):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.thickness = thickness

        # Create physics body
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)

        # Create shape
        self.shape = pymunk.Segment(
            self.body,
            start_pos,
            end_pos,
            thickness / 2
        )
        self.shape.friction = 1.0
        self.shape.elasticity = 0.5
        self.shape.collision_type = 3

        space.add(self.body, self.shape)

    def draw(self, screen, camera):
        start = camera.apply(self.start_pos[0], self.start_pos[1])
        end = camera.apply(self.end_pos[0], self.end_pos[1])

        if start[0] > -1000 or end[0] > -1000:
            pygame.draw.line(screen, GRAY, start, end, int(self.thickness))