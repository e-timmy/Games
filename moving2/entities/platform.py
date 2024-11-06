import pymunk
import pygame
from constants.game_constants import *

class Platform:
    def __init__(self, space, pos, width, height):
        self.width = width
        self.height = height

        # Create physics body
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = pos

        # Create shape
        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.friction = 1.0
        self.shape.elasticity = 0.2
        self.shape.collision_type = 5  # New collision type for platforms

        space.add(self.body, self.shape)

    def draw(self, screen, camera):
        pos = camera.apply(self.body.position.x, self.body.position.y)
        if pos[0] > -1000:
            pygame.draw.rect(screen, GRAY,
                             (pos[0] - self.width/2, pos[1] - self.height/2,
                              self.width, self.height))