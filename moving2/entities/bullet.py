import pymunk
import pygame
from constants.game_constants import *


class Bullet:
    def __init__(self, space, pos, direction, speed=500):
        self.space = space

        # Create physics body
        self.body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
        self.body.position = pos
        self.body.velocity_func = lambda body, gravity, damping, dt: (body.velocity.x, body.velocity.y)

        # Create shape
        self.shape = pymunk.Circle(self.body, 5)
        self.shape.collision_type = 4
        self.shape.elasticity = 0.8
        self.shape.friction = 0.5

        # Set velocity
        self.body.velocity = (direction[0] * speed, direction[1] * speed)

        space.add(self.body, self.shape)

    def draw(self, screen, camera):
        pos = camera.apply(self.body.position.x, self.body.position.y)
        if pos[0] > -1000:
            pygame.draw.circle(screen, RED, (int(pos[0]), int(pos[1])), 5)