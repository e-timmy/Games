import pymunk

from constants.game_constants import FPS


class PhysicsEngine:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)

    def update(self):
        self.space.step(1 / FPS)