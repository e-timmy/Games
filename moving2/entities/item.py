import pymunk
import pygame
from constants.game_constants import *


class Item:
    def __init__(self, space, pos, powerup_type):
        self.powerup_type = powerup_type
        self.collected = False

        # Create physics body
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = pos

        # Create shape
        self.shape = pymunk.Poly.create_box(self.body, (ITEM_SIZE, ITEM_SIZE))
        self.shape.sensor = True
        self.shape.collision_type = 2

        space.add(self.body, self.shape)

    def draw(self, screen, camera):
        if not self.collected:
            pos = camera.apply(self.body.position.x, self.body.position.y)
            if pos[0] > -1000:
                pygame.draw.rect(screen, GREEN,
                                 (pos[0] - ITEM_SIZE / 2,
                                  pos[1] - ITEM_SIZE / 2,
                                  ITEM_SIZE, ITEM_SIZE))