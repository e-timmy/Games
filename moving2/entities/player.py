import pygame
import pymunk
import math
from entities.game_states import PlayerState
from constants.game_constants import *


class Player:
    def __init__(self, space, pos):
        self.space = space
        self.aim_angle = -math.pi / 2
        self.state = PlayerState.FALLING

        # Create physics body
        mass = 1
        vertices = self._create_semicircle_vertices(PLAYER_SIZE / 2, 10)
        moment = pymunk.moment_for_poly(mass, vertices)

        self.body = pymunk.Body(mass, moment)
        self.body.position = pos

        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.collision_type = 1
        self.shape.friction = 0.7
        self.shape.elasticity = 0.5

        space.add(self.body, self.shape)

    def _create_semicircle_vertices(self, radius, num_segments):
        vertices = []
        vertices.append((-radius, 0))
        vertices.append((radius, 0))

        for i in range(num_segments):
            angle = math.pi * i / (num_segments - 1)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertices.append((x, -y))

        return vertices

    def update_state(self):
        if abs(self.body.velocity.y) < GROUND_THRESHOLD:
            self.state = PlayerState.GROUNDED
        else:
            self.state = PlayerState.FALLING

    def draw(self, screen, camera):
        pos = camera.apply(self.body.position.x, self.body.position.y)
        if pos[0] > -1000:
            # Draw body
            vertices = []
            for v in self.shape.get_vertices():
                x = v.rotated(self.body.angle) + self.body.position
                screen_pos = camera.apply(x.x, x.y)
                vertices.append(screen_pos)
            pygame.draw.polygon(screen, BLUE, vertices)

            # Draw arm with proper offset
            arm_start = (
                pos[0] + math.cos(self.aim_angle) * PLAYER_SIZE * 0.3,
                pos[1] + math.sin(self.aim_angle) * PLAYER_SIZE * 0.3
            )
            arm_end = (
                pos[0] + math.cos(self.aim_angle) * PLAYER_SIZE * 0.8,
                pos[1] + math.sin(self.aim_angle) * PLAYER_SIZE * 0.8
            )
            pygame.draw.line(screen, BLUE, arm_start, arm_end, 3)