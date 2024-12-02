import pygame
import pymunk
import math

class Car:
    def __init__(self, x, y, space):
        self.width = 10
        self.height = 20
        mass = 1
        moment = pymunk.moment_for_box(mass, (self.width, self.height))
        self.body = pymunk.Body(mass, moment)
        self.body.position = x, y
        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))
        self.shape.elasticity = 0.1
        self.shape.friction = 0.5
        space.add(self.body, self.shape)

        self.acceleration = 500
        self.turn_speed = 3
        self.max_speed = 300

    def handle_input(self, keys):
        force = 0
        if keys[pygame.K_UP]:
            force = self.acceleration
        elif keys[pygame.K_DOWN]:
            force = -self.acceleration

        if keys[pygame.K_LEFT]:
            self.body.angle -= math.radians(self.turn_speed)
        elif keys[pygame.K_RIGHT]:
            self.body.angle += math.radians(self.turn_speed)

        fx = math.sin(self.body.angle) * force
        fy = -math.cos(self.body.angle) * force
        self.body.apply_force_at_local_point((fx, fy), (0, 0))

        # Limit speed
        velocity = self.body.velocity
        speed = velocity.length
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.body.velocity = velocity.x * scale, velocity.y * scale

    def draw_direction_indicator(self, screen):
        start_pos = self.body.position
        end_pos = (
            start_pos.x + math.sin(self.body.angle) * 30,
            start_pos.y - math.cos(self.body.angle) * 30
        )
        pygame.draw.line(screen, (255, 0, 0), start_pos, end_pos, 2)