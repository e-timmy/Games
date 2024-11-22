import pymunk
import pygame
import math


class Platform:
    def __init__(self, space, p1, p2, thickness=20):
        self.p1 = p1
        self.p2 = p2
        self.thickness = int(thickness)

        physics_thickness = thickness / 2

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, p1, p2, physics_thickness)
        self.shape.friction = 0.7
        self.shape.elasticity = 0.2
        self.shape.collision_type = 1  # Platform collision type

        space.add(self.body, self.shape)

    def draw(self, screen):
        start_pos = (int(self.p1[0]), int(self.p1[1]))
        end_pos = (int(self.p2[0]), int(self.p2[1]))

        pygame.draw.line(screen, (0, 200, 0), start_pos, end_pos, self.thickness)
        pygame.draw.line(screen, (0, 255, 0), start_pos, end_pos, 2)


class Ladder:
    def __init__(self, x, y1, y2):
        self.x = x
        self.y1 = min(y1, y2)  # Ensure y1 is the top of the ladder
        self.y2 = max(y1, y2)  # Ensure y2 is the bottom of the ladder
        self.width = 30

    def draw(self, screen):
        # Draw vertical lines
        pygame.draw.line(screen, (139, 69, 19), (self.x, self.y1), (self.x, self.y2), 3)
        pygame.draw.line(screen, (139, 69, 19), (self.x + self.width, self.y1), (self.x + self.width, self.y2), 3)

        # Draw rungs
        rung_spacing = 20
        for y in range(int(self.y1), int(self.y2), rung_spacing):
            pygame.draw.line(screen, (139, 69, 19), (self.x, y), (self.x + self.width, y), 2)

    def contains_point(self, x, y):
        return (self.x <= x <= self.x + self.width and
                self.y1 <= y <= self.y2)


class Ball:
    def __init__(self, space, position, size=15):  # Increased default size
        self.radius = size
        self.mass = 8.0
        moment = pymunk.moment_for_circle(self.mass, 0, self.radius)
        self.body = pymunk.Body(self.mass, moment)
        self.body.position = position

        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0.3
        self.shape.friction = 0.7
        self.shape.collision_type = 2

        gravity_force = 1.2
        self.body.apply_force_at_local_point((0, self.mass * 900 * gravity_force), (0, 0))

        space.add(self.body, self.shape)

    def draw(self, screen):
        pos = int(self.body.position.x), int(self.body.position.y)
        pygame.draw.circle(screen, (200, 140, 0), pos, int(self.radius))
        pygame.draw.circle(screen, (255, 165, 0), pos, int(self.radius - 2))


def create_ball(space, size=15):  # Updated default size
    return Ball(space, (850, 30), size)