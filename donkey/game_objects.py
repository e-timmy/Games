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
        self.shape.collision_type = 1

        space.add(self.body, self.shape)

    def draw(self, screen):
        start_pos = (int(self.p1[0]), int(self.p1[1]))
        end_pos = (int(self.p2[0]), int(self.p2[1]))

        # Calculate the angle of the platform
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])

        # Draw main platform
        pygame.draw.line(screen, (100, 100, 100), start_pos, end_pos, self.thickness)

        # Draw top edge highlight
        highlight_offset = int(math.sin(angle) * (self.thickness // 2))
        shadow_offset = int(math.sin(angle) * (self.thickness // 2))

        highlight_start = (start_pos[0], start_pos[1] - highlight_offset)
        highlight_end = (end_pos[0], end_pos[1] - highlight_offset)
        pygame.draw.line(screen, (160, 160, 160), highlight_start, highlight_end, 2)

        # Draw bottom edge shadow
        shadow_start = (start_pos[0], start_pos[1] + shadow_offset)
        shadow_end = (end_pos[0], end_pos[1] + shadow_offset)
        pygame.draw.line(screen, (60, 60, 60), shadow_start, shadow_end, 2)


class Ladder:
    def __init__(self, x, y1, y2):
        self.x = x
        self.y1 = min(y1, y2)
        self.y2 = max(y1, y2)
        self.width = 30

    def draw(self, screen):
        pygame.draw.line(screen, (139, 69, 19), (self.x, self.y1), (self.x, self.y2), 3)
        pygame.draw.line(screen, (139, 69, 19), (self.x + self.width, self.y1), (self.x + self.width, self.y2), 3)

        rung_spacing = 20
        for y in range(int(self.y1), int(self.y2), rung_spacing):
            pygame.draw.line(screen, (165, 82, 22), (self.x, y), (self.x + self.width, y), 2)

    def contains_point(self, x, y):
        return (self.x <= x <= self.x + self.width and
                self.y1 <= y <= self.y2)


class Ball:
    def __init__(self, space, position, size=15):
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
        pygame.draw.circle(screen, (200, 0, 0), pos, int(self.radius))
        pygame.draw.circle(screen, (255, 0, 0), pos, int(self.radius - 2))


def create_ball(space, size=15):
    return Ball(space, (850, 30), size)