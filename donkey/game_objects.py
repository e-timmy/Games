import pymunk
import pygame
import math


class Platform:
    def __init__(self, space, p1, p2, thickness=20):
        self.p1 = p1
        self.p2 = p2
        # Convert thickness to integer for pygame drawing
        self.thickness = int(thickness)

        # Use original (float) thickness for physics
        physics_thickness = thickness / 2

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, p1, p2, physics_thickness)
        self.shape.friction = 0.7
        self.shape.elasticity = 0.2
        self.shape.collision_type = 1  # Platform collision type

        space.add(self.body, self.shape)

    def draw(self, screen):
        # Convert coordinates to integers for pygame drawing
        start_pos = (int(self.p1[0]), int(self.p1[1]))
        end_pos = (int(self.p2[0]), int(self.p2[1]))

        # Draw main platform line
        pygame.draw.line(screen, (0, 200, 0), start_pos, end_pos, self.thickness)
        # Draw highlight
        pygame.draw.line(screen, (0, 255, 0), start_pos, end_pos, 2)

    def draw(self, screen):
        # Convert positions to integers for drawing
        start_pos = (int(self.p1[0]), int(self.p1[1]))
        end_pos = (int(self.p2[0]), int(self.p2[1]))

        # Draw main platform line with integer thickness
        pygame.draw.line(screen, (0, 200, 0), start_pos, end_pos, self.thickness)


class Ball:
    def __init__(self, space, position, size=10):  # Add size parameter
        self.radius = size
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = position

        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0.95
        self.shape.friction = 0.9
        self.shape.collision_type = 2

        space.add(self.body, self.shape)

    def draw(self, screen):
        pos = int(self.body.position.x), int(self.body.position.y)
        pygame.draw.circle(screen, (200, 140, 0), pos, int(self.radius))
        pygame.draw.circle(screen, (255, 165, 0), pos, int(self.radius - 2))


def create_ball(space, size=10):
    return Ball(space, (850, 30), size)


def create_platforms(space):
    platforms = []
    thickness = 16  # Slightly thinner platforms for better aesthetics

    platform_positions = [
        ((0, 580), (800, 580)),  # Floor
        ((-20, 490), (700, 520)),  # First slope (left to right)
        ((820, 400), (100, 430)),  # Second slope (right to left)
        ((-20, 310), (700, 340)),  # Third slope (left to right)
        ((820, 220), (100, 250)),  # Fourth slope (right to left)
        ((-20, 130), (700, 160)),  # Fifth slope (left to right)
        ((900, 40), (100, 70))  # Top slope (right to left, extends off-screen)
    ]

    for p1, p2 in platform_positions:
        platforms.append(Platform(space, p1, p2, thickness))

    # Create walls with proper thickness
    walls = [
        ((-20, 0), (-20, 600)),  # Left wall (moved slightly off-screen)
        ((820, 100), (820, 600))  # Right wall (moved slightly off-screen)
    ]

    for p1, p2 in walls:
        platforms.append(Platform(space, p1, p2, thickness))

    return platforms
