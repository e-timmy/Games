import pygame
import numpy as np
from math_3d import Vector3, project_point
import random


class GameObject3D:
    def __init__(self, position: Vector3):
        self.position = position
        self.velocity = Vector3(0, 0, 0)


class SplatterParticle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(2, 8)
        self.dx = np.cos(angle) * speed
        self.dy = np.sin(angle) * speed
        self.lifetime = 1.0  # seconds
        self.color = color
        self.size = random.randint(10, 25)

    def update(self, delta_time):
        self.x += self.dx
        self.y += self.dy
        self.lifetime -= delta_time
        self.size *= 0.95

    def draw(self, screen):
        alpha = int(255 * (self.lifetime))
        surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (*self.color, alpha),
                           (self.size, self.size), self.size)
        screen.blit(surface, (self.x - self.size, self.y - self.size))


class Player3D(GameObject3D):
    def __init__(self, x, y):
        super().__init__(Vector3(x, y, 2))  # Move player forward
        self.radius = 20
        self.base_color = (255, 165, 0)
        self.trail = []

    def draw(self, screen, projection_matrix):
        screen_pos = project_point(self.position, projection_matrix)
        screen_x = int((screen_pos.x + 1) * screen.get_width() / 2)
        screen_y = int((screen_pos.y + 1) * screen.get_height() / 2)

        # Draw shadow
        shadow_radius = self.radius + 4
        pygame.draw.circle(screen, (0, 0, 0), (screen_x + 2, screen_y + 2), shadow_radius)

        # Draw sphere-like gradient
        for i in range(self.radius, 0, -2):
            brightness = 255 - (self.radius - i) * 8
            color = (min(255, brightness), min(255, brightness * 0.65), 0)
            pygame.draw.circle(screen, color, (screen_x, screen_y), i)

        # Highlight
        highlight_pos = (screen_x - self.radius // 3, screen_y - self.radius // 3)
        pygame.draw.circle(screen, (255, 255, 200), highlight_pos, self.radius // 4)


class Projectile3D(GameObject3D):
    def __init__(self, position: Vector3, velocity: Vector3, color=(255, 0, 0)):
        super().__init__(position)
        self.velocity = velocity
        self.base_size = 40
        self.color = color  # Now accepts custom colors
        self.active = True
        self.trail = []
        self.max_trail_length = 10
        self.has_splattered = False
        self.collision_radius = 0.3

    def create_splatter_particles(self, screen_x, screen_y):
        particles = []
        for _ in range(20):  # Create 20 particles
            particle = SplatterParticle(screen_x, screen_y, self.color)
            particles.append(particle)
        return particles

    def update(self, delta_time):
        # Store current position in trail
        self.trail.append((Vector3(self.position.x, self.position.y, self.position.z)))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

        # Update position
        next_z = self.position.z + self.velocity.z * delta_time

        # Check if the projectile will cross the screen plane in this frame
        if self.position.z <= 0 and next_z > 0:
            self.has_splattered = True
            self.active = False
            return

        self.position.x += self.velocity.x * delta_time
        self.position.y += self.velocity.y * delta_time
        self.position.z = next_z

        # Deactivate if somehow passed without splattering
        if self.position.z > 2:
            self.active = False

    def draw(self, screen, projection_matrix, time_frozen=False):
        if not self.active:
            return

        # Change color when time is frozen
        current_color = (0, 150, 255) if time_frozen else self.color

        for i, trail_pos in enumerate(self.trail):
            screen_pos = project_point(trail_pos, projection_matrix)
            screen_x = int((screen_pos.x + 1) * screen.get_width() / 2)
            screen_y = int((screen_pos.y + 1) * screen.get_height() / 2)

            depth_scale = 3 / (-trail_pos.z + 2)
            scaled_size = int(self.base_size * depth_scale)

            alpha = int(255 * (i / len(self.trail)))
            trail_surface = pygame.Surface((scaled_size, scaled_size), pygame.SRCALPHA)
            pygame.draw.circle(trail_surface, (*current_color, alpha),
                               (scaled_size // 2, scaled_size // 2), scaled_size // 2)
            screen.blit(trail_surface, (screen_x - scaled_size // 2, screen_y - scaled_size // 2))

        screen_pos = project_point(self.position, projection_matrix)
        screen_x = int((screen_pos.x + 1) * screen.get_width() / 2)
        screen_y = int((screen_pos.y + 1) * screen.get_height() / 2)

        depth_scale = 3 / (-self.position.z + 2)
        scaled_size = int(self.base_size * depth_scale)

        # Glow effect
        glow_surface = pygame.Surface((scaled_size * 3, scaled_size * 3), pygame.SRCALPHA)
        for i in range(4):
            glow_size = scaled_size + i * 6
            alpha = 120 - i * 30
            pygame.draw.circle(glow_surface, (*current_color, alpha),
                               (scaled_size * 1.5, scaled_size * 1.5), glow_size)
        screen.blit(glow_surface, (screen_x - scaled_size * 1.5, screen_y - scaled_size * 1.5))

        # Main projectile
        pygame.draw.circle(screen, current_color, (screen_x, screen_y), scaled_size)