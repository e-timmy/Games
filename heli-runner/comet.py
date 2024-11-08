import pygame
import math
import random


class CometParticle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.radius = random.randint(2, 4)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3)
        self.dx = math.cos(angle) * speed
        self.dy = math.sin(angle) * speed
        self.life = random.randint(30, 60)

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.life -= 1
        return self.life <= 0

    def draw(self, screen):
        alpha = int((self.life / 60) * 255)
        color = (*self.color, alpha)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)


class Comet:
    def __init__(self, screen_width, screen_height, environment_offset):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = screen_width + abs(environment_offset)

        # Calculate the middle path
        self.ceiling_height = screen_height * 0.3
        self.floor_height = screen_height * 0.7
        middle_y = (self.ceiling_height + self.floor_height) / 2

        # Start at the right side, aiming for the middle left
        self.y = random.uniform(self.ceiling_height, self.floor_height)
        target_y = middle_y + random.uniform(-30, 30)  # Slight variance in end point

        self.radius = 10
        self.speed = 7

        # Calculate dx and dy for the path
        dx = -screen_width  # Full width of the screen
        dy = target_y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        self.dx = (dx / distance) * self.speed
        self.dy = (dy / distance) * self.speed

        self.color = (255, 165, 0)  # Orange color for the comet
        self.trail_length = 15
        self.trail = [(self.x, self.y) for _ in range(self.trail_length)]

        self.exploding = False
        self.explosion_particles = []

    def update(self, scroll_speed):
        if self.exploding:
            self.explosion_particles = [p for p in self.explosion_particles if not p.update()]
            return len(self.explosion_particles) > 0

        self.x += self.dx + scroll_speed
        self.y += self.dy

        self.trail.pop(0)
        self.trail.append((self.x, self.y))

        return True

    def draw(self, screen):
        if self.exploding:
            for particle in self.explosion_particles:
                particle.draw(screen)
        else:
            for i, pos in enumerate(self.trail):
                alpha = int(255 * (i / self.trail_length))
                radius = int(self.radius * (i / self.trail_length))

                trail_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(trail_surface, (*self.color, alpha), (radius, radius), radius)
                screen.blit(trail_surface, (int(pos[0] - radius), int(pos[1] - radius)))

            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def collides_with(self, other):
        if hasattr(other, 'collision_rect'):
            return other.collision_rect.collidepoint(self.x, self.y)
        elif hasattr(other, 'body'):
            dx = self.x - other.body.position.x
            dy = self.y - other.body.position.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            return distance < (self.radius + other.size / 2)
        return False

    def explode(self):
        self.exploding = True
        for _ in range(20):
            self.explosion_particles.append(CometParticle(self.x, self.y, self.color))

    def is_off_screen(self):
        return self.x < -self.radius