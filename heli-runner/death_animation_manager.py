import pygame
import random
import math


class DeathParticle:
    def __init__(self, x, y, dx, dy, size, color):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.size = size
        self.color = color
        self.life = random.randint(30, 60)
        self.gravity = 0.5

    def update(self, scroll_speed=0):
        self.x -= scroll_speed  # Adjust for screen scrolling
        self.x += self.dx
        self.y += self.dy
        self.dy += self.gravity
        self.life -= 1
        return self.life <= 0

    def draw(self, screen):
        alpha = int((self.life / 60) * 255)
        particle_surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(particle_surface, (*self.color, alpha), (self.size, self.size), self.size)
        screen.blit(particle_surface, (int(self.x - self.size), int(self.y - self.size)))


class DeathAnimationManager:
    def __init__(self, ship_points, size):
        self.is_active = False
        self.ship_points = ship_points
        self.size = size
        self.death_particles = []
        self.fire_particles = []
        self.smoke_particles = []
        self.animation_time = 120
        self.timer = 0
        self.death_type = None
        self.death_velocity = None
        self.collision_point = None
        self.initial_position = None
        self.position = None
        self.environment_offset = 0
        self.initial_environment_offset = 0

    def start_animation(self, death_type, position, environment_offset, collision_point=None):
        self.is_active = True
        self.timer = 0
        self.death_type = death_type
        self.initial_position = position
        self.position = position
        self.collision_point = collision_point
        self.initial_environment_offset = environment_offset
        self.environment_offset = environment_offset

        if death_type == "obstacle":
            angle = math.atan2(collision_point[1] - position[1],
                               collision_point[0] - position[0])
            deflection_speed = 5
            self.death_velocity = (-math.cos(angle) * deflection_speed,
                                   -math.sin(angle) * deflection_speed)
            self.create_fire_particles()
        else:  # Wall collision
            self.death_velocity = (0, 0)
            self.create_explosion_particles()
            self.create_smoke_particles()

    def create_smoke_particles(self):
        for _ in range(15):
            dx = random.uniform(-0.5, 0.5)
            dy = random.uniform(-2, -1)
            size = random.randint(3, 6)
            color = (100, 100, 100)  # Gray smoke
            particle = DeathParticle(self.position[0], self.position[1], dx, dy, size, color)
            self.smoke_particles.append(particle)

    def create_fire_particles(self):
        for _ in range(10):
            dx = random.uniform(-1, 1)
            dy = random.uniform(-3, -1)
            size = random.randint(2, 4)
            color = (255, random.randint(100, 200), 0)
            particle = DeathParticle(self.collision_point[0], self.collision_point[1], dx, dy, size, color)
            self.fire_particles.append(particle)

    def create_explosion_particles(self):
        for _ in range(random.randint(30, 40)):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            size = random.randint(2, 4)
            colors = [(255, 165, 0), (255, 69, 0), (255, 0, 0)]
            color = random.choice(colors)
            particle = DeathParticle(self.position[0], self.position[1], dx, dy, size, color)
            self.death_particles.append(particle)

    def update(self, current_environment_offset):
        if not self.is_active:
            return None

        self.timer += 1

        # Calculate offset difference
        offset_diff = current_environment_offset - self.environment_offset
        self.environment_offset = current_environment_offset

        if self.death_type == "obstacle":
            if self.timer < 60:
                # Update position for obstacle collision
                new_x = self.position[0] + self.death_velocity[0]
                new_y = self.position[1] + self.death_velocity[1]
                self.position = (new_x, new_y)
                self.fire_particles = [p for p in self.fire_particles if not p.update()]
            elif self.timer == 60:
                self.create_explosion_particles()
                self.create_smoke_particles()
        else:
            # For wall collision, adjust x position to stay relative to initial crash site
            new_x = self.initial_position[0] - (self.initial_environment_offset - current_environment_offset)
            self.position = (new_x, self.position[1])

            if self.timer % 15 == 0:  # Create new smoke particles periodically
                self.create_smoke_particles()

        self.death_particles = [p for p in self.death_particles if not p.update()]
        self.smoke_particles = [p for p in self.smoke_particles if not p.update()]

        return self.position

    def draw(self, screen):
        if not self.is_active:
            return

        if self.timer < 60 or self.death_type == "wall":
            distortion = min(self.timer / 60, 1)
            distorted_points = []
            for x, y in self.ship_points:
                dx = random.uniform(-3, 3) * distortion
                dy = random.uniform(-3, 3) * distortion
                distorted_points.append((int(x + dx + self.position[0]), int(y + dy + self.position[1])))

            pygame.draw.polygon(screen, (200, 200, 200), distorted_points)

        if self.death_type == "obstacle" and self.timer < 60:
            for particle in self.fire_particles:
                particle.draw(screen)

        for particle in self.death_particles:
            particle.draw(screen)

        for particle in self.smoke_particles:
            particle.draw(screen)

    def is_complete(self):
        return self.is_active and self.timer >= self.animation_time
