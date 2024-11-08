import pygame
import pymunk
from death_animation_manager import DeathAnimationManager


class Bullet:
    def __init__(self, x, y, game_offset):
        self.x = x
        self.y = y
        self.speed = 10
        self.radius = 3
        self.game_offset = game_offset

    def update(self, offset_change):
        self.x += self.speed
        self.game_offset = offset_change

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.radius)


class Player:
    def __init__(self, space, x, y):
        self.size = 20
        self.color = (0, 255, 0)

        mass = 2.0
        moment = pymunk.moment_for_box(mass, (self.size, self.size))
        self.body = pymunk.Body(mass, moment)
        self.body.position = (x, y)
        self.shape = pymunk.Poly.create_box(self.body, (self.size, self.size))
        self.shape.elasticity = 0
        self.shape.friction = 0.7
        space.add(self.body, self.shape)
        self.space = space
        self.bullets = []

        self.max_thrust = -400
        self.thrust_buildup_rate = 15
        self.current_thrust = 0

        self.ship_points = [
            (0, -self.size // 2),
            (self.size // 2, self.size // 2),
            (self.size // 4, self.size // 2),
            (0, self.size // 4),
            (-self.size // 4, self.size // 2),
            (-self.size // 2, self.size // 2),
        ]

        self.flame_points = [
            (0, self.size // 2),
            (self.size // 4, self.size),
            (-self.size // 4, self.size)
        ]

        self.death_manager = DeathAnimationManager(self.ship_points, self.size)

    def start_death_animation(self, collision_type, environment_offset, collision_point=None):
        self.death_manager.start_animation(collision_type,
                                           (self.body.position.x, self.body.position.y),
                                           environment_offset,
                                           collision_point)

    def apply_thrust(self):
        self.current_thrust = min(self.current_thrust + self.thrust_buildup_rate, abs(self.max_thrust))
        thrust_force = -self.current_thrust
        self.body.apply_impulse_at_local_point((0, thrust_force), (0, 0))

    def reset_thrust(self):
        self.current_thrust = 0

    def shoot(self, game_offset):
        if not self.death_manager.is_active:
            bullet = Bullet(self.body.position.x + self.size / 2, self.body.position.y, game_offset)
            self.bullets.append(bullet)

    def update(self, game_offset):
        if self.death_manager.is_active:
            new_position = self.death_manager.update(game_offset)
            if new_position:
                self.body.position = pymunk.Vec2d(*new_position)
            return

        self.body.velocity = (0, self.body.velocity.y)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.apply_thrust()
        else:
            self.reset_thrust()

        max_velocity = 400
        min_velocity = -400
        if self.body.velocity.y > max_velocity:
            self.body.velocity = (0, max_velocity)
        elif self.body.velocity.y < min_velocity:
            self.body.velocity = (0, min_velocity)

        for bullet in self.bullets[:]:
            bullet.update(game_offset)
            if bullet.x > 800:
                self.bullets.remove(bullet)

    def draw(self, screen):
        if self.death_manager.is_active:
            self.death_manager.draw(screen)
            return

        pos = self.body.position

        # Draw ship body
        transformed_points = [(int(x + pos.x), int(y + pos.y)) for x, y in self.ship_points]
        pygame.draw.polygon(screen, (200, 200, 200), transformed_points)

        # Draw window
        window_pos = (int(pos.x), int(pos.y - self.size // 4))
        pygame.draw.circle(screen, (100, 200, 255), window_pos, self.size // 4)

        # Draw flame when thrusting
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            flame_scale = self.current_thrust / abs(self.max_thrust)
            flame_points = [(x, y * (0.5 + flame_scale)) for x, y in self.flame_points]
            flame_color = (255, 165, 0)
            transformed_flame = [(int(x + pos.x), int(y + pos.y)) for x, y in flame_points]
            pygame.draw.polygon(screen, flame_color, transformed_flame)

        for bullet in self.bullets:
            bullet.draw(screen)

    def is_death_animation_complete(self):
        return self.death_manager.is_complete()

    def reset(self, x, y):
        self.space.remove(self.body, self.shape)
        self.__init__(self.space, x, y)
        self.bullets = []