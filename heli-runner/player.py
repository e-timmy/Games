import math
import pygame
import pymunk
from death_animation_manager import DeathAnimationManager

class Bullet:
    def __init__(self, x, y, game_offset, target=None, guided=False):
        self.x = x
        self.y = y
        self.speed = 10
        self.radius = 3
        self.game_offset = game_offset
        self.target = target
        self.guided = guided
        self.color = (255, 50, 50) if guided else (255, 255, 0)

        if guided and target:
            self.speed = 7  # Slower speed for guided bullets

    def update(self, offset_change):
        if self.guided and self.target:
            # Calculate direction to target
            target_x = self.target.x + offset_change
            target_y = self.target.y

            dx = target_x - self.x
            dy = target_y - self.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > 0:
                # Normalize and apply speed
                self.x += (dx / distance) * self.speed
                self.y += (dy / distance) * self.speed
        else:
            self.x += self.speed

        self.game_offset = offset_change

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


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

        self.has_shield = False
        self.shield_active = False
        self.shield_timer = 0
        self.shield_duration = 180  # 3 seconds at 60 FPS
        self.shield_flash_interval = 15  # Flash every 0.25 seconds

        self.has_scatter = False
        self.has_slow_down = False

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

        if self.shield_active:
            self.shield_timer += 1
            if self.shield_timer >= self.shield_duration:
                self.remove_shield()

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

        # Draw power-up indicators from outer to inner
        # Slow-down (outermost)
        if self.has_slow_down:
            pygame.draw.circle(screen, (0, 255, 0), (int(pos.x), int(pos.y)), self.size + 8, 2)

        # Shield (middle)
        if self.has_shield:
            if not self.shield_active or (self.shield_timer // self.shield_flash_interval) % 2 == 0:
                pygame.draw.circle(screen, (0, 191, 255), (int(pos.x), int(pos.y)), self.size + 5, 2)

        # Scatter shot (innermost)
        if self.has_scatter:
            pygame.draw.circle(screen, (255, 50, 50), (int(pos.x), int(pos.y)), self.size + 2, 2)

    def is_death_animation_complete(self):
        return self.death_manager.is_complete()

    def reset(self, x, y):
        self.space.remove(self.body, self.shape)
        self.__init__(self.space, x, y)
        self.bullets = []

    def add_shield(self):
        self.has_shield = True
        self.shield_active = False
        self.shield_timer = 0

    def activate_shield(self):
        if self.has_shield and not self.shield_active:
            self.shield_active = True
            self.shield_timer = 0

    def remove_shield(self):
        self.has_shield = False
        self.shield_active = False
        self.shield_timer = 0

    def add_scatter(self):
        self.has_scatter = True

    def activate_scatter(self, game_offset, visible_obstacles):
        if not self.has_scatter:
            return

        for obstacle in visible_obstacles:
            bullet = Bullet(
                self.body.position.x + self.size / 2,
                self.body.position.y,
                game_offset,
                target=obstacle,
                guided=True
            )
            self.bullets.append(bullet)

        # Remove scatter shot after use
        self.has_scatter = False

    def add_slow_down(self):
        self.has_slow_down = True

    def activate_slow_down(self):
        if self.has_slow_down:
            self.has_slow_down = False
            return True
        return False

    def start_death_animation(self, collision_type, environment_offset, collision_point=None):
        self.death_manager.start_animation(collision_type,
                                           (self.body.position.x, self.body.position.y),
                                           environment_offset,
                                           collision_point)