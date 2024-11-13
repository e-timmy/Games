import pygame
import pymunk
import math
import random
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
            self.speed = 7

    def update(self, offset_change):
        if self.guided and self.target:
            target_x = self.target.x + offset_change
            target_y = self.target.y

            dx = target_x - self.x
            dy = target_y - self.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > 0:
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

        self.original_gravity = space.gravity
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

        self.max_thrust = 400
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

        # Power-up attributes
        self.has_shield = False
        self.shield_active = False
        self.shield_timer = 0
        self.shield_duration = 180
        self.shield_flash_interval = 15

        self.has_scatter = False
        self.has_slow_down = False

        # Autopilot attributes
        self.has_autopilot = False
        self.autopilot_active = False
        self.autopilot_timer = 0
        self.autopilot_duration = 600
        self.autopilot_shoot_cooldown = 0
        self.autopilot_shoot_cooldown_max = 30
        self.autopilot_target_y = 0
        self.autopilot_y_variation = 0
        self.autopilot_y_variation_timer = 0
        self.autopilot_speed = 200  # Constant speed for autopilot movement

    def manual_update(self):
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

    def apply_thrust(self):
        self.current_thrust = min(self.current_thrust + self.thrust_buildup_rate, self.max_thrust)
        self.body.apply_impulse_at_local_point((0, -self.current_thrust))

    def reset_thrust(self):
        self.current_thrust = 0

    def autopilot_update(self, environment):
        player_x = self.body.position.x
        ceiling_height = None
        floor_height = None

        # Find ceiling and floor heights
        for i in range(len(environment.ceiling_points) - 1):
            x1, y1, _ = environment.ceiling_points[i]
            x2, y2, _ = environment.ceiling_points[i + 1]
            adjusted_x1 = x1 + environment.offset
            adjusted_x2 = x2 + environment.offset

            if adjusted_x1 <= player_x <= adjusted_x2:
                t = (player_x - adjusted_x1) / (adjusted_x2 - adjusted_x1)
                ceiling_height = y1 + t * (y2 - y1)
                break

        for i in range(len(environment.floor_points) - 1):
            x1, y1, _ = environment.floor_points[i]
            x2, y2, _ = environment.floor_points[i + 1]
            adjusted_x1 = x1 + environment.offset
            adjusted_x2 = x2 + environment.offset

            if adjusted_x1 <= player_x <= adjusted_x2:
                t = (player_x - adjusted_x1) / (adjusted_x2 - adjusted_x1)
                floor_height = y1 + t * (y2 - y1)
                break

        if ceiling_height is not None and floor_height is not None:
            middle_y = (ceiling_height + floor_height) / 2

            self.autopilot_y_variation_timer += 1
            if self.autopilot_y_variation_timer >= 60:
                self.autopilot_y_variation = random.uniform(-20, 20)
                self.autopilot_y_variation_timer = 0

            self.autopilot_target_y = middle_y + self.autopilot_y_variation

            # Move towards the target
            direction = 1 if self.autopilot_target_y > self.body.position.y else -1
            self.body.velocity = (0, direction * self.autopilot_speed)

            # Adjust speed based on distance to target
            distance_to_target = abs(self.autopilot_target_y - self.body.position.y)
            if distance_to_target < 10:
                self.body.velocity = (0, 0)

            # Set current_thrust for flame animation
            self.current_thrust = abs(self.body.velocity.y)

        # Shooting logic
        if self.autopilot_shoot_cooldown <= 0:
            detection_range = 300
            for obstacle in environment.obstacles:
                adjusted_x = obstacle.x + environment.offset
                if (player_x < adjusted_x < player_x + detection_range and
                        abs(obstacle.y - self.body.position.y) < 50):
                    self.shoot(environment.offset)
                    self.autopilot_shoot_cooldown = self.autopilot_shoot_cooldown_max
                    break
            else:
                for comet in environment.comets:
                    if (not comet.exploding and
                            player_x < comet.x < player_x + detection_range and
                            abs(comet.y - self.body.position.y) < 50):
                        self.shoot(environment.offset)
                        self.autopilot_shoot_cooldown = self.autopilot_shoot_cooldown_max
                        break
        else:
            self.autopilot_shoot_cooldown -= 1

    def shoot(self, game_offset):
        if not self.death_manager.is_active:
            bullet = Bullet(self.body.position.x + self.size / 2, self.body.position.y, game_offset)
            self.bullets.append(bullet)

    def update(self, game_offset, environment=None):
        if self.death_manager.is_active:
            new_position = self.death_manager.update(game_offset)
            if new_position:
                self.body.position = pymunk.Vec2d(*new_position)
            return

        # Handle autopilot
        if self.autopilot_active and environment:
            # Disable gravity during autopilot
            self.space.gravity = (0, 0)

            self.autopilot_timer += 1
            if self.autopilot_timer >= self.autopilot_duration:
                self.deactivate_autopilot()
            else:
                self.autopilot_update(environment)
        else:
            # Restore original gravity when not in autopilot
            self.space.gravity = self.original_gravity
            self.manual_update()

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
        if self.current_thrust > 0:
            flame_scale = self.current_thrust / self.max_thrust
            flame_points = [(x, y * (0.5 + flame_scale)) for x, y in self.flame_points]
            flame_color = (255, 165, 0)
            transformed_flame = [(int(x + pos.x), int(y + pos.y)) for x, y in flame_points]
            pygame.draw.polygon(screen, flame_color, transformed_flame)

        for bullet in self.bullets:
            bullet.draw(screen)

        # Draw power-up indicators
        if self.has_autopilot or self.autopilot_active:
            pygame.draw.circle(screen, (255, 255, 255),
                               (int(pos.x), int(pos.y)),
                               self.size + 11, 2)

        if self.has_slow_down:
            pygame.draw.circle(screen, (0, 255, 0), (int(pos.x), int(pos.y)), self.size + 8, 2)

        if self.has_shield:
            if not self.shield_active or (self.shield_timer // self.shield_flash_interval) % 2 == 0:
                pygame.draw.circle(screen, (0, 191, 255), (int(pos.x), int(pos.y)), self.size + 5, 2)

        if self.has_scatter:
            pygame.draw.circle(screen, (255, 50, 50), (int(pos.x), int(pos.y)), self.size + 2, 2)

    def draw_autopilot_indicator(self, screen):
        if self.autopilot_active:
            bar_width = 200
            bar_height = 10
            x = (screen.get_width() - bar_width) // 2
            y = 20

            # Background bar
            pygame.draw.rect(screen, (100, 100, 100), (x, y, bar_width, bar_height))

            # Remaining time bar
            remaining_ratio = 1 - (self.autopilot_timer / self.autopilot_duration)
            remaining_width = int(bar_width * remaining_ratio)

            color = (255, 255, 255)
            if self.autopilot_duration - self.autopilot_timer < 120:
                if self.autopilot_timer % 30 < 15:
                    color = (255, 0, 0)

            pygame.draw.rect(screen, color, (x, y, remaining_width, bar_height))

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

        self.has_scatter = False

    def add_slow_down(self):
        self.has_slow_down = True

    def activate_slow_down(self):
        if self.has_slow_down:
            self.has_slow_down = False
            return True
        return False

    def add_autopilot(self):
        self.has_autopilot = True
        self.autopilot_active = False
        self.autopilot_timer = 0

    def activate_autopilot(self):
        if self.has_autopilot and not self.autopilot_active:
            print("Autopilot activated!")
            self.autopilot_active = True
            self.autopilot_timer = 0
            self.autopilot_target_y = self.body.position.y
            self.autopilot_y_variation = 0
            self.autopilot_y_variation_timer = 0
            self.has_autopilot = False
            self.body.velocity = (0, 0)  # Reset velocity when activating
            return True
        return False

    def deactivate_autopilot(self):
        print("Autopilot deactivated!")
        self.autopilot_active = False
        self.autopilot_timer = 0
        self.current_thrust = 0
        self.body.velocity = (0, 0)  # Reset velocity when deactivating


    def start_death_animation(self, collision_type, environment_offset, collision_point=None):
        self.death_manager.start_animation(collision_type,
                                           (self.body.position.x, self.body.position.y),
                                           environment_offset,
                                           collision_point)

    def is_death_animation_complete(self):
        return self.death_manager.is_complete()

    def reset(self, x, y):
        self.space.remove(self.body, self.shape)
        self.__init__(self.space, x, y)
        self.bullets = []