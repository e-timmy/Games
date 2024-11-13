import pygame
import math
import random
from comet import Comet
from pickup import ShieldPickup, ScatterShotPickup, SlowDownPickup, AutoPilotPickup


class ParticleEffect:
    def __init__(self, x, y, color=(100, 100, 100)):
        self.particles = []
        num_particles = 15
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            particle = {
                'x': x,
                'y': y,
                'dx': math.cos(angle) * speed,
                'dy': math.sin(angle) * speed,
                'life': 30,
                'size': random.randint(2, 4)
            }
            self.particles.append(particle)

    def update(self):
        for particle in self.particles:
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            particle['life'] -= 1

    def draw(self, screen, offset):
        for particle in self.particles:
            alpha = int((particle['life'] / 30) * 255)
            surf = pygame.Surface((particle['size'], particle['size']))
            surf.set_alpha(alpha)
            surf.fill((100, 100, 100))
            screen.blit(surf, (particle['x'] + offset, particle['y']))

    def is_finished(self):
        return all(p['life'] <= 0 for p in self.particles)


class Obstacle:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.points = self.generate_shape()
        self.being_destroyed = False
        self.destruction_effect = None
        self.collision_rect = self.calculate_collision_rect()

    def generate_shape(self):
        points = []
        for i in range(5):
            angle = 2 * math.pi * i / 5
            distance = self.size * (0.8 + 0.4 * random.random())
            px = self.x + math.cos(angle) * distance
            py = self.y + math.sin(angle) * distance
            points.append((px, py))
        return points

    def calculate_collision_rect(self):
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width = max_x - min_x
        height = max_y - min_y
        return pygame.Rect(min_x, min_y, width, height)

    def draw(self, screen, offset):
        if self.being_destroyed:
            if self.destruction_effect:
                self.destruction_effect.draw(screen, offset)
        else:
            adjusted_points = [(x + offset, y) for x, y in self.points]
            pygame.draw.polygon(screen, (100, 100, 100), adjusted_points)

    def start_destruction(self):
        self.being_destroyed = True
        self.destruction_effect = ParticleEffect(self.x, self.y)

    def update(self):
        if self.being_destroyed and self.destruction_effect:
            self.destruction_effect.update()
            if self.destruction_effect.is_finished():
                return True
        return False

    def collides_with(self, entity):
        if hasattr(entity, 'body'):
            player_rect = pygame.Rect(
                entity.body.position.x - entity.size / 2,
                entity.body.position.y - entity.size / 2,
                entity.size, entity.size
            )
            adjusted_rect = self.collision_rect.copy()
            return player_rect.colliderect(adjusted_rect)
        else:
            bullet_rect = pygame.Rect(
                entity.x - entity.radius,
                entity.y - entity.radius,
                entity.radius * 2,
                entity.radius * 2
            )
            adjusted_rect = self.collision_rect.copy()
            adjusted_rect.x += entity.game_offset
            return bullet_rect.colliderect(adjusted_rect)


class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.offset = 0
        self.difficulty = 1
        self.floor_points = []
        self.ceiling_points = []
        self.obstacles = []
        self.destroyed_obstacles = []
        self.segment_width = 20
        self.base_scroll_speed = 1.5
        self.scroll_speed = self.base_scroll_speed
        self.speed_increment = 0.05

        self.floor_color = (76, 50, 35)
        self.ceiling_color = (66, 40, 25)

        # Terrain difficulty attributes
        self.terrain_difficulty = 0.0
        self.terrain_difficulty_increment = 0.0001  # Increase per frame
        self.max_terrain_difficulty = 1.0

        # Dynamic terrain parameters
        self.initial_floor_base = height * 0.9  # Start very low
        self.initial_ceiling_base = height * 0.1  # Start very high
        self.final_floor_base = height * 0.7  # End position
        self.final_ceiling_base = height * 0.3  # End position

        self.initial_max_height_change = 5  # Start with minimal waviness
        self.final_max_height_change = 40  # End with more waviness

        self.floor_base = self.initial_floor_base
        self.ceiling_base = self.initial_ceiling_base
        self.max_height_change = self.initial_max_height_change

        self.obstacle_chance = 0.05

        self.generate_initial_segments()
        self.start_time = pygame.time.get_ticks()

        self.comets = []
        self.comet_spawn_timer = 0
        self.comet_spawn_interval = 180

        self.pickups = []
        self.pickup_spawn_timer = 0
        self.pickup_spawn_interval = 300

    def update_terrain_parameters(self):
        # Update base positions
        self.floor_base = self.initial_floor_base + (
                    self.final_floor_base - self.initial_floor_base) * self.terrain_difficulty
        self.ceiling_base = self.initial_ceiling_base + (
                    self.final_ceiling_base - self.initial_ceiling_base) * self.terrain_difficulty

        # Update waviness
        self.max_height_change = self.initial_max_height_change + (
                    self.final_max_height_change - self.initial_max_height_change) * self.terrain_difficulty

    def generate_initial_segments(self):
        self.floor_points = []
        self.ceiling_points = []
        self.obstacles = []

        last_floor_y = self.floor_base
        last_ceiling_y = self.ceiling_base

        for x in range(0, self.width * 2, self.segment_width):
            self.generate_segment_at(x, last_floor_y, last_ceiling_y)
            last_floor_y = self.floor_points[-1][1]
            last_ceiling_y = self.ceiling_points[-1][1]

    def generate_segment_at(self, x, last_floor_y, last_ceiling_y):
        # Calculate change magnitude based on difficulty
        current_max_change = self.max_height_change * (0.5 + 0.5 * math.sin(x * 0.01 * (1 + self.terrain_difficulty)))

        floor_change = random.randint(int(-current_max_change), int(current_max_change))
        new_floor_y = last_floor_y + floor_change

        ceiling_change = random.randint(int(-current_max_change), int(current_max_change))
        new_ceiling_y = last_ceiling_y + ceiling_change

        # Ensure floor and ceiling stay within bounds
        new_floor_y = max(min(new_floor_y, self.height - 50), self.floor_base - current_max_change)
        new_ceiling_y = max(min(new_ceiling_y, self.ceiling_base + current_max_change), 50)

        # Ensure minimum gap between floor and ceiling (adjusted by difficulty)
        min_gap = 150 * (1 - (self.terrain_difficulty * 0.3))  # Gap shrinks as difficulty increases
        if new_floor_y - new_ceiling_y < min_gap:
            adjustment = (min_gap - (new_floor_y - new_ceiling_y)) / 2
            new_floor_y += adjustment
            new_ceiling_y -= adjustment

        self.floor_points.append((x, new_floor_y, False))
        self.ceiling_points.append((x, new_ceiling_y, False))

        if random.random() < self.obstacle_chance:
            obstacle_y = random.uniform(new_ceiling_y + 50, new_floor_y - 50)
            obstacle_size = random.randint(20, 40)
            self.obstacles.append(Obstacle(x, obstacle_y, obstacle_size))

    def update(self):
        # Update terrain difficulty
        self.terrain_difficulty = min(self.terrain_difficulty + self.terrain_difficulty_increment,
                                      self.max_terrain_difficulty)
        self.update_terrain_parameters()

        self.offset -= self.scroll_speed
        elapsed = (pygame.time.get_ticks() - self.start_time) / 1000.0
        self.difficulty = min(1 + (elapsed * 0.05), 2.0)

        for obstacle in self.destroyed_obstacles[:]:
            if obstacle.update():
                self.destroyed_obstacles.remove(obstacle)

        while self.floor_points and self.floor_points[0][0] + self.offset < -self.segment_width:
            self.floor_points.pop(0)
            self.ceiling_points.pop(0)

        self.obstacles = [obs for obs in self.obstacles if obs.x + self.offset > -50]

        rightmost_x = self.floor_points[-1][0]
        visible_right_edge = -self.offset + self.width

        last_floor_y = self.floor_points[-1][1]
        last_ceiling_y = self.ceiling_points[-1][1]

        while rightmost_x < visible_right_edge + self.width:
            next_x = rightmost_x + self.segment_width
            self.generate_segment_at(next_x, last_floor_y, last_ceiling_y)
            last_floor_y = self.floor_points[-1][1]
            last_ceiling_y = self.ceiling_points[-1][1]
            rightmost_x = next_x

        self.comets = [comet for comet in self.comets if
                       comet.update(self.scroll_speed) and not comet.is_off_screen()]

        self.comet_spawn_timer += 1
        if self.comet_spawn_timer >= self.comet_spawn_interval:
            self.spawn_comet()
            self.comet_spawn_timer = 0

        self.check_comet_collisions()

        for pickup in self.pickups[:]:
            if pickup.x + self.offset <= -50:
                print(
                    f"Removing off-screen pickup of type: {type(pickup).__name__} at position x:{pickup.x + self.offset}")
        self.pickups = [pickup for pickup in self.pickups if pickup.x + self.offset > -50]
        for pickup in self.pickups:
            pickup.update()

        self.pickup_spawn_timer += 1
        if self.pickup_spawn_timer >= self.pickup_spawn_interval:
            self.spawn_pickup()
            self.pickup_spawn_timer = 0

    def spawn_comet(self):
        self.comets.append(Comet(self.width, self.height, self.offset))

    def get_safe_spawn_position(self):
        """Get a safe position within the cave for spawning pickups"""
        # Find the rightmost visible segment
        x = self.width + abs(self.offset)

        # Find the floor and ceiling heights at this position
        floor_height = None
        ceiling_height = None

        for i in range(len(self.floor_points) - 1):
            x1, y1, _ = self.floor_points[i]
            x2, y2, _ = self.floor_points[i + 1]
            if x1 <= x <= x2:
                # Interpolate floor height
                t = (x - x1) / (x2 - x1)
                floor_height = y1 + t * (y2 - y1)
                break

        for i in range(len(self.ceiling_points) - 1):
            x1, y1, _ = self.ceiling_points[i]
            x2, y2, _ = self.ceiling_points[i + 1]
            if x1 <= x <= x2:
                # Interpolate ceiling height
                t = (x - x1) / (x2 - x1)
                ceiling_height = y1 + t * (y2 - y1)
                break

        if floor_height is None or ceiling_height is None:
            return None, None

        # Calculate safe zone (with padding)
        padding = 40  # Increased padding for safety
        safe_min_y = ceiling_height + padding
        safe_max_y = floor_height - padding

        # Ensure there's enough space
        if safe_max_y - safe_min_y < padding * 2:
            return None, None

        # Check for obstacles in spawn area
        spawn_area_rect = pygame.Rect(x - padding, safe_min_y, padding * 2, safe_max_y - safe_min_y)

        for obstacle in self.obstacles:
            if obstacle.collision_rect.colliderect(spawn_area_rect):
                return None, None

        # Return position in middle of safe zone
        y = (safe_min_y + safe_max_y) / 2
        return x, y

    def spawn_pickup(self):
        x, y = self.get_safe_spawn_position()
        if x is not None and y is not None:
            pickup_type = random.choices(
                [AutoPilotPickup],
                weights=[1.0]
            )[0]
            print(f"Spawning pickup of type: {pickup_type.__name__}")
            self.pickups.append(pickup_type(x, y))
            print(f"Current pickups: {[type(p).__name__ for p in self.pickups]}")

    def check_comet_collisions(self):
        for comet in self.comets[:]:
            # Check collision with obstacles
            for obstacle in self.obstacles[:]:
                if comet.collides_with(obstacle):
                    comet.explode()
                    self.remove_obstacle(obstacle)
                    break

    def check_wall_collision(self, comet):
        for points in [self.floor_points, self.ceiling_points]:
            for i in range(len(points) - 1):
                x1, y1, _ = points[i]
                x2, y2, _ = points[i + 1]

                adjusted_x1 = x1 + self.offset
                adjusted_x2 = x2 + self.offset

                if adjusted_x1 <= comet.x <= adjusted_x2:
                    if abs(comet.y - y1) < comet.radius:
                        return True

        return False

    def draw(self, screen):
        def draw_section(points, base_y, color, is_floor):
            visible_points = [(x + self.offset, y) for x, y, _ in points
                              if -self.segment_width <= x + self.offset <= self.width + self.segment_width]

            if len(visible_points) >= 2:
                left_point = (0, visible_points[0][1])
                right_point = (self.width, visible_points[-1][1])

                all_points = [left_point] + visible_points + [right_point]
                extended_points = [(p[0], base_y) for p in all_points]

                if is_floor:
                    pygame.draw.polygon(screen, color, all_points + extended_points[::-1])
                else:
                    pygame.draw.polygon(screen, color, all_points + extended_points[::-1])

        draw_section(self.ceiling_points, 0, self.ceiling_color, False)
        draw_section(self.floor_points, self.height, self.floor_color, True)

        for obstacle in self.obstacles + self.destroyed_obstacles:
            obstacle.draw(screen, self.offset)

        # Draw comets
        for comet in self.comets:
            comet.draw(screen)

        # Draw pickups
        for pickup in self.pickups:
            pickup.draw(screen, self.offset)

    def check_collision(self, player):
        # In test mode, we still want to check collisions but not return them
        # This allows us to track logic without causing death
        collision_detected = False
        collision_type = None
        collision_point = None

        # Check obstacle collisions
        for obstacle in self.obstacles:
            adjusted_obstacle = obstacle.collision_rect.copy()
            adjusted_obstacle.x += self.offset

            player_rect = pygame.Rect(
                player.body.position.x - player.size / 2,
                player.body.position.y - player.size / 2,
                player.size, player.size
            )

            if player_rect.colliderect(adjusted_obstacle):
                collision_detected = True
                collision_type = "obstacle"
                collision_point = (adjusted_obstacle.centerx, adjusted_obstacle.centery)
                break

        # Check wall collisions
        if not collision_detected:
            for points, is_floor in [(self.floor_points, True), (self.ceiling_points, False)]:
                for i in range(len(points) - 1):
                    x1, y1, _ = points[i]
                    x2, y2, _ = points[i + 1]

                    adjusted_x1 = x1 + self.offset
                    adjusted_x2 = x2 + self.offset

                    if adjusted_x1 <= player.body.position.x <= adjusted_x2:
                        wall_height = y1
                        if (is_floor and player.body.position.y + player.size / 2 > wall_height) or \
                                (not is_floor and player.body.position.y - player.size / 2 < wall_height):
                            collision_detected = True
                            collision_type = "wall"
                            collision_point = (player.body.position.x, wall_height)
                            break
                if collision_detected:
                    break

        # Check collision with comets
        if not collision_detected:
            for comet in self.comets:
                if not comet.exploding and comet.collides_with(player):
                    comet.explode()
                    collision_detected = True
                    collision_type = "comet"
                    collision_point = (comet.x, comet.y)
                    break

        return (collision_detected, collision_type, collision_point)

    def remove_obstacle(self, obstacle):
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
            obstacle.start_destruction()
            self.destroyed_obstacles.append(obstacle)

    def reset(self):
        self.offset = 0
        self.difficulty = 1
        self.terrain_difficulty = 0.0
        self.scroll_speed = self.base_scroll_speed
        self.floor_points = []
        self.ceiling_points = []
        self.obstacles = []
        self.destroyed_obstacles = []
        self.comets = []
        self.comet_spawn_timer = 0
        self.pickups = []
        self.pickup_spawn_timer = 0
        self.update_terrain_parameters()
        self.generate_initial_segments()
        self.start_time = pygame.time.get_ticks()