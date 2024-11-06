import pygame
import math
import random


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
        # Create a collision rect that matches the visual representation
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
        # Calculate the actual bounding box of the shape
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

            # Debug visualization of collision rect
            # adjusted_rect = self.collision_rect.copy()
            # adjusted_rect.x += offset
            # pygame.draw.rect(screen, (255, 0, 0), adjusted_rect, 1)

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
        if hasattr(entity, 'body'):  # Player collision
            player_rect = pygame.Rect(
                entity.body.position.x - entity.size / 2,
                entity.body.position.y - entity.size / 2,
                entity.size, entity.size
            )

            # Use the collision rect for checking
            adjusted_rect = self.collision_rect.copy()
            return player_rect.colliderect(adjusted_rect)
        else:  # Bullet collision
            bullet_rect = pygame.Rect(
                entity.x - entity.radius,
                entity.y - entity.radius,
                entity.radius * 2,
                entity.radius * 2
            )

            # Adjust collision rect for bullet collision
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
        self.scroll_speed = 3
        self.speed_increment = 0.1

        self.floor_color = (76, 50, 35)
        self.ceiling_color = (66, 40, 25)
        self.spike_color = (150, 150, 150)

        self.floor_base = height * 0.7
        self.ceiling_base = height * 0.3
        self.max_height_change = 30
        self.spike_chance = 0.1
        self.spike_length = 20
        self.obstacle_chance = 0.05

        self.generate_initial_segments()

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
        floor_change = random.randint(int(-self.max_height_change), int(self.max_height_change))
        new_floor_y = max(min(last_floor_y + floor_change, self.height - 100), self.height * 0.6)

        ceiling_change = random.randint(int(-self.max_height_change), int(self.max_height_change))
        new_ceiling_y = max(min(last_ceiling_y + ceiling_change, self.height * 0.4), 100)

        min_gap = 150
        if new_floor_y - new_ceiling_y < min_gap:
            adjustment = (min_gap - (new_floor_y - new_ceiling_y)) / 2
            new_floor_y += adjustment
            new_ceiling_y -= adjustment

        floor_spike = random.random() < self.spike_chance
        ceiling_spike = random.random() < self.spike_chance

        self.floor_points.append((x, new_floor_y, floor_spike))
        self.ceiling_points.append((x, new_ceiling_y, ceiling_spike))

        if random.random() < self.obstacle_chance:
            obstacle_y = random.uniform(new_ceiling_y + 50, new_floor_y - 50)
            obstacle_size = random.randint(20, 40)
            self.obstacles.append(Obstacle(x, obstacle_y, obstacle_size))

    def update(self):
        self.offset -= self.scroll_speed
        self.difficulty = min(self.difficulty + 0.0003, 2.0)
        self.max_height_change = 30 * self.difficulty
        self.scroll_speed = 3 + self.difficulty * self.speed_increment

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

    def draw(self, screen):
        def draw_section(points, base_y, color, is_floor):
            visible_points = [(x + self.offset, y) for x, y, spike in points
                              if -self.segment_width <= x + self.offset <= self.width + self.segment_width]
            spike_info = [(x + self.offset, spike) for x, y, spike in points
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

                for i, (x, has_spike) in enumerate(spike_info[:-1]):
                    if has_spike:
                        next_x = spike_info[i + 1][0]
                        mid_x = (x + next_x) / 2
                        y = visible_points[i][1]
                        spike_tip = y + (self.spike_length if is_floor else -self.spike_length)

                        pygame.draw.polygon(screen, self.spike_color,
                                            [(x, y), (mid_x, spike_tip), (next_x, y)])

        draw_section(self.ceiling_points, 0, self.ceiling_color, False)
        draw_section(self.floor_points, self.height, self.floor_color, True)

        for obstacle in self.obstacles + self.destroyed_obstacles:
            obstacle.draw(screen, self.offset)

    def check_collision(self, player):
        # First, check obstacle collisions
        for obstacle in self.obstacles:
            adjusted_obstacle = obstacle.collision_rect.copy()
            adjusted_obstacle.x += self.offset

            player_rect = pygame.Rect(
                player.body.position.x - player.size / 2,
                player.body.position.y - player.size / 2,
                player.size, player.size
            )

            if player_rect.colliderect(adjusted_obstacle):
                return True

        # Then check wall collisions
        for points, is_floor in [(self.floor_points, True), (self.ceiling_points, False)]:
            for i in range(len(points) - 1):
                x1, y1, has_spike = points[i]
                x2, y2, _ = points[i + 1]

                adjusted_x1 = x1 + self.offset
                adjusted_x2 = x2 + self.offset

                if adjusted_x1 <= player.body.position.x <= adjusted_x2:
                    wall_height = y1
                    if (is_floor and player.body.position.y + player.size / 2 > wall_height) or \
                            (not is_floor and player.body.position.y - player.size / 2 < wall_height):
                        return True

                    if has_spike:
                        mid_x = (adjusted_x1 + adjusted_x2) / 2
                        if (is_floor and player.body.position.y + player.size / 2 > y1 - self.spike_length) or \
                                (not is_floor and player.body.position.y - player.size / 2 < y1 + self.spike_length):
                            if abs(player.body.position.x - mid_x) < self.segment_width / 2:
                                return True

        return False

    def remove_obstacle(self, obstacle):
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
            obstacle.start_destruction()
            self.destroyed_obstacles.append(obstacle)

    def reset(self):
        self.offset = 0
        self.difficulty = 1
        self.floor_points = []
        self.ceiling_points = []
        self.obstacles = []
        self.destroyed_obstacles = []
        self.generate_initial_segments()