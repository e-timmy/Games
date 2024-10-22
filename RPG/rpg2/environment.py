import math
import pygame
import random

from ForestGenerator import ForestGenerator
from constants import TILE_SIZE

import math


class Quadrant:
    def __init__(self, x, y, width, height, season):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.season = season
        self.ground_tiles = [['' for _ in range(width)] for _ in range(height)]
        self.object_tiles = [['' for _ in range(width)] for _ in range(height)]
        self.map = [[0 for _ in range(width)] for _ in range(height)]
        self.boss_area = None
        self.points_of_interest = []

    def create_boss_ring(self, center_x, center_y):
        ring_radius = 12
        ring_width = 4

        if self.x == 0 and self.y == 0:  # Top-left
            corner_x = self.width // 4
            corner_y = self.height // 4
        elif self.x > 0 and self.y == 0:  # Top-right
            corner_x = self.width - self.width // 4
            corner_y = self.height // 4
        elif self.x == 0 and self.y > 0:  # Bottom-left
            corner_x = self.width // 4
            corner_y = self.height - self.height // 4
        else:  # Bottom-right
            corner_x = self.width - self.width // 4
            corner_y = self.height - self.height // 4

        for y in range(self.height):
            for x in range(self.width):
                dx = x - corner_x
                dy = y - corner_y
                distance = math.sqrt(dx * dx + dy * dy)

                if ring_radius - ring_width <= distance <= ring_radius:
                    self.ground_tiles[y][x] = f'{self.season}_grass'
                    self.object_tiles[y][x] = f'{self.season}_tree'
                    self.map[y][x] = 1
                elif distance < ring_radius - ring_width:
                    self.ground_tiles[y][x] = f'{self.season}_grass'
                    self.object_tiles[y][x] = ''
                    self.map[y][x] = 0

        self.boss_area = (corner_x, corner_y)
        self.add_point_of_interest('boss_arena', self.boss_area)

    def add_point_of_interest(self, poi_type, coordinates):
        self.points_of_interest.append({
            'type': poi_type,
            'coordinates': coordinates
        })

    def find_nearest_point_of_interest(self, current_x, current_y):
        nearest_poi = None
        min_distance = float('inf')

        for poi in self.points_of_interest:
            poi_x, poi_y = poi['coordinates']
            distance = math.sqrt((current_x - poi_x) ** 2 + (current_y - poi_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                nearest_poi = poi

        return nearest_poi, min_distance

    def is_on_point_of_interest(self, current_x, current_y, threshold=1):
        for poi in self.points_of_interest:
            poi_x, poi_y = poi['coordinates']
            distance = math.sqrt((current_x - poi_x) ** 2 + (current_y - poi_y) ** 2)

            if distance <= threshold:
                return poi

        return None


class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ground_tiles = [['' for _ in range(width)] for _ in range(height)]
        self.object_tiles = [['' for _ in range(width)] for _ in range(height)]
        self.map = [[0 for _ in range(width)] for _ in range(height)]

        self.quadrants = []
        self.load_tiles()
        self.generate_environment()

    def load_tiles(self):
        base_path = 'assets/town_rpg_pack/graphics/elements/basictiles_tiles/'
        self.tiles = {
            'ground': {
                'spring_grass': pygame.image.load(f'{base_path}flowers.png').convert_alpha(),
                'summer_grass': pygame.image.load(f'{base_path}grass3.png').convert_alpha(),
                'autumn_grass': pygame.image.load(f'{base_path}grass2.png').convert_alpha(),
                'winter_grass': pygame.image.load(f'{base_path}grass2.png').convert_alpha(),
                'flowers': pygame.image.load(f'{base_path}flowers.png').convert_alpha(),
            },
            'object': {
                'spring_tree': pygame.image.load(f'{base_path}tree1.png').convert_alpha(),
                'summer_tree': pygame.image.load(f'{base_path}tree2.png').convert_alpha(),
                'autumn_tree': pygame.image.load(f'{base_path}tree1.png').convert_alpha(),
                'winter_tree': pygame.image.load(f'{base_path}tree2.png').convert_alpha(),
                'plant': pygame.image.load(f'{base_path}plant.png').convert_alpha(),
            }
        }

    def generate_environment(self):
        mid_x = self.width // 2
        mid_y = self.height // 2

        circle_radius = 15
        exit_angle = -math.pi / 4

        quadrant_width = self.width // 2
        quadrant_height = self.height // 2
        seasons = ['spring', 'summer', 'autumn', 'winter']

        for i, season in enumerate(seasons):
            x = (i % 2) * quadrant_width
            y = (i // 2) * quadrant_height
            quadrant = Quadrant(x, y, quadrant_width, quadrant_height, season)

            for y in range(quadrant.height):
                for x in range(quadrant.width):
                    quadrant.ground_tiles[y][x] = f'{season}_grass'
                    quadrant.map[y][x] = 0

            self.quadrants.append(quadrant)

        # Generate central circular area with flowers
        for y in range(self.height):
            for x in range(self.width):
                dx = x - mid_x
                dy = y - mid_y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance <= circle_radius:
                    self.ground_tiles[y][x] = 'flowers'
                    self.map[y][x] = 0
                    if abs(distance - circle_radius) < 1:
                        angle = math.atan2(dy, dx)
                        if (abs(angle - exit_angle) > 0.3 and
                                abs(angle - (2 * math.pi + exit_angle)) > 0.3):
                            self.object_tiles[y][x] = 'plant'
                            self.map[y][x] = 1

        gate_width = 3

        for y in range(0, mid_y - circle_radius):
            if not (mid_y // 2 - gate_width <= y <= mid_y // 2 + gate_width):
                self.object_tiles[y][mid_x] = random.choice(['spring_tree', 'summer_tree'])
                self.map[y][mid_x] = 1
        for y in range(mid_y + circle_radius + 1, self.height):
            if not ((mid_y + self.height) // 2 - gate_width <= y <= (mid_y + self.height) // 2 + gate_width):
                self.object_tiles[y][mid_x] = random.choice(['spring_tree', 'summer_tree'])
                self.map[y][mid_x] = 1

        for x in range(0, mid_x - circle_radius):
            if not (mid_x // 2 - gate_width <= x <= mid_x // 2 + gate_width):
                self.object_tiles[mid_y][x] = random.choice(['spring_tree', 'summer_tree'])
                self.map[mid_y][x] = 1
        for x in range(mid_x + circle_radius + 1, self.width):
            if not ((mid_x + self.width) // 2 - gate_width <= x <= (mid_x + self.width) // 2 + gate_width):
                self.object_tiles[mid_y][x] = random.choice(['spring_tree', 'summer_tree'])
                self.map[mid_y][x] = 1

        forest_generator = ForestGenerator(
            self.width, self.height, self.map, self.object_tiles, seasons, self.quadrants
        )
        forest_generator.generate_forests()

        for quadrant in self.quadrants:
            quadrant.create_boss_ring(mid_x, mid_y)

        for quadrant in self.quadrants:
            for y in range(quadrant.height):
                for x in range(quadrant.width):
                    global_x = quadrant.x + x
                    global_y = quadrant.y + y

                    if self.ground_tiles[global_y][global_x] == '':
                        self.ground_tiles[global_y][global_x] = quadrant.ground_tiles[y][x]

                    if self.object_tiles[global_y][global_x] == '':
                        self.object_tiles[global_y][global_x] = quadrant.object_tiles[y][x]
                    if self.map[global_y][global_x] == 0:
                        self.map[global_y][global_x] = quadrant.map[y][x]

    def _find_forest_exit(self, quadrant):
        """Find the forest path exit for a given quadrant"""
        mid_x = self.width // 2
        mid_y = self.height // 2

        # Determine exit location based on quadrant position
        if quadrant.x == 0 and quadrant.y == 0:  # top-left
            return (mid_x // 2, 0)
        elif quadrant.x > 0 and quadrant.y == 0:  # top-right
            return (mid_x + mid_x // 2, 0)
        elif quadrant.x == 0 and quadrant.y > 0:  # bottom-left
            return (mid_x // 2, self.height - 1)
        else:  # bottom-right
            return (mid_x + mid_x // 2, self.height - 1)

    def is_collision(self, x, y):
        tile_x = int(x // TILE_SIZE)
        tile_y = int(y // TILE_SIZE)

        if (x < 0 or x + (TILE_SIZE - 1) >= self.width * TILE_SIZE or
                y < 0 or y + (TILE_SIZE - 1) >= self.height * TILE_SIZE):
            return True

        return self.map[tile_y][tile_x] == 1

    def get_walkable_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.width and 0 <= new_y < self.height and
                    not self.is_collision(new_x * TILE_SIZE, new_y * TILE_SIZE)):
                neighbors.append((new_x, new_y))
        return neighbors
