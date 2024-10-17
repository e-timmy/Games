import math

import pygame
import random
from constants import TILE_SIZE

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ground_tiles = [['' for _ in range(width)] for _ in range(height)]
        self.object_tiles = [['' for _ in range(width)] for _ in range(height)]
        self.map = [[0 for _ in range(width)] for _ in range(height)]
        self.load_tiles()
        self.generate_environment()

    def load_tiles(self):
        base_path = 'assets/town_rpg_pack/graphics/elements/basictiles_tiles/'
        self.tiles = {
            'ground': {
                'grass2': pygame.image.load(f'{base_path}grass2.png').convert_alpha(),
                'grass3': pygame.image.load(f'{base_path}grass3.png').convert_alpha(),
                'flowers': pygame.image.load(f'{base_path}flowers.png').convert_alpha(),
            },
            'object': {
                'tree1': pygame.image.load(f'{base_path}tree1.png').convert_alpha(),
                'tree2': pygame.image.load(f'{base_path}tree2.png').convert_alpha(),
                'plant': pygame.image.load(f'{base_path}plant.png').convert_alpha(),
            }
        }



    def generate_environment(self):
        # Initialize with basic grass
        for y in range(self.height):
            for x in range(self.width):
                self.ground_tiles[y][x] = 'grass2'
                self.object_tiles[y][x] = ''
                self.map[y][x] = 0

        # Calculate center of the map
        mid_x = self.width // 2
        mid_y = self.height // 2

        # Player's circular area parameters
        circle_radius = 15
        exit_angle = -math.pi / 4  # North-east exit

        # Create circular border of trees
        for y in range(self.height):
            for x in range(self.width):
                # Distance from center
                dx = x - mid_x
                dy = y - mid_y
                distance = math.sqrt(dx * dx + dy * dy)

                # Check if point is on the circle's edge
                if abs(distance - circle_radius) < 1:
                    # Determine if this is the exit point
                    angle = math.atan2(dy, dx)
                    if (abs(angle - exit_angle) > 0.3 and
                            abs(angle - (2 * math.pi + exit_angle)) > 0.3):
                        self.object_tiles[y][x] = random.choice(['tree1', 'tree2'])
                        self.map[y][x] = 1
                    else:
                        # Exit point, leave as open grass
                        self.ground_tiles[y][x] = 'grass2'
                        self.map[y][x] = 0
                elif distance < circle_radius:
                    # Fill the inside with flowers
                    self.ground_tiles[y][x] = 'flowers'
                    self.map[y][x] = 0

        # Add quadrant barriers with gates
        gate_width = 3  # Width of the gate passages

        # Vertical barrier
        for y in range(0, mid_y - circle_radius):
            if not (mid_y // 2 - gate_width <= y <= mid_y // 2 + gate_width):  # Skip the gate position
                self.object_tiles[y][mid_x] = random.choice(['tree1', 'tree2'])
                self.map[y][mid_x] = 1
        for y in range(mid_y + circle_radius + 1, self.height):
            if not ((mid_y + self.height) // 2 - gate_width <= y <= (
                    mid_y + self.height) // 2 + gate_width):  # Skip the gate position
                self.object_tiles[y][mid_x] = random.choice(['tree1', 'tree2'])
                self.map[y][mid_x] = 1

        # Horizontal barrier
        for x in range(0, mid_x - circle_radius):
            if not (mid_x // 2 - gate_width <= x <= mid_x // 2 + gate_width):  # Skip the gate position
                self.object_tiles[mid_y][x] = random.choice(['tree1', 'tree2'])
                self.map[mid_y][x] = 1
        for x in range(mid_x + circle_radius + 1, self.width):
            if not ((mid_x + self.width) // 2 - gate_width <= x <= (
                    mid_x + self.width) // 2 + gate_width):  # Skip the gate position
                self.object_tiles[mid_y][x] = random.choice(['tree1', 'tree2'])
                self.map[mid_y][x] = 1

        # Generate forests on quadrant boundaries
        self.generate_forest_on_boundary(mid_x, mid_y // 2, "vertical")  # Top
        self.generate_forest_on_boundary(mid_x, (mid_y + self.height) // 2, "vertical")  # Bottom
        self.generate_forest_on_boundary(mid_x // 2, mid_y, "horizontal")  # Left
        self.generate_forest_on_boundary((mid_x + self.width) // 2, mid_y, "horizontal")  # Right

        # Add some additional forest elements
        self.generate_forest_biomes()
        self.generate_flower_fields()

    def generate_forest_on_boundary(self, x, y, direction):
        forest_radius = 15
        forest_density = 0.65

        # Initialize forest cells
        forest_cells = set()
        attempts = 0
        while len(forest_cells) < int(math.pi * forest_radius ** 2 * forest_density) and attempts < 1000:
            # Generate points based on direction
            if direction == "vertical":
                # Distance from center line
                dx = random.gauss(0, forest_radius // 3)
                # Random along the line
                dy = random.randint(-forest_radius, forest_radius)
                fx = x + int(dx)
                fy = y + int(dy)
            else:  # horizontal
                # Distance from center line
                dy = random.gauss(0, forest_radius // 3)
                # Random along the line
                dx = random.randint(-forest_radius, forest_radius)
                fx = x + int(dx)
                fy = y + int(dy)

            # Check if point is valid
            if (0 <= fx < self.width and 0 <= fy < self.height and
                    not self.is_in_starting_circle(fx, fy)):
                # Use distance from center to create more natural edge
                dist_from_center = math.sqrt((fx - x) ** 2 + (fy - y) ** 2)
                if dist_from_center <= forest_radius and random.random() < 1 - (dist_from_center / forest_radius) ** 2:
                    forest_cells.add((fx, fy))

            attempts += 1

        # Grow forest
        for _ in range(3):
            new_cells = set()
            for cell in forest_cells:
                x, y = cell
                # Spread to neighboring cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        new_x, new_y = x + dx, y + dy
                        # Bias growth towards the center of the forest region
                        dist_from_center = math.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
                        if (0 <= new_x < self.width and 0 <= new_y < self.height and
                                not self.is_in_starting_circle(new_x, new_y) and
                                random.random() < 0.4 * (1 - dist_from_center / (forest_radius * 2))):
                            new_cells.add((new_x, new_y))
            forest_cells.update(new_cells)

        # Place trees and update map
        for (x, y) in forest_cells:
            if (0 <= x < self.width and 0 <= y < self.height):
                self.object_tiles[y][x] = random.choice(['tree1', 'tree2'])
                self.map[y][x] = 1

                # Randomly add some plants around trees
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < self.width and 0 <= new_y < self.height and
                            not self.object_tiles[new_y][new_x] and
                            random.random() < 0.2):
                        self.object_tiles[new_y][new_x] = 'plant'
                        self.map[new_y][new_x] = 1

    def is_in_starting_circle(self, x, y):
        mid_x = self.width // 2
        mid_y = self.height // 2
        circle_radius = 15

        dx = x - mid_x
        dy = y - mid_y
        distance = math.sqrt(dx * dx + dy * dy)

        return distance <= circle_radius

    def is_in_gate_passage(self, x, y):
        mid_x = self.width // 2
        mid_y = self.height // 2
        gate_width = 3  # Width of the gate passages

        # Check if the position is within the vertical gate passages
        if (x == mid_x and
                ((mid_y // 2 - gate_width <= y <= mid_y // 2 + gate_width) or
                 ((mid_y + self.height) // 2 - gate_width <= y <= (mid_y + self.height) // 2 + gate_width))):
            return True

        # Check if the position is within the horizontal gate passages
        if (y == mid_y and
                ((mid_x // 2 - gate_width <= x <= mid_x // 2 + gate_width) or
                 ((mid_x + self.width) // 2 - gate_width <= x <= (mid_x + self.width) // 2 + gate_width))):
            return True

    def generate_forest_biomes(self):
        # Create forest seeds
        forest_seeds = []
        for _ in range(self.width * self.height // 400):
            while True:
                seed_x = random.randint(5, self.width - 5)
                seed_y = random.randint(5, self.height - 5)
                if not self.is_in_starting_circle(seed_x, seed_y):
                    forest_seeds.append((seed_x, seed_y))
                    break

        # Rest of the method remains the same, but add a check in the main loop
        forest_cells = set()
        for seed in forest_seeds:
            forest_cells.add(seed)

        for _ in range(4):
            new_cells = set()
            for cell in forest_cells:
                x, y = cell
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        new_x, new_y = x + dx, y + dy
                        if (0 < new_x < self.width-1 and
                            0 < new_y < self.height-1 and
                            random.random() < 0.45):
                            new_cells.add((new_x, new_y))
            forest_cells.update(new_cells)

        for (x, y) in forest_cells:
            if not self.is_in_starting_circle(x, y) and not self.is_in_gate_passage(x, y):
                if random.random() < 0.6:
                    self.object_tiles[y][x] = random.choice(['tree1', 'tree2'])
                    self.map[y][x] = 1
                    # Similar check for surrounding tiles
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_x, new_y = x + dx, y + dy
                        if (0 <= new_x < self.width and
                                0 <= new_y < self.height and
                                not self.is_in_starting_circle(new_x, new_y) and
                                not self.is_in_gate_passage(new_x, new_y) and
                                random.random() < 0.3 and
                                not self.object_tiles[new_y][new_x]):
                            self.object_tiles[new_y][new_x] = 'plant'
                            self.map[new_y][new_x] = 1

    def generate_flower_fields(self):
        field_seeds = []
        for _ in range(self.width * self.height // 500):
            while True:
                seed_x = random.randint(5, self.width - 5)
                seed_y = random.randint(5, self.height - 5)
                if not self.is_in_starting_circle(seed_x, seed_y):
                    field_seeds.append((seed_x, seed_y))
                    break

        # Rest of the method remains similar, with checks added
        flower_cells = set()
        for seed in field_seeds:
            flower_cells.add(seed)

        for _ in range(3):
            new_cells = set()
            for cell in flower_cells:
                x, y = cell
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        new_x, new_y = x + dx, y + dy
                        if (0 < new_x < self.width-1 and
                            0 < new_y < self.height-1 and
                            random.random() < 0.55):
                            new_cells.add((new_x, new_y))
            flower_cells.update(new_cells)

        for (x, y) in flower_cells:
            if not self.is_in_starting_circle(x, y):
                if not self.object_tiles[y][x]:
                    if random.random() < 0.7:
                        self.ground_tiles[y][x] = 'flowers'

    def is_collision(self, x, y):
        tile_x = int(x // TILE_SIZE)
        tile_y = int(y // TILE_SIZE)

        # First check if the position would be outside the map boundaries
        # Add TILE_SIZE-1 to check the full tile space the player occupies
        if (x < 0 or x + (TILE_SIZE - 1) >= self.width * TILE_SIZE or
                y < 0 or y + (TILE_SIZE - 1) >= self.height * TILE_SIZE):
            return True

        # Then check for collision with objects
        if 0 <= tile_x < self.width and 0 <= tile_y < self.height:
            return self.map[tile_y][tile_x] == 1

        return True

    def get_walkable_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-directional movement
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.width and 0 <= new_y < self.height and
                    not self.is_collision(new_x * TILE_SIZE, new_y * TILE_SIZE)):
                neighbors.append((new_x, new_y))
        return neighbors