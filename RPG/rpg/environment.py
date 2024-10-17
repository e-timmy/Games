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

        # Generate biomes
        self.generate_forest_biomes()
        self.generate_flower_fields()

    def generate_forest_biomes(self):
        # Create forest seeds
        forest_seeds = []
        for _ in range(self.width * self.height // 400):
            forest_seeds.append((
                random.randint(5, self.width-5),
                random.randint(5, self.height-5)
            ))

        # Grow forests using cellular automata
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

        # Place trees in forest areas
        for (x, y) in forest_cells:
            if random.random() < 0.6:
                self.object_tiles[y][x] = random.choice(['tree1', 'tree2'])
                self.map[y][x] = 1
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    if (0 <= x+dx < self.width and
                        0 <= y+dy < self.height and
                        random.random() < 0.3 and
                        not self.object_tiles[y+dy][x+dx]):
                        self.object_tiles[y+dy][x+dx] = 'plant'
                        self.map[y+dy][x+dx] = 1

    def generate_flower_fields(self):
        field_seeds = []
        for _ in range(self.width * self.height // 500):
            field_seeds.append((
                random.randint(5, self.width-5),
                random.randint(5, self.height-5)
            ))

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