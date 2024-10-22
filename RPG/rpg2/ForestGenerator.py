import random
import math


class ForestGenerator:
    def __init__(self, width, height, map_grid, object_tiles, seasons, quadrants):
        self.width = width
        self.height = height
        self.map_grid = map_grid
        self.object_tiles = object_tiles
        self.seasons = seasons
        self.quadrants = quadrants
        self.forest_radius = 15

    def generate_forests(self):
        mid_x = self.width // 2
        mid_y = self.height // 2

        forest_configs = [
            {"center_x": mid_x, "center_y": mid_y // 2, "direction": "horizontal", "season": self.seasons[0],
             "quadrant_index": 0},
            {"center_x": mid_x, "center_y": (mid_y + self.height) // 2, "direction": "horizontal",
             "season": self.seasons[2], "quadrant_index": 2},
            {"center_x": mid_x // 2, "center_y": mid_y, "direction": "vertical", "season": self.seasons[3],
             "quadrant_index": 3},
            {"center_x": (mid_x + self.width) // 2, "center_y": mid_y, "direction": "vertical",
             "season": self.seasons[1], "quadrant_index": 1}
        ]

        for config in forest_configs:
            self.generate_forest_with_path(**config)

    def generate_forest_with_path(self, center_x, center_y, direction, season, quadrant_index):
        forest_density = 0.65
        forest_cells = set()
        attempts = 0

        while len(forest_cells) < int(math.pi * self.forest_radius ** 2 * forest_density) and attempts < 1000:
            dx = random.randint(-self.forest_radius, self.forest_radius)
            dy = random.randint(-self.forest_radius, self.forest_radius)

            fx, fy = center_x + dx, center_y + dy

            if (0 <= fx < self.width and 0 <= fy < self.height):
                dist_from_center = math.sqrt(dx * dx + dy * dy)
                if dist_from_center <= self.forest_radius and random.random() < 1 - (
                        dist_from_center / self.forest_radius) ** 2:
                    forest_cells.add((fx, fy))

            attempts += 1

        for _ in range(3):
            new_cells = set()
            for x, y in forest_cells:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < self.width and 0 <= new_y < self.height):
                        dist_from_center = math.sqrt((new_x - center_x) ** 2 + (new_y - center_y) ** 2)
                        if dist_from_center <= self.forest_radius and random.random() < 0.4:
                            new_cells.add((new_x, new_y))
            forest_cells.update(new_cells)

        for x, y in forest_cells:
            if (0 <= x < self.width and 0 <= y < self.height):
                self.object_tiles[y][x] = f'{season}_tree'
                self.map_grid[y][x] = 1

        path_cells = self.create_winding_path(center_x, center_y, direction, self.forest_radius)

        for x, y in path_cells:
            if (0 <= x < self.width and 0 <= y < self.height):
                self.map_grid[y][x] = 0
                self.object_tiles[y][x] = ''

        # Add forest exit points to the quadrant's points of interest
        path_exit_x, path_exit_y = self._get_path_exit(center_x, center_y, direction)

        # Access the correct quadrant and add the point of interest
        quadrant = self.quadrants[quadrant_index]
        quadrant.add_point_of_interest('forest_exit', (path_exit_x, path_exit_y))

    def create_winding_path(self, center_x, center_y, direction, forest_radius, path_width=1):
        amplitude = forest_radius // 3
        path_cells = set()

        if direction == "vertical":
            path_y_start = max(0, center_y - forest_radius)
            path_y_end = min(self.height - 1, center_y + forest_radius)

            for y in range(path_y_start, path_y_end + 1):
                progress = (y - path_y_start) / (path_y_end - path_y_start)
                offset = amplitude * math.sin(2 * math.pi * progress)
                path_center_x = int(center_x + offset)

                for dx in range(-path_width // 2, path_width // 2 + 1):
                    x = path_center_x + dx
                    if 0 <= x < self.width and 0 <= y < self.height:
                        path_cells.add((x, y))

        else:  # horizontal
            path_x_start = max(0, center_x - forest_radius)
            path_x_end = min(self.width - 1, center_x + forest_radius)

            for x in range(path_x_start, path_x_end + 1):
                progress = (x - path_x_start) / (path_x_end - path_x_start)
                offset = amplitude * math.sin(2 * math.pi * progress)
                path_center_y = int(center_y + offset)

                for dy in range(-path_width // 2, path_width // 2 + 1):
                    y = path_center_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        path_cells.add((x, y))

        return path_cells

    def _get_path_exit(self, center_x, center_y, direction):
        if direction == "vertical":
            exit_x = center_x + (self.forest_radius if random.random() < 0.5 else -self.forest_radius)
            exit_y = center_y
        else:  # horizontal
            exit_x = center_x
            exit_y = center_y + (self.forest_radius if random.random() < 0.5 else -self.forest_radius)

        return exit_x, exit_y