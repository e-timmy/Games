import random


class MapGenerator:
    """Generates a procedural map using characters to represent different elements."""

    def __init__(self, width, height):
        """Initialize the map generator with given dimensions."""
        self.width = width
        self.height = height
        self.map = []

    def generate(self):
        """Generate a procedural map using cellular automata.

        Returns:
            A 2D array of characters representing the map.
        """
        # Initialize with random walls and floors
        self.map = [['#' if random.random() < 0.4 else '.' for _ in range(self.width)] for _ in range(self.height)]

        # Apply cellular automata rules
        for _ in range(4):
            new_map = [row[:] for row in self.map]
            for y in range(self.height):
                for x in range(self.width):
                    # Count walls in 3x3 neighborhood
                    wall_count = 0
                    for ny in range(y - 1, y + 2):
                        for nx in range(x - 1, x + 2):
                            if 0 <= nx < self.width and 0 <= ny < self.height and self.map[ny][nx] == '#':
                                wall_count += 1

                    # Apply rules
                    if self.map[y][x] == '#':
                        if wall_count < 3:
                            new_map[y][x] = '.'
                    else:
                        if wall_count > 4:
                            new_map[y][x] = '#'

            self.map = new_map

        # Ensure map borders are walls
        for y in range(self.height):
            self.map[y][0] = '#'
            self.map[y][self.width - 1] = '#'
        for x in range(self.width):
            self.map[0][x] = '#'
            self.map[self.height - 1][x] = '#'

        # Ensure center is walkable (for player start)
        center_x, center_y = self.width // 2, self.height // 2
        for y in range(center_y - 2, center_y + 3):
            for x in range(center_x - 2, center_x + 3):
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.map[y][x] = '.'

        return self.map

    def print_map(self):
        """Print the map to the console."""
        for row in self.map:
            print(''.join(row))