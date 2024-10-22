import random


class MazeGenerator:
    def __init__(self, quadrant, entrance, boss_area, exit_point=None):
        """
        Initialize the maze generator

        :param quadrant: The quadrant to generate the maze in
        :param entrance: Coordinates of the quadrant entrance
        :param boss_area: Coordinates of the boss arena
        :param exit_point: Coordinates of the quadrant exit (optional)
        """
        self.quadrant = quadrant
        self.width = quadrant.width
        self.height = quadrant.height
        self.entrance = entrance
        self.boss_area = boss_area
        self.exit_point = exit_point

        # Initialize maze grid
        self.maze = [[1 for _ in range(self.width)] for _ in range(self.height)]

    def generate_maze(self):
        """
        Generate a maze using an iterative approach
        """
        # Track visited cells and create a stack for maze generation
        visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        stack = []

        # Helper function to get valid unvisited neighbors
        def get_unvisited_neighbors(x, y):
            neighbors = []
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                        not visited[ny][nx]):
                    neighbors.append((nx, ny))
            return neighbors

        # Start from entrance
        start_x, start_y = self.entrance
        current_x, current_y = start_x, start_y

        # Ensure start is on grid and walkable
        current_x = max(0, min(current_x, self.width - 1))
        current_y = max(0, min(current_y, self.height - 1))

        stack.append((current_x, current_y))
        visited[current_y][current_x] = True
        self.maze[current_y][current_x] = 0

        # Iterative maze generation
        while stack:
            current_x, current_y = stack[-1]

            # Get unvisited neighbors
            unvisited_neighbors = get_unvisited_neighbors(current_x, current_y)

            if unvisited_neighbors:
                # Choose a random unvisited neighbor
                nx, ny = random.choice(unvisited_neighbors)

                # Mark path between current cell and neighbor
                wall_x = (current_x + nx) // 2
                wall_y = (current_y + ny) // 2

                self.maze[wall_y][wall_x] = 0  # Clear wall
                self.maze[ny][nx] = 0  # Mark neighbor cell

                visited[ny][nx] = True
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()

        # Ensure paths to key points
        self._connect_points()

        # Update quadrant map
        self._update_quadrant_map()

    def _connect_points(self):
        """
        Ensure paths between entrance, boss area, and exit
        """
        points_to_connect = [self.entrance, self.boss_area]
        if self.exit_point:
            points_to_connect.append(self.exit_point)

        for i in range(len(points_to_connect) - 1):
            start_x, start_y = points_to_connect[i]
            end_x, end_y = points_to_connect[i + 1]

            # Simple pathfinding to create connections
            current_x, current_y = start_x, start_y
            while current_x != end_x or current_y != end_y:
                # Move towards target
                dx = 1 if end_x > current_x else -1 if end_x < current_x else 0
                dy = 1 if end_y > current_y else -1 if end_y < current_y else 0

                # Prefer moving in one direction at a time
                if random.random() < 0.5:
                    current_x += dx
                else:
                    current_y += dy

                # Ensure within bounds
                current_x = max(0, min(current_x, self.width - 1))
                current_y = max(0, min(current_y, self.height - 1))

                # Carve the path
                self.maze[current_y][current_x] = 0

    def _update_quadrant_map(self):
        """Update the quadrant's map with the maze information"""
        for y in range(self.height):
            for x in range(self.width):
                # Preserve existing obstacles while incorporating maze
                if self.maze[y][x] == 0:
                    self.quadrant.map[y][x] = 0