import heapq
import math
from constants import TILE_SIZE

class Pathfinder:
    def __init__(self, environment):
        self.environment = environment

    def heuristic(self, a, b):
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.environment.width and
                    0 <= new_y < self.environment.height and
                    not self.environment.is_collision(new_x * TILE_SIZE, new_y * TILE_SIZE)):
                    neighbors.append((new_x, new_y))
        return neighbors

    def find_path(self, start, goal):
        start = (int(start[0] // TILE_SIZE), int(start[1] // TILE_SIZE))
        goal = (int(goal[0] // TILE_SIZE), int(goal[1] // TILE_SIZE))

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                break

            for next in self.get_neighbors(*current):
                new_cost = cost_so_far[current] + self.heuristic(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        if goal not in came_from:
            return None

        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()

        return [(x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2) for x, y in path]