from queue import PriorityQueue
from constants import EMPTY_TILE

class Node:
    def __init__(self, position, g_cost, h_cost, parent=None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(position, tiles, can_move_func):
    neighbors = []
    # Include diagonal movements for more natural pathfinding
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for dx, dy in directions:
        new_pos = (position[0] + dx, position[1] + dy)
        if 0 <= new_pos[0] < len(tiles[0]) and 0 <= new_pos[1] < len(tiles):
            if can_move_func(new_pos):
                neighbors.append(new_pos)
    return neighbors


def a_star(start, goal, tiles, can_move_func):
    if not start or not goal:
        return None

    if not can_move_func(start) or not can_move_func(goal):
        return None

    start_node = Node(start, 0, heuristic(start, goal))
    open_list = PriorityQueue()
    open_list.put(start_node)
    closed_set = set()
    came_from = {}

    while not open_list.empty():
        current_node = open_list.get()

        if current_node.position == goal:
            # Reconstruct path
            path = []
            current = current_node.position
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current_node.position)

        for neighbor_pos in get_neighbors(current_node.position, tiles, can_move_func):
            if neighbor_pos in closed_set:
                continue

            g_cost = current_node.g_cost + 1
            if neighbor_pos[0] != current_node.position[0] and neighbor_pos[1] != current_node.position[1]:
                g_cost += 0.414  # Slightly higher cost for diagonal movement

            h_cost = heuristic(neighbor_pos, goal)
            neighbor = Node(neighbor_pos, g_cost, h_cost, current_node)

            if neighbor.position not in came_from or g_cost < came_from[neighbor.position][0]:
                came_from[neighbor.position] = (g_cost, current_node.position)
                open_list.put(neighbor)

    return None


# Debug function to visualize the path
def visualize_path(tiles, path):
    if not path:
        print("No path found")
        return

    visual = [['.' for _ in range(len(tiles[0]))] for _ in range(len(tiles))]
    for y in range(len(tiles)):
        for x in range(len(tiles[0])):
            if tiles[y][x] != EMPTY_TILE:
                visual[y][x] = '#'

    for i, (x, y) in enumerate(path):
        if i == 0:
            visual[y][x] = 'S'
        elif i == len(path) - 1:
            visual[y][x] = 'E'
        else:
            visual[y][x] = 'o'

    for row in visual:
        print(''.join(row))
    print()