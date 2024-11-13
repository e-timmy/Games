# grid.py
import pygame
from constants import *


class Grid:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.cells = [[0 for _ in range(difficulty.grid_size)]
                      for _ in range(difficulty.grid_size)]
        self.revealed = [[False for _ in range(difficulty.grid_size)]
                         for _ in range(difficulty.grid_size)]
        self.flagged = [[False for _ in range(difficulty.grid_size)]
                        for _ in range(difficulty.grid_size)]
        self.test_mode = False
        self.calculate_cell_size()
        self.font = pygame.font.Font(None, max(12, self.cell_size - 4))
        self.place_mines()
        self.calculate_numbers()

    def calculate_cell_size(self):
        max_width = WINDOW_WIDTH - GRID_PADDING * 2
        max_height = WINDOW_HEIGHT - GRID_PADDING * 2
        self.cell_size = min(
            max_width // self.difficulty.grid_size,
            max_height // self.difficulty.grid_size,
            MAX_CELL_SIZE
        )
        self.cell_size = max(self.cell_size, MIN_CELL_SIZE)

    def place_mines(self):
        import random
        mines_placed = 0
        while mines_placed < self.difficulty.mines:
            x = random.randint(0, self.difficulty.grid_size - 1)
            y = random.randint(0, self.difficulty.grid_size - 1)
            if self.cells[x][y] != -1:
                self.cells[x][y] = -1
                mines_placed += 1

    def calculate_numbers(self):
        for x in range(self.difficulty.grid_size):
            for y in range(self.difficulty.grid_size):
                if self.cells[x][y] != -1:
                    self.cells[x][y] = self.count_adjacent_mines(x, y)

    def count_adjacent_mines(self, x, y):
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.difficulty.grid_size and 0 <= new_y < self.difficulty.grid_size:
                    if self.cells[new_x][new_y] == -1:
                        count += 1
        return count

    def draw_grass(self, screen, x, y):
        pygame.draw.rect(screen, GRASS_GREEN, (x, y, self.cell_size, self.cell_size))
        for i in range(3):
            start_x = x + (self.cell_size * (i + 1) // 4)
            pygame.draw.line(screen, GREEN,
                             (start_x, y + self.cell_size),
                             (start_x - 3, y + self.cell_size // 2),
                             2)

    def draw_mine(self, screen, x, y):
        center = (x + self.cell_size // 2, y + self.cell_size // 2)
        radius = self.cell_size // 3
        pygame.draw.circle(screen, BLACK, center, radius)
        for angle in range(0, 360, 45):
            import math
            rad = math.radians(angle)
            start_x = center[0] + radius * math.cos(rad)
            start_y = center[1] + radius * math.sin(rad)
            end_x = center[0] + (radius + 5) * math.cos(rad)
            end_y = center[1] + (radius + 5) * math.sin(rad)
            pygame.draw.line(screen, BLACK, (start_x, start_y), (end_x, end_y), 2)

    def draw_flag(self, screen, x, y):
        pygame.draw.line(screen, BLACK,
                         (x + self.cell_size // 1.5, y + self.cell_size // 1.2),
                         (x + self.cell_size // 1.5, y + self.cell_size // 4),
                         2)
        flag_points = [
            (x + self.cell_size // 1.5, y + self.cell_size // 4),
            (x + self.cell_size // 4, y + self.cell_size // 3),
            (x + self.cell_size // 1.5, y + self.cell_size // 2),
        ]
        pygame.draw.polygon(screen, RED, flag_points)

    def draw(self, screen):
        total_grid_size = self.difficulty.grid_size * self.cell_size
        offset_x = (WINDOW_WIDTH - total_grid_size) // 2
        offset_y = (WINDOW_HEIGHT - total_grid_size) // 2

        for x in range(self.difficulty.grid_size):
            for y in range(self.difficulty.grid_size):
                pos_x = offset_x + x * self.cell_size
                pos_y = offset_y + y * self.cell_size

                if self.revealed[x][y] or (self.test_mode and not self.flagged[x][y]):
                    pygame.draw.rect(screen, DIRT_BROWN, (pos_x, pos_y, self.cell_size, self.cell_size))
                    if self.cells[x][y] == -1:
                        self.draw_mine(screen, pos_x, pos_y)
                    elif self.cells[x][y] > 0:
                        text = self.font.render(str(self.cells[x][y]), True,
                                                NUMBER_COLORS.get(self.cells[x][y], BLACK))
                        text_rect = text.get_rect(center=(pos_x + self.cell_size // 2,
                                                          pos_y + self.cell_size // 2))
                        screen.blit(text, text_rect)
                else:
                    self.draw_grass(screen, pos_x, pos_y)
                    if self.flagged[x][y]:
                        self.draw_flag(screen, pos_x, pos_y)

                pygame.draw.rect(screen, GRID_LINE_COLOR,
                                 (pos_x, pos_y, self.cell_size, self.cell_size), 1)

    def reveal(self, x, y):
        if self.flagged[x][y]:
            return False

        if self.cells[x][y] == -1:
            self.revealed[x][y] = True
            return True

        if not self.revealed[x][y]:
            self.revealed[x][y] = True
            if self.cells[x][y] == 0:
                self.reveal_adjacent(x, y)
        return False

    def reveal_adjacent(self, x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.difficulty.grid_size and 0 <= new_y < self.difficulty.grid_size:
                    if not self.revealed[new_x][new_y] and not self.flagged[new_x][new_y]:
                        self.reveal(new_x, new_y)

    def toggle_flag(self, x, y):
        if not self.revealed[x][y]:
            self.flagged[x][y] = not self.flagged[x][y]

    def check_win(self):
        for x in range(self.difficulty.grid_size):
            for y in range(self.difficulty.grid_size):
                if self.cells[x][y] != -1 and not self.revealed[x][y]:
                    return False
        return True

    def get_cell_pos(self, screen_x, screen_y):
        total_grid_size = self.difficulty.grid_size * self.cell_size
        offset_x = (WINDOW_WIDTH - total_grid_size) // 2
        offset_y = (WINDOW_HEIGHT - total_grid_size) // 2

        grid_x = (screen_x - offset_x) // self.cell_size
        grid_y = (screen_y - offset_y) // self.cell_size

        if 0 <= grid_x < self.difficulty.grid_size and 0 <= grid_y < self.difficulty.grid_size:
            return grid_x, grid_y
        return None