# game.py
import pygame
from constants import *
from grid import Grid


class Game:
    def __init__(self, screen, difficulty):
        self.screen = screen
        self.difficulty = difficulty
        self.grid = Grid(difficulty)
        self.game_over = False
        self.won = False
        self.load_assets()
        self.metal_detector_active = False
        self.metal_detector_result = None
        self.flash_start_time = 0
        self.flash_position = None
        self.detector_uses_left = difficulty.detector_uses

        # Update metal detector position based on window size
        self.metal_detector_x = WINDOW_WIDTH - METAL_DETECTOR_SIZE - 20
        self.metal_detector_y = 20

    def load_assets(self):
        self.font = pygame.font.Font(None, 36)

    def draw_metal_detector_icon(self, x, y, size):
        pygame.draw.circle(self.screen, DARK_GRAY, (x + size // 2, y + size // 2), size // 3)
        pygame.draw.circle(self.screen, GRAY, (x + size // 2, y + size // 2), size // 3 - 2)
        handle_start = (x + size // 2, y + size // 2 + size // 4)
        handle_end = (x + size // 2 + size // 3, y + size - 5)
        pygame.draw.line(self.screen, DARK_GRAY, handle_start, handle_end, 4)
        uses_text = self.font.render(str(self.detector_uses_left), True, BLACK)
        uses_rect = uses_text.get_rect(center=(x + size // 2, y - 10))
        self.screen.blit(uses_text, uses_rect)

    def draw(self):
        self.screen.fill(GREEN)
        self.grid.draw(self.screen)

        if self.game_over:
            text = "Game Over! Press R to restart or M for menu"
            if self.won:
                text = "You Won! Press R to restart or M for menu"
            text_surface = self.font.render(text, True, BLACK)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, 30))
            self.screen.blit(text_surface, text_rect)

        pygame.draw.rect(self.screen, LIGHT_GREEN if self.metal_detector_active else WHITE,
                         (self.metal_detector_x, self.metal_detector_y,
                          METAL_DETECTOR_SIZE, METAL_DETECTOR_SIZE))
        self.draw_metal_detector_icon(self.metal_detector_x, self.metal_detector_y, METAL_DETECTOR_SIZE)

        if self.metal_detector_result is not None and self.flash_position is not None:
            current_time = pygame.time.get_ticks()
            if current_time - self.flash_start_time < FLASH_DURATION:
                total_grid_size = self.grid.difficulty.grid_size * self.grid.cell_size
                offset_x = (WINDOW_WIDTH - total_grid_size) // 2
                offset_y = (WINDOW_HEIGHT - total_grid_size) // 2
                flash_x = offset_x + self.flash_position[0] * self.grid.cell_size
                flash_y = offset_y + self.flash_position[1] * self.grid.cell_size
                color = RED if self.metal_detector_result else SAFE_GREEN
                pygame.draw.rect(self.screen, color,
                                 (flash_x, flash_y, self.grid.cell_size, self.grid.cell_size))
            else:
                self.metal_detector_result = None
                self.flash_position = None

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.__init__(self.screen, self.difficulty)
                elif event.key == pygame.K_m:
                    return "menu"

            if not self.game_over:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()

                    if (self.metal_detector_x <= x <= self.metal_detector_x + METAL_DETECTOR_SIZE and
                            self.metal_detector_y <= y <= self.metal_detector_y + METAL_DETECTOR_SIZE and
                            self.detector_uses_left > 0):
                        self.metal_detector_active = not self.metal_detector_active
                    else:
                        cell_pos = self.grid.get_cell_pos(x, y)
                        if cell_pos:
                            grid_x, grid_y = cell_pos
                            if self.metal_detector_active:
                                self.use_metal_detector(grid_x, grid_y)
                            elif event.button == 1:  # Left click
                                if self.grid.reveal(grid_x, grid_y):
                                    self.game_over = True
                            elif event.button == 3:  # Right click
                                self.grid.toggle_flag(grid_x, grid_y)

        keys = pygame.key.get_pressed()
        self.grid.test_mode = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        if self.grid.check_win():
            self.game_over = True
            self.won = True

        return None

    def use_metal_detector(self, x, y):
        if not self.grid.revealed[x][y] and self.detector_uses_left > 0:
            self.metal_detector_result = self.grid.cells[x][y] == -1
            self.flash_start_time = pygame.time.get_ticks()
            self.flash_position = (x, y)
            self.metal_detector_active = False
            self.detector_uses_left -= 1

    def run(self):
        result = self.handle_input()
        if result:
            return result

        self.draw()
        pygame.display.flip()
        return "game"