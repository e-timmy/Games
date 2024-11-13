# menu.py
import pygame
from constants import *
from difficulty import Difficulties


class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 24)
        self.difficulties = Difficulties.get_all()
        self.hover_difficulty = None

    def get_difficulty_positions(self):
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        positions = []
        total_width = (len(self.difficulties) - 1) * DIFFICULTY_SPACING
        start_x = center_x - total_width // 2

        for i in range(len(self.difficulties)):
            x = start_x + i * DIFFICULTY_SPACING
            positions.append((x, center_y))

        return positions

    def draw_tooltip(self, difficulty, pos):
        lines = [
            f"{difficulty.name}",
            f"Grid: {difficulty.grid_size}x{difficulty.grid_size}",
            f"Mines: {difficulty.mines}",
            f"Detector uses: {difficulty.detector_uses}"
        ]
        if difficulty.best_time != float('inf'):
            lines.append(f"Best Time: {difficulty.best_time:.1f}s")

        # Calculate tooltip dimensions
        line_height = 25
        tooltip_height = line_height * len(lines) + TOOLTIP_PADDING * 2
        tooltip_width = max(self.small_font.size(line)[0] for line in lines) + TOOLTIP_PADDING * 2

        # Draw tooltip background
        tooltip_x = pos[0] - tooltip_width // 2
        tooltip_y = pos[1] - tooltip_height - DIFFICULTY_BUTTON_RADIUS - 10
        pygame.draw.rect(self.screen, WHITE,
                         (tooltip_x, tooltip_y, tooltip_width, tooltip_height))
        pygame.draw.rect(self.screen, BLACK,
                         (tooltip_x, tooltip_y, tooltip_width, tooltip_height), 1)

        # Draw text
        for i, line in enumerate(lines):
            text = self.small_font.render(line, True, BLACK)
            text_rect = text.get_rect(centerx=pos[0],
                                      top=tooltip_y + TOOLTIP_PADDING + i * line_height)
            self.screen.blit(text, text_rect)

    def draw(self):
        self.screen.fill(GREEN)

        # Draw title
        title = self.font.render("Garden Minesweeper", True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3))
        self.screen.blit(title, title_rect)

        # Draw difficulty buttons
        positions = self.get_difficulty_positions()
        mouse_pos = pygame.mouse.get_pos()

        for difficulty, pos in zip(self.difficulties, positions):
            # Draw button
            pygame.draw.circle(self.screen, difficulty.color, pos, DIFFICULTY_BUTTON_RADIUS)
            pygame.draw.circle(self.screen, BLACK, pos, DIFFICULTY_BUTTON_RADIUS, 2)

            # Check hover
            distance = ((mouse_pos[0] - pos[0]) ** 2 + (mouse_pos[1] - pos[1]) ** 2) ** 0.5
            if distance <= DIFFICULTY_BUTTON_RADIUS:
                self.hover_difficulty = (difficulty, pos)

        # Draw tooltip for hovered difficulty
        if self.hover_difficulty:
            self.draw_tooltip(*self.hover_difficulty)

        return positions

    def run(self):
        positions = self.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit", None

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                # Check if any difficulty button was clicked
                for difficulty, pos in zip(self.difficulties, positions):
                    distance = ((mouse_pos[0] - pos[0]) ** 2 +
                                (mouse_pos[1] - pos[1]) ** 2) ** 0.5
                    if distance <= DIFFICULTY_BUTTON_RADIUS:
                        return "game", difficulty

        pygame.display.flip()
        return "menu", None