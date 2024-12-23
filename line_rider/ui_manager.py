import pygame
from constants import *


import pygame
from constants import *

class UIManager:
    def __init__(self):
        # Drawing tools on the left
        button_y = 50
        self.drawing_buttons = {
            "draw": pygame.Rect(HUD_LEFT_WIDTH // 2 - TOOL_BUTTON_SIZE // 2, button_y, TOOL_BUTTON_SIZE, TOOL_BUTTON_SIZE),
            "create_rider": pygame.Rect(HUD_LEFT_WIDTH // 2 - TOOL_BUTTON_SIZE // 2, button_y + TOOL_BUTTON_SIZE + 10, TOOL_BUTTON_SIZE, TOOL_BUTTON_SIZE),
            "erase": pygame.Rect(HUD_LEFT_WIDTH // 2 - TOOL_BUTTON_SIZE // 2, button_y + 2 * (TOOL_BUTTON_SIZE + 10), TOOL_BUTTON_SIZE, TOOL_BUTTON_SIZE),
            "clear": pygame.Rect(HUD_LEFT_WIDTH // 2 - TOOL_BUTTON_SIZE // 2, button_y + 3 * (TOOL_BUTTON_SIZE + 10), TOOL_BUTTON_SIZE, TOOL_BUTTON_SIZE),
            "finish_rider": pygame.Rect(HUD_LEFT_WIDTH // 2 - TOOL_BUTTON_SIZE // 2, button_y + 4 * (TOOL_BUTTON_SIZE + 10), TOOL_BUTTON_SIZE, TOOL_BUTTON_SIZE),
        }

        # Line thickness slider
        self.thickness_slider = pygame.Rect(10, button_y + 5 * (TOOL_BUTTON_SIZE + 10), HUD_LEFT_WIDTH - 20, 20)
        self.min_thickness = 1
        self.max_thickness = 10

        # Color palette
        self.color_buttons = []
        colors = [BLACK, RED, GREEN, BLUE, YELLOW, PURPLE]
        color_size = 20
        color_margin = 5
        color_x = 10
        color_y = self.thickness_slider.bottom + 10
        for color in colors:
            self.color_buttons.append((pygame.Rect(color_x, color_y, color_size, color_size), color))
            color_x += color_size + color_margin
            if color_x + color_size > HUD_LEFT_WIDTH:
                color_x = 10
                color_y += color_size + color_margin

        # Control buttons at the bottom
        self.control_buttons = {
            "play": pygame.Rect(WINDOW_WIDTH // 2 - CONTROL_BUTTON_SIZE // 2, WINDOW_HEIGHT - 60, CONTROL_BUTTON_SIZE, CONTROL_BUTTON_SIZE),
            "reset": pygame.Rect(WINDOW_WIDTH // 2 + CONTROL_BUTTON_SIZE, WINDOW_HEIGHT - 60, CONTROL_BUTTON_SIZE, CONTROL_BUTTON_SIZE),
        }

    def draw(self, screen, game_state):
        # Draw left HUD background
        pygame.draw.rect(screen, GRAY, (0, 0, HUD_LEFT_WIDTH, WINDOW_HEIGHT))

        # Draw buttons
        self._draw_buttons(screen, self.drawing_buttons, game_state)
        self._draw_buttons(screen, self.control_buttons, game_state)

        # Draw thickness slider
        pygame.draw.rect(screen, LIGHT_GRAY, self.thickness_slider)
        slider_pos = self._get_slider_pos(game_state.line_thickness)
        pygame.draw.circle(screen, BLACK, (slider_pos, self.thickness_slider.centery), 10)

        # Draw color palette
        for color_rect, color in self.color_buttons:
            pygame.draw.rect(screen, color, color_rect)
            if color == game_state.current_color:
                pygame.draw.rect(screen, WHITE, color_rect, 2)

        # Hide finish rider button when not in create rider mode
        if game_state.current_tool != "create_rider":
            pygame.draw.rect(screen, GRAY, self.drawing_buttons["finish_rider"])

    def _draw_buttons(self, screen, buttons, game_state):
        font = pygame.font.Font(None, 14)  # Small font for labels

        for name, rect in buttons.items():
            if game_state.current_tool == name or (name == "play" and game_state.is_playing):
                bg_color = WHITE
                symbol_color = BLACK
            else:
                bg_color = LIGHT_GRAY
                symbol_color = BLACK

            pygame.draw.rect(screen, bg_color, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)  # Button border

            # Draw symbols and add text labels
            if name == "draw":
                # Pencil icon
                pygame.draw.line(screen, symbol_color, (rect.left + 20, rect.top + 15),
                                 (rect.right - 20, rect.bottom - 15), 2)
                text = font.render("Draw", True, BLACK)
            elif name == "erase":
                # Eraser icon
                pygame.draw.rect(screen, symbol_color, (rect.centerx - 12, rect.centery - 8, 24, 16))
                text = font.render("Erase", True, BLACK)
            elif name == "clear":
                # X icon
                pygame.draw.line(screen, symbol_color, (rect.left + 15, rect.top + 15),
                                 (rect.right - 15, rect.bottom - 15), 2)
                pygame.draw.line(screen, symbol_color, (rect.left + 15, rect.bottom - 15),
                                 (rect.right - 15, rect.top + 15), 2)
                text = font.render("Clear", True, BLACK)
            elif name == "create_rider":
                # Stick figure icon
                pygame.draw.circle(screen, symbol_color, (rect.centerx, rect.top + 15), 8, 1)
                pygame.draw.line(screen, symbol_color, (rect.centerx, rect.top + 23),
                                 (rect.centerx, rect.bottom - 20), 1)
                text = font.render("Rider", True, BLACK)
            elif name == "finish_rider":
                # Checkmark icon
                pygame.draw.line(screen, symbol_color, (rect.left + 15, rect.centery),
                                 (rect.centerx, rect.bottom - 15), 2)
                pygame.draw.line(screen, symbol_color, (rect.centerx, rect.bottom - 15),
                                 (rect.right - 15, rect.top + 15), 2)
                text = font.render("Finish", True, BLACK)
            elif name == "play":
                if game_state.is_playing:
                    # Pause icon
                    pygame.draw.rect(screen, symbol_color, (rect.centerx - 8, rect.centery - 10, 5, 20))
                    pygame.draw.rect(screen, symbol_color, (rect.centerx + 3, rect.centery - 10, 5, 20))
                    text = font.render("Pause", True, BLACK)
                else:
                    # Play icon
                    pygame.draw.polygon(screen, symbol_color, [
                        (rect.centerx - 8, rect.centery - 10),
                        (rect.centerx - 8, rect.centery + 10),
                        (rect.centerx + 12, rect.centery)
                    ])
                    text = font.render("Play", True, BLACK)
            elif name == "reset":
                # Reset icon (circular arrow)
                pygame.draw.arc(screen, symbol_color, (rect.left + 10, rect.top + 10,
                                                       rect.width - 20, rect.height - 20),
                                0, 270, 2)
                pygame.draw.polygon(screen, symbol_color, [
                    (rect.right - 20, rect.top + 15),
                    (rect.right - 15, rect.top + 10),
                    (rect.right - 10, rect.top + 15)
                ])
                text = font.render("Reset", True, BLACK)
            else:
                text = font.render(name.capitalize(), True, BLACK)

            # Position text below the icon
            text_rect = text.get_rect(center=(rect.centerx, rect.bottom + 12))
            screen.blit(text, text_rect)

    def handle_click(self, pos):
        for buttons in [self.drawing_buttons, self.control_buttons]:
            for name, rect in buttons.items():
                if rect.collidepoint(pos):
                    return name

        for color_rect, color in self.color_buttons:
            if color_rect.collidepoint(pos):
                return ("color", color)

        return None

    def handle_slider(self, pos):
        if self.thickness_slider.collidepoint(pos):
            return self._get_thickness_from_pos(pos[0])
        return None

    def _get_slider_pos(self, thickness):
        ratio = (thickness - self.min_thickness) / (self.max_thickness - self.min_thickness)
        return int(self.thickness_slider.left + ratio * self.thickness_slider.width)

    def _get_thickness_from_pos(self, x):
        ratio = (x - self.thickness_slider.left) / self.thickness_slider.width
        return int(self.min_thickness + ratio * (self.max_thickness - self.min_thickness))
