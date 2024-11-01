import pygame

from constants.game_constants import BLACK


class RenderEngine:
    def __init__(self, screen):
        self.screen = screen

    def clear(self):
        self.screen.fill(BLACK)

    def present(self):
        pygame.display.flip()

    def draw_circle(self, pos, radius, color):
        pygame.draw.circle(self.screen, color, pos, radius)

    def draw_rect(self, rect, color):
        pygame.draw.rect(self.screen, color, rect)

    def draw_line(self, start, end, color, width=1):
        pygame.draw.line(self.screen, color, start, end, width)

    def draw_polygon(self, vertices, color):
        pygame.draw.polygon(self.screen, color, vertices)

    def draw_text(self, text, pos, color, font=None):
        if font is None:
            font = pygame.font.Font(None, 36)
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def draw_rect_coords(self, x, y, width, height, color):
        pygame.draw.rect(self.screen, color, (x, y, width, height))