import pygame
from constants import *


class Renderer:
    def __init__(self, screen, camera):
        self.screen = screen
        self.camera = camera

    def draw_line(self, start_pos, end_pos, color, width=2):
        screen_start = self.camera.world_to_screen(start_pos)
        screen_end = self.camera.world_to_screen(end_pos)
        pygame.draw.line(self.screen, color, screen_start, screen_end, width)

    def draw_circle(self, pos, radius, color, width=0):
        screen_pos = self.camera.world_to_screen(pos)
        pygame.draw.circle(self.screen, color, screen_pos, radius, width)

    def draw_polygon(self, points, color):
        screen_points = [self.camera.world_to_screen(p) for p in points]
        pygame.draw.polygon(self.screen, color, screen_points)

    def draw_hud(self, ui_manager, game_state):
        # HUD is drawn in screen coordinates directly
        ui_manager.draw(self.screen, game_state)