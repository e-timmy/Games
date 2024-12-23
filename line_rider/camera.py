import pygame
from constants import *

class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
        self.smoothness = 0.1

    def set_target(self, target_pos):
        self.target_x = target_pos[0] - (WINDOW_WIDTH - HUD_LEFT_WIDTH) // 4
        self.target_y = target_pos[1] - WINDOW_HEIGHT // 2

    def follow(self, target_pos):
        self.set_target(target_pos)
        self.x += (self.target_x - self.x) * self.smoothness
        self.y += (self.target_y - self.y) * self.smoothness

    def world_to_screen(self, pos):
        screen_x = pos[0] - self.x
        screen_y = pos[1] - self.y
        if screen_x < HUD_LEFT_WIDTH:
            screen_x = HUD_LEFT_WIDTH
        return (screen_x, screen_y)

    def screen_to_world(self, pos):
        world_x = pos[0] + self.x
        world_y = pos[1] + self.y
        return (world_x, world_y)