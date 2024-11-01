import pygame
from entities.level import Level
from constants.game_constants import *

class LevelController:
    def __init__(self, space):
        self.space = space
        self.current_level_number = 0
        self.transitioning = False
        self.camera_offset = 0
        self.levels = [Level(space, 0), Level(space, 1)]
        self.first_landing = False

    def check_level_complete(self, player_pos):
        level_boundary = (self.current_level_number + 1) * WINDOW_WIDTH
        if not self.transitioning and player_pos.x >= level_boundary:
            self.start_transition()
            return True
        return False

    def start_transition(self):
        self.transitioning = True
        self.add_next_level()

    def add_next_level(self):
        next_level_num = self.current_level_number + 2
        if len(self.levels) < 3:  # Keep only 2-3 levels loaded at a time
            self.levels.append(Level(self.space, next_level_num))

    def update(self):
        if self.transitioning:
            self.camera_offset += 10
            if self.camera_offset >= WINDOW_WIDTH:
                self.complete_transition()

        for level in self.levels:
            level.update()

    def complete_transition(self):
        if len(self.levels) > 2:
            oldest_level = self.levels.pop(0)
            oldest_level.cleanup(self.space)
        self.current_level_number += 1
        self.transitioning = False
        self.camera_offset = 0

    def draw(self, screen, camera):
        for level in self.levels:
            level.draw(screen, camera)

    def get_item_from_shape(self, shape):
        for level in self.levels:
            for item in level.items:
                if item.shape == shape and not item.collected:
                    level.remove_blocking_wall()
                    return item
        return None