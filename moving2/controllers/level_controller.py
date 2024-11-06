from constants.game_constants import WINDOW_WIDTH
from entities.level import Level


class LevelController:
    def __init__(self, space):
        self.space = space
        self.current_level_number = 0
        self.transitioning = False
        self.camera_offset = 0
        self.levels = [Level(space, 0), Level(space, 1)]
        self.first_landing = False
        self.previous_level = None

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
        if len(self.levels) < 3:
            self.levels.append(Level(self.space, next_level_num))

    def update(self):
        if self.transitioning:
            self.camera_offset += 10
            if self.camera_offset >= WINDOW_WIDTH:
                self.complete_transition()

        for level in self.levels:
            level.update()

        if self.previous_level:
            self.previous_level.update()

    def complete_transition(self):
        if len(self.levels) > 2:
            oldest_level = self.levels.pop(0)
            if self.previous_level:  # Clean up the previous-previous level
                self.previous_level.cleanup(self.space)
            self.previous_level = oldest_level  # Store the current level as previous
            self.previous_level.start_wall_descent()  # Start wall descent after transition

        self.current_level_number += 1
        self.transitioning = False
        self.camera_offset = 0

    def draw(self, screen, camera):
        if self.previous_level:
            self.previous_level.draw(screen, camera)
        for level in self.levels:
            level.draw(screen, camera)

    def get_item_from_shape(self, shape):
        for level in self.levels:
            for item in level.items:
                if item.shape == shape and not item.collected:
                    return item, level
        return None, None

    def collect_item(self, item, level):
        if item and not item.collected:
            item.collected = True
            if level == self.levels[0]:  # If it's the current level
                level.start_wall_ascent()
            return True
        return False