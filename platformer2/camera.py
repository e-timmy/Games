from constants import *


class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0

    def update(self, player_pos):
        target_x = player_pos[0] - SCREEN_WIDTH // 3
        # Smooth camera movement
        self.x += (target_x - self.x) * 0.1

    def get_offset(self):
        return (-int(self.x), -int(self.y))