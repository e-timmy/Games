from config import ENEMIES_PER_ROW, ENEMY_ROWS


class Level:
    def __init__(self):
        self.current_level = 0  # Start at 0 so first next_level() call makes it 1
        self.enemies_per_row = ENEMIES_PER_ROW
        self.enemy_rows = len(ENEMY_ROWS)
        self.enemy_speed = 0.5
        self.enemy_shoot_chance_multiplier = 1.0

    def next_level(self):
        self.current_level += 1

        # Increase difficulty
        self.enemies_per_row = min(self.enemies_per_row + 1, 12)  # Max 12 enemies per row
        self.enemy_speed *= 1.2  # 20% speed increase each level
        self.enemy_shoot_chance_multiplier *= 1.1  # 10% more frequent shooting

        # Return new formation parameters
        return {
            'enemies_per_row': self.enemies_per_row,
            'enemy_rows': self.enemy_rows,
            'enemy_speed': self.enemy_speed,
            'shoot_chance_multiplier': self.enemy_shoot_chance_multiplier
        }