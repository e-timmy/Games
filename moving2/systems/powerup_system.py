from enum import Enum

class PowerUpType(Enum):
    JUMP = 1
    SHOOT = 2
    GRAVITY = 3
    AOE = 4

class PowerUpSystem:
    def __init__(self):
        self.active_powerups = set()

    def add_powerup(self, powerup_type):
        self.active_powerups.add(powerup_type)

    def has_powerup(self, powerup_type):
        return powerup_type in self.active_powerups