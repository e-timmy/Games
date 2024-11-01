from enum import Enum


class PlayerState(Enum):
    FALLING = 1
    GROUNDED = 2
    AIMING = 3