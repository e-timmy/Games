from enum import Enum
import pymunk

class PlayerState(Enum):
    FALLING = 1
    GROUNDED = 2
    AIMING = 3


class PowerUpType(Enum):
    JUMP = 1
    SHOOT = 2


class Bullet:
    def __init__(self, space, pos, direction, speed=500):
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)  # Changed to KINEMATIC
        self.body.position = pos
        self.shape = pymunk.Circle(self.body, 5)
        self.shape.collision_type = 4  # Bullet collision type

        # Apply velocity in the direction of aim
        self.body.velocity = (direction[0] * speed, direction[1] * speed)

        space.add(self.body, self.shape)

class PowerUpManager:
    def __init__(self):
        self.active_powerups = set()

    def add_powerup(self, powerup_type):
        self.active_powerups.add(powerup_type)

    def has_powerup(self, powerup_type):
        return powerup_type in self.active_powerups

class Item:
    def __init__(self, space, pos, powerup_type):
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = pos
        self.shape = pymunk.Poly.create_box(self.body, (20, 20))
        self.shape.sensor = True
        self.shape.collision_type = 2
        self.powerup_type = powerup_type
        self.collected = False
        space.add(self.body, self.shape)