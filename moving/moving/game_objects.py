from enum import Enum
import pymunk

class PlayerState(Enum):
    FALLING = 1
    GROUNDED = 2
    AIMING = 3


class PowerUpType(Enum):
    JUMP = 1
    SHOOT = 2
    GRAVITY = 3


class Bullet:
    def __init__(self, space, pos, direction, speed=500):
        self.body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
        self.body.position = pos
        # This is the key line we need:
        self.body.velocity_func = lambda body, gravity, damping, dt: (body.velocity.x, body.velocity.y)

        self.shape = pymunk.Circle(self.body, 5)
        self.shape.collision_type = 4
        self.shape.elasticity = 0.8
        self.shape.friction = 0.5

        # Set initial velocity
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