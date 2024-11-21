import pymunk
import pygame
import math


class Player:
    def __init__(self, space, position, size, jump_height):
        self.width = size
        self.height = size
        mass = 1
        moment = pymunk.moment_for_box(mass, (self.width, self.height))
        self.body = pymunk.Body(mass, moment)
        self.body.position = position

        self.body.angular_velocity_limit = 15
        self.body.angular_damping = 0.9

        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))
        self.shape.friction = 0.7
        self.shape.elasticity = 0
        self.shape.collision_type = 3

        space.add(self.body, self.shape)

        self.jump_speed = jump_height
        self.move_speed = jump_height * 0.6  # Adjust movement speed relative to jump height
        self.on_ground = False
        self.jump_cooldown = 0
        self.JUMP_COOLDOWN_TIME = 5

    def update(self, keys, space):
        # Horizontal movement
        vel_x = 0
        if keys[pygame.K_LEFT]:
            vel_x = -self.move_speed
        elif keys[pygame.K_RIGHT]:
            vel_x = self.move_speed

        # Get current velocity
        current_vel = self.body.velocity

        # Apply horizontal velocity in the world coordinate system
        world_vel = (vel_x, current_vel.y)
        self.body.velocity = world_vel

        # Keep player within screen bounds
        if self.body.position.x < 0:
            self.body.position = (0, self.body.position.y)
            self.body.velocity = (0, self.body.velocity.y)
        elif self.body.position.x > 800:
            self.body.position = (800, self.body.position.y)
            self.body.velocity = (0, self.body.velocity.y)

        # Update jump cooldown
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1

    def jump(self):
        if self.on_ground and self.jump_cooldown == 0:
            jump_impulse = (0, -self.jump_speed * self.body.mass)
            self.body.apply_impulse_at_world_point(jump_impulse, self.body.position)
            self.on_ground = False
            self.jump_cooldown = self.JUMP_COOLDOWN_TIME

    def draw(self, screen):
        points = [self.body.local_to_world(v) for v in self.shape.get_vertices()]
        points = [(int(p.x), int(p.y)) for p in points]

        pygame.draw.polygon(screen, (200, 0, 0), points)
        pygame.draw.polygon(screen, (255, 0, 0), points, 2)

        center = (int(self.body.position.x), int(self.body.position.y))
        pygame.draw.circle(screen, (255, 255, 255), center, 2)


def collision_handler(arbiter, space, data):
    player = data['player']

    # Get collision normal
    normal = arbiter.normal

    # Check if collision is more vertical than horizontal
    if abs(normal.y) > abs(normal.x):
        player.on_ground = True

    return True