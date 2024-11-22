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

        # Calculate required initial velocity for desired jump height
        # Using physics equation: h = v^2 / (2g)
        # Solve for v: v = sqrt(2gh)
        gravity = abs(space.gravity[1])
        self.jump_speed = math.sqrt(2 * gravity * (jump_height + 5))  # Add 5 for safety margin

        self.move_speed = 300
        self.on_ground = False
        self.on_ladder = False
        self.current_ladder = None
        self.jump_cooldown = 0
        self.JUMP_COOLDOWN_TIME = 5

    def update(self, keys, space, ladders):
        near_ladder = False
        nearest_ladder = None

        # Only check for ladders if we're not already on one
        if not self.on_ladder:
            # Find the nearest ladder
            for ladder in ladders:
                # Check if player is near a ladder
                if (abs(self.body.position.x - (ladder.x + ladder.width / 2)) < 20 and
                        ladder.y1 <= self.body.position.y <= ladder.y2):
                    near_ladder = True
                    nearest_ladder = ladder
                    break

            # Try to get on ladder
            if near_ladder and (keys[pygame.K_UP] or keys[pygame.K_DOWN]):
                self.on_ladder = True
                self.current_ladder = nearest_ladder
                # Center the player on the ladder
                self.body.position = (nearest_ladder.x + nearest_ladder.width / 2, self.body.position.y)
                self.body.velocity = (0, 0)

        # Ladder movement
        if self.on_ladder:
            # Disable gravity while on ladder
            self.body.gravity = (0, 0)

            move_y = 0
            if keys[pygame.K_UP]:
                move_y = -4
            elif keys[pygame.K_DOWN]:
                move_y = 4

            if move_y != 0:
                new_y = self.body.position.y + move_y

                # Check ladder boundaries
                if new_y <= self.current_ladder.y1:
                    # Reached top of ladder
                    self.body.position = (self.body.position.x, self.current_ladder.y1 - self.height / 2)
                    self.on_ladder = False
                    self.body.gravity = space.gravity
                    self.body.velocity = (0, -100)  # Small upward boost
                elif new_y >= self.current_ladder.y2:
                    # Reached bottom of ladder
                    self.body.position = (self.body.position.x, self.current_ladder.y2)
                    self.on_ladder = False
                    self.body.gravity = space.gravity
                else:
                    # Normal ladder movement
                    self.body.position = (self.body.position.x, new_y)

            # Get off ladder when moving horizontally
            if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
                self.on_ladder = False
                self.body.gravity = space.gravity
            else:
                # Keep player centered on ladder if not getting off
                self.body.position = (self.current_ladder.x + self.current_ladder.width / 2, self.body.position.y)
                self.body.velocity = (0, 0)
        else:
            # Normal movement
            self.body.gravity = space.gravity
            vel_x = 0
            if keys[pygame.K_LEFT]:
                vel_x = -self.move_speed
            elif keys[pygame.K_RIGHT]:
                vel_x = self.move_speed

            current_vel = self.body.velocity
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
        if (self.on_ground or self.on_ladder) and self.jump_cooldown == 0:
            jump_impulse = (0, -self.jump_speed * self.body.mass)
            self.body.apply_impulse_at_world_point(jump_impulse, self.body.position)
            self.on_ground = False
            self.on_ladder = False
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
    normal = arbiter.normal
    if abs(normal.y) > abs(normal.x):
        player.on_ground = True
    return True