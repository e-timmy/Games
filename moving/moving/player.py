import pymunk
import pygame
import math
from enum import Enum
from constants import *
from game_objects import PowerUpManager, PowerUpType, Bullet


class PlayerState(Enum):
    FALLING = 1
    GROUNDED = 2


class Player:
    def __init__(self, space, pos):
        # Create semi-circular physics body
        mass = 1
        vertices = self._create_semicircle_vertices(PLAYER_SIZE / 2, 10)  # 10 segments for smoothness
        moment = pymunk.moment_for_poly(mass, vertices)

        self.body = pymunk.Body(mass, moment)
        self.body.position = pos

        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.elasticity = 0.5
        self.shape.friction = 0.7
        self.shape.collision_type = 1

        self.state = PlayerState.FALLING
        self.powerup_manager = PowerUpManager()
        self.aim_angle = -math.pi / 2  # Start pointing up
        self.space = space
        self.bullets = []
        space.add(self.body, self.shape)

        # Constants
        self.JUMP_FORCE = JUMP_FORCE
        self.gravity_normal = True
        self.AIM_SPEED = 3
        self.ARM_LENGTH = PLAYER_SIZE * 0.6
        self.ARM_OFFSET = PLAYER_SIZE * 0.6  # New constant for arm position

    def _create_semicircle_vertices(self, radius, num_segments):
        """Create vertices for a semi-circle physics body"""
        vertices = []
        # Add the bottom corners first
        vertices.append((-radius, 0))
        vertices.append((radius, 0))

        # Add the arc segments
        for i in range(num_segments):
            angle = math.pi * i / (num_segments - 1)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertices.append((x, -y))  # Negative y to make flat side on bottom

        return vertices

    def update_aim(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.aim_angle -= math.radians(self.AIM_SPEED)
        if keys[pygame.K_RIGHT]:
            self.aim_angle += math.radians(self.AIM_SPEED)

        # Restrict angle to upper 180 degrees when gravity is normal
        if self.gravity_normal:
            if self.aim_angle > 0:
                self.aim_angle = 0
            if self.aim_angle < -math.pi:
                self.aim_angle = -math.pi
        else:
            if self.aim_angle > 0:
                self.aim_angle = 0
            if self.aim_angle < -math.pi:
                self.aim_angle = -math.pi

    def draw(self, screen, camera):
        pos = camera.apply(self.body.position.x, self.body.position.y)
        if pos[0] > -1000:
            # Draw the semi-circle body matching physics shape
            vertices = []
            for v in self.shape.get_vertices():
                x = v.rotated(self.body.angle) + self.body.position
                screen_pos = camera.apply(x.x, x.y)
                vertices.append(screen_pos)

            # Draw the shape
            pygame.draw.polygon(screen, BLUE, vertices)

            # Draw arm from slightly outside the semi-circle
            if self.gravity_normal:
                arm_start = (
                    pos[0] + math.cos(self.aim_angle) * self.ARM_OFFSET * 0.5,
                    pos[1] + math.sin(self.aim_angle) * self.ARM_OFFSET * 0.5
                )
                arm_end = (
                    pos[0] + math.cos(self.aim_angle) * self.ARM_LENGTH,
                    pos[1] + math.sin(self.aim_angle) * self.ARM_LENGTH
                )
            else:
                arm_start = (
                    pos[0] + math.cos(self.aim_angle) * self.ARM_OFFSET * 0.5,
                    pos[1] - math.sin(self.aim_angle) * self.ARM_OFFSET * 0.5
                )
                arm_end = (
                    pos[0] + math.cos(self.aim_angle) * self.ARM_LENGTH,
                    pos[1] - math.sin(self.aim_angle) * self.ARM_LENGTH
                )

            pygame.draw.line(screen, BLUE, arm_start, arm_end, 3)

        # Draw bullets
        for bullet in self.bullets:
            bullet_pos = camera.apply(bullet.body.position.x, bullet.body.position.y)
            if bullet_pos[0] > -1000:
                pygame.draw.circle(screen, RED,
                                   (int(bullet_pos[0]), int(bullet_pos[1])), 5)

    def jump(self):
        if (self.state == PlayerState.GROUNDED and
                self.powerup_manager.has_powerup(PowerUpType.JUMP)):
            # Get the visual direction of the arm
            direction = pygame.math.Vector2(
                math.cos(self.aim_angle),
                math.sin(self.aim_angle)
            )

            if not self.gravity_normal:
                # When gravity is reversed, only invert the X component
                direction.x *= -1
                # Y component stays as is to match the visual aim

            impulse = direction * self.JUMP_FORCE

            # Apply impulse from the edge of the semi-circle
            offset = (math.cos(self.aim_angle) * self.ARM_OFFSET * 0.5,
                      math.sin(self.aim_angle) * self.ARM_OFFSET * 0.5)

            self.body.apply_impulse_at_local_point((impulse.x, impulse.y), offset)
            self.state = PlayerState.FALLING

    def shoot(self):
        if self.powerup_manager.has_powerup(PowerUpType.SHOOT):
            direction = pygame.math.Vector2(
                math.cos(self.aim_angle),
                math.sin(self.aim_angle)
            )

            if not self.gravity_normal:
                direction.y *= -1

            # Calculate bullet start position from the arm's end
            arm_offset = self.ARM_OFFSET * 0.5
            start_pos = (
                self.body.position.x + math.cos(self.aim_angle) * self.ARM_LENGTH,
                self.body.position.y + (math.sin(self.aim_angle) * self.ARM_LENGTH
                                        if self.gravity_normal
                                        else -math.sin(self.aim_angle) * self.ARM_LENGTH)
            )

            bullet = Bullet(self.space, start_pos, (direction.x, direction.y))
            self.bullets.append(bullet)

    def toggle_gravity(self):
        if self.powerup_manager.has_powerup(PowerUpType.GRAVITY):
            self.gravity_normal = not self.gravity_normal
            # Flip the body
            self.body.angle = 0 if self.gravity_normal else math.pi
            if self.gravity_normal:
                self.space.gravity = (0, 981)
            else:
                self.space.gravity = (0, -981)

    def update_state(self):
        if abs(self.body.velocity.y) < 1:
            self.state = PlayerState.GROUNDED
        else:
            self.state = PlayerState.FALLING

    def update_bullets(self):
        self.bullets = [bullet for bullet in self.bullets
                        if -WINDOW_WIDTH <= bullet.body.position.x <= WINDOW_WIDTH * 4
                        and -WINDOW_HEIGHT <= bullet.body.position.y <= WINDOW_HEIGHT * 2]