import pymunk
import pygame
import math
from enum import Enum
from constants import *
from game_objects import PowerUpManager, PowerUpType, Bullet


class PlayerState(Enum):
    FALLING = 1
    GROUNDED = 2
    AIMING = 3


class Player:
    def __init__(self, space, pos):
        moment = pymunk.moment_for_box(1, (PLAYER_SIZE, PLAYER_SIZE))
        self.body = pymunk.Body(1, moment)
        self.body.position = pos
        self.shape = pymunk.Poly.create_box(self.body, (PLAYER_SIZE, PLAYER_SIZE))
        self.shape.elasticity = 0.5
        self.shape.friction = 0.7
        self.shape.collision_type = 1
        self.state = PlayerState.FALLING
        self.powerup_manager = PowerUpManager()
        self.aim_angle = 0
        self.charge_start = None
        self.space = space
        self.bullets = []
        space.add(self.body, self.shape)
        self.JUMP_FORCE = 500
        self.MAX_CHARGE_TIME = 1.5

    def start_aiming(self):
        if self.state == PlayerState.GROUNDED:
            self.state = PlayerState.AIMING
            self.charge_start = pygame.time.get_ticks()

    def jump(self):
        if (self.state == PlayerState.AIMING and
                self.powerup_manager.has_powerup(PowerUpType.JUMP)):
            charge_time = min((pygame.time.get_ticks() - self.charge_start) / 1000.0,
                              self.MAX_CHARGE_TIME)
            force = self.JUMP_FORCE * (0.5 + charge_time / self.MAX_CHARGE_TIME)

            direction = pygame.math.Vector2(
                math.cos(self.aim_angle),
                math.sin(self.aim_angle)
            )

            impulse = direction * force
            self.body.apply_impulse_at_local_point((impulse.x, impulse.y))
            self.state = PlayerState.FALLING

    def shoot(self):
        if (self.state == PlayerState.AIMING and
                self.powerup_manager.has_powerup(PowerUpType.SHOOT)):
            direction = pygame.math.Vector2(
                math.cos(self.aim_angle),
                math.sin(self.aim_angle)
            ).normalize()

            bullet = Bullet(self.space,
                            (self.body.position.x, self.body.position.y),
                            (direction.x, direction.y))
            self.bullets.append(bullet)

    def update_aim(self):
        if self.state == PlayerState.AIMING:
            self.aim_angle += ROTATION_SPEED * (1 / FPS)

    def update_state(self):
        if self.state != PlayerState.AIMING:
            if abs(self.body.velocity.y) < 1:
                self.state = PlayerState.GROUNDED
            else:
                self.state = PlayerState.FALLING

    def draw(self, screen, camera):
        # Draw player
        pos = camera.apply(self.body.position.x, self.body.position.y)
        pygame.draw.rect(screen, BLUE,
                         (pos[0] - PLAYER_SIZE / 2,
                          pos[1] - PLAYER_SIZE / 2,
                          PLAYER_SIZE, PLAYER_SIZE))

        # Draw aim indicator if aiming
        if self.state == PlayerState.AIMING:
            charge_time = (pygame.time.get_ticks() - self.charge_start) / 1000.0
            length = min(50 * charge_time, 100)

            end_x = pos[0] + math.cos(self.aim_angle) * length
            end_y = pos[1] + math.sin(self.aim_angle) * length

            pygame.draw.line(screen, YELLOW,
                             (pos[0], pos[1]),
                             (end_x, end_y), 3)

        # Draw bullets
        for bullet in self.bullets:
            pos = camera.apply(bullet.body.position.x, bullet.body.position.y)
            pygame.draw.circle(screen, RED, (int(pos[0]), int(pos[1])), 5)

    def update_bullets(self):
        # Remove bullets that have gone off screen
        self.bullets = [bullet for bullet in self.bullets
                        if 0 <= bullet.body.position.x <= WINDOW_WIDTH * 3
                        and 0 <= bullet.body.position.y <= WINDOW_HEIGHT]