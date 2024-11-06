import pygame
import math

from controllers import level_controller
from entities.player import Player
from systems.powerup_system import PowerUpSystem, PowerUpType
from entities.bullet import Bullet
from entities.aoe import AOEEffect
from entities.game_states import PlayerState
from constants.game_constants import *

class PlayerController:
    def __init__(self, space, level_controller):
        self.space = space
        self.level_controller = level_controller
        self.player = Player(space, (WINDOW_WIDTH // 2, 50))
        self.powerup_system = PowerUpSystem()
        self.bullets = []
        self.aoe_effect = None

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_j:
                self.jump()
            elif event.key == pygame.K_s:
                self.shoot()
            elif event.key == pygame.K_g:
                self.toggle_gravity()
            elif event.key == pygame.K_a:
                self.activate_aoe()

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player.aim_angle -= math.radians(ROTATION_SPEED)
        if keys[pygame.K_RIGHT]:
            self.player.aim_angle += math.radians(ROTATION_SPEED)

        self.player.update_state()
        self.update_bullets()
        self.update_aoe()

    def update_bullets(self):
        current_level_offset = self.level_controller.current_level_number * WINDOW_WIDTH
        min_x = current_level_offset - WINDOW_WIDTH
        max_x = current_level_offset + WINDOW_WIDTH * 2

        # Update bullet removal check
        for bullet in self.bullets[:]:
            if not (min_x <= bullet.body.position.x <= max_x and
                    -WINDOW_HEIGHT <= bullet.body.position.y <= WINDOW_HEIGHT * 2):
                self.space.remove(bullet.body, bullet.shape)
                self.bullets.remove(bullet)


    def jump(self):
        if (self.player.state == PlayerState.GROUNDED and
                self.powerup_system.has_powerup(PowerUpType.JUMP)):
            direction = pygame.math.Vector2(
                math.cos(self.player.aim_angle),
                math.sin(self.player.aim_angle)
            )
            self.player.body.apply_impulse_at_local_point(
                (direction.x * JUMP_FORCE, direction.y * JUMP_FORCE)
            )
            self.player.state = PlayerState.FALLING

    def shoot(self):
        if self.powerup_system.has_powerup(PowerUpType.SHOOT):
            direction = pygame.math.Vector2(
                math.cos(self.player.aim_angle),
                math.sin(self.player.aim_angle)
            )
            start_pos = (
                self.player.body.position.x + direction.x * PLAYER_SIZE,
                self.player.body.position.y + direction.y * PLAYER_SIZE
            )
            bullet = Bullet(self.space, start_pos, (direction.x, direction.y))
            self.bullets.append(bullet)

    def toggle_gravity(self):
        if self.powerup_system.has_powerup(PowerUpType.GRAVITY):
            direction = pygame.math.Vector2(
                math.cos(self.player.aim_angle),
                math.sin(self.player.aim_angle)
            )
            self.space.gravity = (direction.x * 981, direction.y * 981)

    def activate_aoe(self):
        if self.powerup_system.has_powerup(PowerUpType.AOE) and not self.aoe_effect:
            self.aoe_effect = AOEEffect(self.space, self.player.body.position)

    def update_aoe(self):
        if self.aoe_effect:
            self.aoe_effect.update()
            items_collected = self.level_controller.check_aoe_item_collection(self.aoe_effect)
            for item in items_collected:
                self.powerup_system.add_powerup(item.powerup_type)
            if self.aoe_effect.is_finished():
                self.aoe_effect = None

    def draw(self, screen, camera):
        self.player.draw(screen, camera)
        for bullet in self.bullets:
            bullet.draw(screen, camera)
        if self.aoe_effect:
            self.aoe_effect.draw(screen, camera)