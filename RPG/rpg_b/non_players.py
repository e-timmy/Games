import pygame
import random
from character import Character

class NPC(Character):
    def __init__(self, x, y, sprite_sheet, environment):
        super().__init__(x, y, sprite_sheet, 24, 32)
        self.environment = environment
        self.move_cooldown = 0
        self.move_delay = random.randint(1, 3)  # Random delay between movements

    def move(self, dx, dy, environment):
        new_x = self.x + dx
        new_y = self.y + dy

        if not environment.is_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.is_moving = True
        else:
            self.is_moving = False

    def update(self, dt):
        super().update(dt)

        self.move_cooldown += dt
        if self.move_cooldown >= self.move_delay:
            self.move_cooldown = 0
            self.move_delay = random.randint(1, 3)  # Set a new random delay

            # Generate random movement
            dx = random.randint(-1, 1) * self.sprite_width
            dy = random.randint(-1, 1) * self.sprite_height

            if dx != 0 or dy != 0:
                if abs(dx) > abs(dy):
                    self.direction = 'left' if dx < 0 else 'right'
                else:
                    self.direction = 'up' if dy < 0 else 'down'

                self.move(dx, dy, self.environment)

class Npc1(NPC):
    def __init__(self, x, y, environment):
        sprite_sheet = pygame.image.load('assets/town_rpg_pack/graphics/characters/characters.png').convert_alpha()
        super().__init__(x, y, sprite_sheet, environment)