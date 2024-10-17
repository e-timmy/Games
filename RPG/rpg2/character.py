from abc import ABC, abstractmethod
import pygame

class Character(ABC):
    def __init__(self, x, y, sprite_sheet, sprite_width, sprite_height):
        self.x = x
        self.y = y
        self.sprite_sheet = sprite_sheet
        self.sprite_width = sprite_width
        self.sprite_height = sprite_height
        self.direction = 'down'
        self.animation_frame = 1  # Start with the middle frame
        self.animation_speed = 0.15
        self.animation_time = 0
        self.is_moving = False
        self.sprites = self.load_sprites()

    def load_sprites(self):
        sprites = {}
        for i, direction in enumerate(['left', 'up', 'down']):
            sprites[direction] = [
                self.sprite_sheet.subsurface((j * self.sprite_width, i * self.sprite_height, self.sprite_width, self.sprite_height))
                for j in range(3)
            ]
        return sprites

    @abstractmethod
    def move(self, dx, dy, environment):
        pass

    def update(self, dt):
        if self.is_moving:
            self.animation_time += dt
            if self.animation_time >= self.animation_speed:
                self.animation_frame = (self.animation_frame + 1) % 3
                self.animation_time = 0
        else:
            self.animation_frame = 1  # Set to middle frame when static
            self.animation_time = 0  # Reset animation time

    def get_current_sprite(self):
        return self.sprites[self.direction][self.animation_frame]