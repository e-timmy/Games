import pygame

def create_player_sprite():
    surface = pygame.Surface((40, 40))
    surface.fill((0, 255, 0))
    return surface

def create_shield_sprite():
    surface = pygame.Surface((70, 50))
    surface.fill((0, 255, 0))
    return surface

def create_bullet_sprite():
    surface = pygame.Surface((4, 15))
    surface.fill((255, 255, 255))
    return surface