import pygame
from constants import *

class TileManager:
    def __init__(self):
        self.tiles = [[EMPTY_TILE for _ in range(SCREEN_WIDTH // TILE_SIZE)]
                     for _ in range(SCREEN_HEIGHT // TILE_SIZE)]

    def place_tile(self, grid_x, grid_y, tile_type):
        if 0 <= grid_x < len(self.tiles[0]) and 0 <= grid_y < len(self.tiles):
            self.tiles[grid_y][grid_x] = tile_type

    def draw(self, screen):
        for y in range(len(self.tiles)):
            for x in range(len(self.tiles[0])):
                if self.tiles[y][x] == STONE_TILE:
                    pygame.draw.rect(screen, STONE_COLOR,
                                   (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))