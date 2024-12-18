import pygame
from config import *

class ShieldPiece(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((SHIELD_PIECE_SIZE, SHIELD_PIECE_SIZE))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Shield:
    def __init__(self, x, y):
        self.pieces = pygame.sprite.Group()
        self.create_shield_pieces(x, y)

    def create_shield_pieces(self, start_x, start_y):
        for row_index, row in enumerate(SHIELD_SHAPE):
            for col_index, cell in enumerate(row):
                if cell == 1:
                    x = start_x + (col_index * SHIELD_PIECE_SIZE)
                    y = start_y + (row_index * SHIELD_PIECE_SIZE)
                    piece = ShieldPiece(x, y)
                    self.pieces.add(piece)