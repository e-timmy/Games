import pygame
from game import Game

def main():
    """Entry point for the 2D RPG game."""
    pygame.init()
    game = Game()
    game.run()
    pygame.quit()

if __name__ == "__main__":
    main()