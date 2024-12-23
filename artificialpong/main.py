import pygame
from game import Game
from menu import Menu


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    menu = Menu(screen)
    difficulty = menu.run()

    if difficulty is not None:
        game = Game(difficulty)
        game.run()

    pygame.quit()


if __name__ == "__main__":
    main()