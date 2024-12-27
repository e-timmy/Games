import pygame
from menu import Menu
from pong_game import PongGame
from game_engine import GameEngine
from config import Config


def main():
    config = Config()
    engine = GameEngine(config)

    screen = pygame.display.set_mode((config.screen_width, config.screen_height))
    menu = Menu(screen)
    difficulty = menu.run()

    if difficulty is not None:
        game = PongGame(difficulty, config)
        engine.run(game)


if __name__ == "__main__":
    main()