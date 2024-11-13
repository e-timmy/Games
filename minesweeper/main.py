# main.py
import pygame
import sys
from game import Game
from menu import Menu
from constants import WINDOW_WIDTH, WINDOW_HEIGHT, CAPTION


class GameController:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(CAPTION)
        self.clock = pygame.time.Clock()

        # Initialize states
        self.menu = Menu(self.screen)
        self.game = None
        self.current_state = "menu"

    def run(self):
        while True:
            if self.current_state == "menu":
                self.current_state, difficulty = self.menu.run()
                if self.current_state == "game":
                    self.game = Game(self.screen, difficulty)
                elif self.current_state == "quit":
                    pygame.quit()
                    sys.exit()

            elif self.current_state == "game":
                self.current_state = self.game.run()
                if self.current_state == "menu":
                    self.menu = Menu(self.screen)

            self.clock.tick(60)


if __name__ == "__main__":
    game = GameController()
    game.run()