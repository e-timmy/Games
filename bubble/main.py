import pygame
import sys

from colors import BACKGROUND_COLOR
from game_state import GameState
from settings import *


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bubble Absorption Game")
        self.clock = pygame.time.Clock()
        self.game_state = GameState()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * PLAYER_MAX_SPEED
        dy = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * PLAYER_MAX_SPEED
        # self.game_state.update_player_movement(dx, dy)
        return True

    def run(self):
        running = True
        while running:
            running = self.handle_events()

            # Update game state
            self.game_state.update()

            # Draw
            self.screen.fill(BACKGROUND_COLOR)
            self.game_state.draw(self.screen)
            pygame.display.flip()

            # Cap the frame rate
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()