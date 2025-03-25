import pygame
import sys
from constants import *
from game import Game
from render import Renderer


def main():
    # Initialize pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Jumper")
    clock = pygame.time.Clock()

    log_debug("Game starting")
    game = Game()
    renderer = Renderer(screen)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            game.handle_event(event)

        game.update()
        renderer.draw_game(game)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    log_debug("Game exiting")
    sys.exit()


if __name__ == "__main__":
    main()