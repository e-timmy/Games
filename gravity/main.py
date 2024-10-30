import pygame
import sys
from game_objects import Player3D
from game_manager import GameManager3D, GameState
from math_3d import Vector3

def main():
    pygame.init()
    pygame.font.init()

    WINDOW_WIDTH = 1024
    WINDOW_HEIGHT = 768
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("3D Gravity Dodge")

    # Initialize game manager
    game_manager = GameManager3D(WINDOW_WIDTH, WINDOW_HEIGHT)
    clock = pygame.time.Clock()

    while True:
        delta_time = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            game_manager.handle_event(event)

        keys = pygame.key.get_pressed()
        game_manager.handle_input(keys, delta_time)
        game_manager.update(delta_time)
        game_manager.draw(screen)

        pygame.display.flip()

if __name__ == "__main__":
    main()