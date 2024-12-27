import pygame
import pygame
from game import Game
from menu import Menu


def main():
    pygame.init()
    screen = pygame.display.set_mode((1024, 768))
    pygame.display.set_caption("Racing Game")

    while True:
        # Show menu and get configurations
        menu = Menu(screen)
        player_configs, target_laps = menu.run()

        if not player_configs:
            break

        # Start game with configurations
        game = Game(screen, player_configs, target_laps)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return

            if game.update():  # Game finished
                break

            game.draw()
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
