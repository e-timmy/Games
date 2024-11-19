import pygame
from game_state import GameState
from game import Game
from menu import Menu

class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Chicken Crossing")
        self.clock = pygame.time.Clock()
        self.state = GameState.MENU
        self.game = None
        self.menu = Menu(self.screen)
        self.waiting_for_key_release = False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            # Wait for key release before processing new state
            if self.waiting_for_key_release:
                if not keys[pygame.K_SPACE] and not keys[pygame.K_RETURN]:
                    self.waiting_for_key_release = False
                continue

            if self.state == GameState.MENU:
                new_state = self.menu.update()
                if new_state == GameState.PLAYING:
                    selected_level = self.menu.get_selected_level()
                    self.game = Game(self.screen)
                    self.game.current_level = selected_level
                    self.game.reset(show_message=True)
                    self.state = GameState.PLAYING
                    self.waiting_for_key_release = True
            elif self.state == GameState.PLAYING:
                new_state = self.game.update()
                if new_state != GameState.PLAYING:  # State change requested
                    self.state = new_state
                    self.waiting_for_key_release = True
            elif self.state == GameState.RETRY:
                self.state = GameState.MENU
                self.waiting_for_key_release = True

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    game = Main()
    game.run()