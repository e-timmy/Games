import pygame
from config import Config
from game_event_manager import GameEventManager


class GameEngine:
    def __init__(self, config: Config):
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode((config.screen_width, config.screen_height))
        self.clock = pygame.time.Clock()
        self.event_manager = GameEventManager()

    def run(self, game):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                game.handle_events(event)

            game.update()
            game.render(self.screen)

            pygame.display.flip()
            self.clock.tick(self.config.fps)

        pygame.quit()