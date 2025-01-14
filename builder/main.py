import pygame
import sys
from game_state import GameState
from character import Player, AI
from tile_manager import TileManager
from constants import *


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Building Helper")
        self.clock = pygame.time.Clock()

        # Initialize game components
        self.game_state = GameState()
        self.tile_manager = TileManager()
        self.player = Player(100, GROUND_LEVEL - CHAR_HEIGHT)
        self.ai = AI(50, GROUND_LEVEL - CHAR_HEIGHT)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN and not self.game_state.is_playing:
                # Handle tile placement in building mode
                mouse_pos = pygame.mouse.get_pos()
                grid_x = mouse_pos[0] // TILE_SIZE
                grid_y = mouse_pos[1] // TILE_SIZE
                self.tile_manager.place_tile(grid_x, grid_y, STONE_TILE)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.game_state.toggle_play()
                    if self.game_state.is_playing:
                        self.ai.plan_path(self.tile_manager.tiles)

        return True

    def update(self):
        if self.game_state.is_playing:
            keys = pygame.key.get_pressed()
            self.player.update(keys, self.tile_manager.tiles)
            self.ai.update(self.tile_manager.tiles)

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)

        # Draw tiles
        self.tile_manager.draw(self.screen)

        # Draw ground
        pygame.draw.rect(self.screen, GROUND_COLOR,
                         (0, GROUND_LEVEL, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_LEVEL))

        # Draw characters
        self.player.draw(self.screen)
        self.ai.draw(self.screen)

        # Draw UI elements
        if not self.game_state.is_playing:
            font = pygame.font.Font(None, 36)
            text = font.render("Building Mode - Press SPACE to start", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()