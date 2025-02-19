import pygame
import sys
import math
from player import Player
from level import Level
from settings import *
from game_manager import GameManager


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Magical Platformer")
        self.clock = pygame.time.Clock()

        self.game_manager = GameManager()
        self.level = Level()
        self.reset_player()
        self.reset_animation = False
        self.level_transition = False
        self.fade_alpha = 0

    def reset_player(self):
        self.player = Player(100, SCREEN_HEIGHT - 150)  # Starting position

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self.player.handle_event(event)
        return True

    def update(self):
        if self.level_transition:
            self.fade_alpha += 15
            if self.fade_alpha >= 255:
                self.level.generate_new_level()
                self.level_transition = False
                self.start_reset_animation()
        else:
            self.fade_alpha = max(0, self.fade_alpha - 15)

        self.level.update()  # Update moving platforms
        self.player.update(self.level.get_platforms())
        if self.game_manager.check_win_condition(self.player):
            self.level_transition = True

    def start_reset_animation(self):
        self.reset_animation = True
        self.reset_animation_progress = 0
        self.start_pos = self.player.rect.center
        self.end_pos = (100, SCREEN_HEIGHT - 150)  # Starting position

    def update_reset_animation(self):
        self.reset_animation_progress += 0.02
        if self.reset_animation_progress >= 1:
            self.reset_animation = False
            self.reset_player()
            self.game_manager.win = False
        else:
            t = self.reset_animation_progress
            # Ease out cubic function
            t = 1 - (1 - t) ** 3
            x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t
            y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t

            # Add a slight arc to the movement
            arc_height = 100
            y -= math.sin(t * math.pi) * arc_height

            self.player.rect.center = (x, y)

    def draw(self):
        # Magical gradient background
        for i in range(SCREEN_HEIGHT):
            color = (
                int(20 + (i / SCREEN_HEIGHT) * 20),  # Dark blue
                int(10 + (i / SCREEN_HEIGHT) * 40),  # Deeper blue
                int(50 + (i / SCREEN_HEIGHT) * 100)  # Bright blue
            )
            pygame.draw.line(self.screen, color, (0, i), (SCREEN_WIDTH, i))

        # Draw sparkles (magical effect)
        self.game_manager.draw_effects(self.screen)

        # Draw level and player
        self.level.draw(self.screen)
        self.player.draw(self.screen)

        # Draw fade overlay for level transition
        if self.fade_alpha > 0:
            fade_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            fade_surface.fill((0, 0, 0))
            fade_surface.set_alpha(self.fade_alpha)
            self.screen.blit(fade_surface, (0, 0))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            if self.reset_animation:
                self.update_reset_animation()
            else:
                self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()