import pygame
import pymunk
import sys
from player import Player, collision_handler
from level import Level
from menu import Menu
from settings import *

pygame.init()
pygame.font.init()



class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Platform Climber")
        self.clock = pygame.time.Clock()
        self.menu = Menu()
        self.state = MAIN_MENU
        self.current_level = 1
        self.reset_game()

    def reset_game(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)

        self.level = Level(self.current_level)
        self.level.generate_platforms(self.space)

        start_y = 600 - self.level.platform_spacing - 30
        self.player = Player(self.space, (WINDOW_WIDTH // 2, start_y),
                             self.level.player_size, self.level.get_jump_height())

        handler = self.space.add_collision_handler(3, 1)
        handler.data["player"] = self.player
        handler.begin = collision_handler

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.state == MAIN_MENU:
                        self.state = PLAYING
                    elif self.state == LEVEL_COMPLETE:
                        self.current_level += 1
                        self.reset_game()
                        self.state = PLAYING
                    elif self.state == GAME_OVER:
                        self.current_level = 1
                        self.reset_game()
                        self.state = MAIN_MENU
                    elif self.state == PLAYING:
                        self.player.jump()
        return True

    def update(self):
        if self.state == PLAYING:
            keys = pygame.key.get_pressed()
            self.player.update(keys, self.space, self.level.get_ladders())
            self.level.update(self.space)

            if self.level.check_finish(self.player):
                self.state = LEVEL_COMPLETE

            for ball in self.level.balls:
                if abs(ball.body.position.x - self.player.body.position.x) < 20 and \
                        abs(ball.body.position.y - self.player.body.position.y) < 20:
                    self.state = GAME_OVER

            self.space.step(1 / FPS)

    def draw(self):
        self.screen.fill((30, 30, 50))  # Dark blue background

        if self.state == MAIN_MENU:
            self.menu.draw_main_menu(self.screen)
        elif self.state == PLAYING:
            self.level.draw(self.screen)
            self.player.draw(self.screen)
        elif self.state == LEVEL_COMPLETE:
            self.level.draw(self.screen)
            self.player.draw(self.screen)
            self.menu.draw_overlay(self.screen, f'Level {self.current_level} Complete!')
        elif self.state == GAME_OVER:
            self.level.draw(self.screen)
            self.player.draw(self.screen)
            self.menu.draw_overlay(self.screen, 'Game Over')

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
    sys.exit()