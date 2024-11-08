import pygame
import pymunk
import sys
from game_state import GameState
from player import Player
from environment import Environment


class Game:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Cave Runner")

        self.space = pymunk.Space()
        self.space.gravity = (0, 900)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.game_state = GameState.MENU
        self.player = Player(self.space, self.width // 4, self.height // 2)
        self.environment = Environment(self.width, self.height)

        self.start_time = 0
        self.score = 0

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self.handle_input(event)

            self.update()
            self.draw()

            self.clock.tick(60)

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if self.game_state == GameState.MENU and event.key == pygame.K_SPACE:
                self.start_game()
            elif self.game_state == GameState.GAME_OVER and event.key == pygame.K_SPACE:
                self.restart_game()
            elif self.game_state == GameState.PLAYING and event.key == pygame.K_RETURN:
                self.player.shoot(self.environment.offset)

        if self.game_state == GameState.PLAYING:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.player.apply_thrust()

    def update(self):
        if self.game_state == GameState.PLAYING:
            self.space.step(1 / 60.0)
            self.environment.update()
            self.player.update(self.environment.offset)
            self.score = int(pygame.time.get_ticks() - self.start_time) // 1000

            if not self.player.death_manager.is_active:
                collision_result = self.environment.check_collision(self.player)
                if collision_result[0]:
                    collision_type, collision_point = collision_result[1], collision_result[2]
                    self.player.start_death_animation(collision_type, self.environment.offset, collision_point)
            elif self.player.is_death_animation_complete():
                self.game_state = GameState.GAME_OVER

            # Handle bullet collisions
            for bullet in self.player.bullets[:]:
                for obstacle in self.environment.obstacles[:]:
                    if obstacle.collides_with(bullet):
                        self.environment.remove_obstacle(obstacle)
                        if bullet in self.player.bullets:
                            self.player.bullets.remove(bullet)
                        break

                # Check bullet collisions with comets
                for comet in self.environment.comets[:]:
                    if not comet.exploding and comet.collides_with(bullet):
                        comet.explode()
                        if bullet in self.player.bullets:
                            self.player.bullets.remove(bullet)
                        self.score += 10  # Bonus points for shooting a comet
                        break

    def draw(self):
        self.screen.fill((0, 0, 0))

        if self.game_state == GameState.MENU:
            self.draw_menu("Press SPACE to Start")
        elif self.game_state == GameState.PLAYING:
            self.environment.draw(self.screen)
            self.player.draw(self.screen)
            self.draw_score()
        elif self.game_state == GameState.GAME_OVER:
            self.draw_menu(f"Game Over! Score: {self.score}\nPress SPACE to Restart")

        pygame.display.flip()

    def draw_menu(self, text):
        text_lines = text.split('\n')
        for i, line in enumerate(text_lines):
            text_surface = self.font.render(line, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2 + i * 40))
            self.screen.blit(text_surface, text_rect)

    def draw_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

    def start_game(self):
        self.game_state = GameState.PLAYING
        self.start_time = pygame.time.get_ticks()
        self.environment.reset()
        self.player.reset(self.width // 4, self.height // 2)

    def restart_game(self):
        self.start_game()


if __name__ == "__main__":
    game = Game()
    game.run()