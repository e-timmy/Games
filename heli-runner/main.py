import pygame
import pymunk
import sys
from game_state import GameState
from player import Player
from environment import Environment
from pickup import ShieldPickup, ScatterShotPickup, SlowDownPickup, AutoPilotPickup


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
        self.test_mode = False

        self.initial_scroll_speed = self.environment.base_scroll_speed
        self.speed_increase_rate = self.environment.speed_increment
        self.speed_slowdown_factor = 1.0
        self.game_time_started = 0

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
            elif self.game_state == GameState.PLAYING:
                if event.key == pygame.K_RETURN:
                    self.player.shoot(self.environment.offset)
                elif event.key == pygame.K_o:
                    self.activate_slow_down()
                elif event.key == pygame.K_p:
                    # Scatter shot activation moved here
                    visible_obstacles = [obs for obs in self.environment.obstacles
                                       if 0 <= obs.x + self.environment.offset <= self.width]
                    self.player.activate_scatter(self.environment.offset, visible_obstacles)
                elif event.key == pygame.K_i:
                    self.player.activate_autopilot()

        # Check for Shift key to enable/disable test mode
        keys = pygame.key.get_pressed()
        self.test_mode = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

    def update(self):
        if self.game_state == GameState.PLAYING:
            self.space.step(1 / 60.0)
            elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
            self.update_scroll_speed(elapsed_time)
            self.environment.update()
            self.player.update(self.environment.offset, self.environment)
            self.score = int(elapsed_time)

            if not self.test_mode:
                if not self.player.death_manager.is_active:
                    collision_result = self.environment.check_collision(self.player)
                    if collision_result[0]:
                        collision_type, collision_point = collision_result[1], collision_result[2]
                        if self.player.has_shield:
                            if not self.player.shield_active:
                                self.player.activate_shield()
                            else:
                                # Shield is already active, player survives this hit
                                pass
                        else:
                            self.player.start_death_animation(collision_type, self.environment.offset, collision_point)
                elif self.player.is_death_animation_complete():
                    self.game_state = GameState.GAME_OVER
            else:
                # Constrain player position in test mode
                self.player.body.position = pymunk.Vec2d(
                    self.width // 4,
                    max(self.player.size // 2, min(self.height - self.player.size // 2, self.player.body.position.y))
                )

            # Check pickup collisions
            for pickup in self.environment.pickups[:]:
                # Temporarily adjust pickup x position by offset for collision check
                pickup.x += self.environment.offset
                if pickup.collides_with(self.player):
                    print(f"Player collided with pickup of type: {type(pickup).__name__}")
                    if isinstance(pickup, ShieldPickup) and not self.player.has_shield:
                        self.player.add_shield()
                    elif isinstance(pickup, ScatterShotPickup) and not self.player.has_scatter:
                        self.player.add_scatter()
                    elif isinstance(pickup, SlowDownPickup) and not self.player.has_slow_down:
                        self.player.add_slow_down()
                    elif isinstance(pickup,
                                    AutoPilotPickup) and not self.player.has_autopilot and not self.player.autopilot_active:
                        self.player.add_autopilot()
                        print(f"Player collected autopilot pickup")
                    self.environment.pickups.remove(pickup)
                pickup.x -= self.environment.offset  # Restore original position

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

            # Handle scatter shot activation
            keys = pygame.key.get_pressed()
            if keys[pygame.K_p] and self.player.has_scatter:
                # Get visible obstacles
                visible_obstacles = [obs for obs in self.environment.obstacles
                                   if 0 <= obs.x + self.environment.offset <= self.width]
                self.player.activate_scatter(self.environment.offset, visible_obstacles)

            # Check pickup collisions
            for pickup in self.environment.pickups[:]:
                # Temporarily adjust pickup x position by offset for collision check
                pickup.x += self.environment.offset
                if pickup.collides_with(self.player):
                    print(f"Player collided with pickup of type: {type(pickup).__name__}")
                    if isinstance(pickup, ShieldPickup) and not self.player.has_shield:
                        self.player.add_shield()
                    elif isinstance(pickup, ScatterShotPickup) and not self.player.has_scatter:
                        self.player.add_scatter()
                    elif isinstance(pickup, SlowDownPickup) and not self.player.has_slow_down:
                        self.player.add_slow_down()
                        print(f"Player collected slow-down pickup (current scroll speed: {self.environment.scroll_speed:.2f})")
                    self.environment.pickups.remove(pickup)
                pickup.x -= self.environment.offset  # Restore original position

    def draw(self):
        self.screen.fill((0, 0, 0))

        if self.game_state == GameState.MENU:
            self.draw_menu("Press SPACE to Start" if not self.test_mode else "TEST MODE - Press SPACE to Start")
        elif self.game_state == GameState.PLAYING:
            self.environment.draw(self.screen)
            self.player.draw(self.screen)
            self.player.draw_autopilot_indicator(self.screen)  # Add this line
            self.draw_score()
            if self.test_mode:
                self.draw_test_mode_indicator()
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

    def draw_test_mode_indicator(self):
        test_mode_text = self.font.render("TEST MODE", True, (255, 255, 0))
        self.screen.blit(test_mode_text, (self.width - test_mode_text.get_width() - 10, 10))

    def draw_slow_down_indicator(self):
        slow_down_text = self.font.render("SLOW DOWN", True, (0, 255, 0))
        self.screen.blit(slow_down_text, (self.width - slow_down_text.get_width() - 10, 50))

    def start_game(self):
        self.game_state = GameState.PLAYING
        current_time = pygame.time.get_ticks() / 1000.0
        self.start_time = current_time * 1000  # Convert back to milliseconds for consistency
        self.game_time_started = current_time  # Initialize game time
        self.environment.reset()
        self.player.reset(self.width // 4, self.height // 2)
        self.environment.scroll_speed = self.initial_scroll_speed

    def restart_game(self):
        self.start_game()

    def activate_slow_down(self):
        print(f"Attempting to activate slow-down (Player has slow-down: {self.player.has_slow_down})")
        if self.player.activate_slow_down():
            old_speed = self.environment.scroll_speed
            # Reset the game time to now, effectively resetting the speed progression
            self.game_time_started = pygame.time.get_ticks() / 1000.0
            # Update the speed immediately
            self.environment.scroll_speed = self.initial_scroll_speed
            print(f"Slow down activated! Speed reduced from {old_speed:.2f} to {self.environment.scroll_speed:.2f}")
            return True
        return False

    def update_scroll_speed(self, elapsed_time):
        # Calculate speed based on time since last reset
        effective_time = pygame.time.get_ticks() / 1000.0 - self.game_time_started
        new_speed = min(self.initial_scroll_speed + (effective_time * self.speed_increase_rate), 8)
        self.environment.scroll_speed = new_speed


if __name__ == "__main__":
    game = Game()
    game.run()