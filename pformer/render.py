import pygame
from constants import *

class Renderer:
    def __init__(self, screen):
        self.screen = screen

    def draw_game(self, game):
        self.screen.fill(BACKGROUND_COLOR)

        # Draw background grid
        for i in range(0, WIDTH, 40):
            pygame.draw.line(self.screen, (20, 20, 60), (i, 0), (i, HEIGHT), 1)
        for i in range(0, HEIGHT, 40):
            offset = (int(game.camera_y) // 2) % 40
            pygame.draw.line(self.screen, (20, 20, 60), (0, i - offset), (WIDTH, i - offset), 1)

        if game.state == TITLE_SCREEN:
            self.draw_title_screen(game)
        elif game.state in [PLAYING, ENTER_NAME, GAME_OVER]:
            self.draw_gameplay(game)
            if game.state == GAME_OVER:
                self.draw_game_over(game)
            elif game.state == ENTER_NAME:
                self.draw_name_entry(game)

        self.draw_debug_log()

    def draw_debug_log(self):
        for i, message in enumerate(debug_log[-5:]):
            text = small_font.render(message, True, (200, 200, 200))
            self.screen.blit(text, (10, HEIGHT - 100 + i * 20))

    def draw_ground(self, game):
        ground_screen_y = HEIGHT - GROUND_HEIGHT - int(game.camera_y)
        if 0 <= ground_screen_y <= HEIGHT:
            ground_rect = pygame.Rect(0, ground_screen_y, WIDTH, GROUND_HEIGHT)
            pygame.draw.rect(self.screen, GROUND_COLOR, ground_rect)
            pygame.draw.line(self.screen, (100, 100, 255), (0, ground_screen_y), (WIDTH, ground_screen_y), 2)

    def draw_title_screen(self, game):
        title = title_font.render("JUMPER", True, TITLE_COLOR)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 5))

        instruction = font.render("Press SPACE to Start", True, TEXT_COLOR)
        self.screen.blit(instruction, (WIDTH // 2 - instruction.get_width() // 2, HEIGHT // 2 - 30))

        control1 = font.render("Controls:", True, TEXT_COLOR)
        control2 = font.render("← → to move", True, TEXT_COLOR)

        self.screen.blit(control1, (WIDTH // 2 - control1.get_width() // 2, HEIGHT // 2 + 20))
        self.screen.blit(control2, (WIDTH // 2 - control2.get_width() // 2, HEIGHT // 2 + 60))

        if game.high_scores:
            high_score_title = font.render("TOP SCORES", True, TITLE_COLOR)
            self.screen.blit(high_score_title, (WIDTH // 2 - high_score_title.get_width() // 2, HEIGHT - 180))

            for i, (name, score) in enumerate(game.high_scores[:3]):
                text = font.render(f"{name}: {score}", True, TEXT_COLOR)
                self.screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 140 + i * 30))

    def draw_gameplay(self, game):
        self.draw_ground(game)

        platform_count = len(game.platforms)
        debug_text1 = small_font.render(f"Platforms: {platform_count} | Camera: {game.camera_y:.1f}", True, TEXT_COLOR)
        self.screen.blit(debug_text1, (10, 40))

        debug_text2 = small_font.render(
            f"Player: x={game.player.rect.x}, y={game.player.rect.y:.1f}, vy={game.player.velocity_y:.1f}", True,
            TEXT_COLOR)
        self.screen.blit(debug_text2, (10, 60))

        if game.platforms:
            highest_y = min(p.rect.y for p in game.platforms)
            lowest_y = max(p.rect.y for p in game.platforms)
            platform_range = f"{highest_y:.1f} to {lowest_y:.1f}"
            highest_text = small_font.render(f"Platform range: {platform_range}", True, TEXT_COLOR)
            self.screen.blit(highest_text, (10, 80))

            normal_count = len([p for p in game.platforms if p.platform_type == NORMAL_PLATFORM])
            moving_count = len([p for p in game.platforms if p.platform_type == MOVING_PLATFORM])
            breakable_count = len([p for p in game.platforms if p.platform_type == BREAKABLE_PLATFORM])

            platform_counts = small_font.render(
                f"Platform types - Normal: {normal_count}, Moving: {moving_count}, Breakable: {breakable_count}",
                True, TEXT_COLOR)
            self.screen.blit(platform_counts, (10, 100))

        for platform in game.platforms:
            platform_screen_y = int(platform.rect.y - game.camera_y)
            if 0 <= platform_screen_y <= HEIGHT and not platform.is_broken:
                draw_rect = pygame.Rect(
                    platform.rect.x,
                    platform_screen_y,
                    platform.rect.width,
                    platform.rect.height
                )
                pygame.draw.rect(self.screen, platform.color, draw_rect)

                if platform.platform_type == BREAKABLE_PLATFORM:
                    mid_x = draw_rect.x + draw_rect.width // 2
                    pygame.draw.line(self.screen, (100, 0, 0),
                                     (mid_x - 10, draw_rect.y),
                                     (mid_x + 10, draw_rect.y + PLATFORM_HEIGHT), 2)
                    pygame.draw.line(self.screen, (100, 0, 0),
                                     (mid_x + 10, draw_rect.y),
                                     (mid_x - 10, draw_rect.y + PLATFORM_HEIGHT), 2)

                elif platform.platform_type == MOVING_PLATFORM:
                    if platform.direction > 0:
                        arrow_x = draw_rect.right - 15
                        pygame.draw.line(self.screen, (200, 200, 0),
                                         (arrow_x, draw_rect.y + PLATFORM_HEIGHT // 2),
                                         (arrow_x + 10, draw_rect.y + PLATFORM_HEIGHT // 2), 2)
                        pygame.draw.line(self.screen, (200, 200, 0),
                                         (arrow_x + 10, draw_rect.y + PLATFORM_HEIGHT // 2),
                                         (arrow_x + 5, draw_rect.y + 2), 2)
                    else:
                        arrow_x = draw_rect.left + 15
                        pygame.draw.line(self.screen, (200, 200, 0),
                                         (arrow_x, draw_rect.y + PLATFORM_HEIGHT // 2),
                                         (arrow_x - 10, draw_rect.y + PLATFORM_HEIGHT // 2), 2)
                        pygame.draw.line(self.screen, (200, 200, 0),
                                         (arrow_x - 10, draw_rect.y + PLATFORM_HEIGHT // 2),
                                         (arrow_x - 5, draw_rect.y + 2), 2)

                pygame.draw.line(self.screen, (255, 255, 255, 150),
                                 (draw_rect.x, draw_rect.y),
                                 (draw_rect.x + draw_rect.width, draw_rect.y), 2)

        player_screen_y = int(game.player.rect.y - game.camera_y)
        if 0 <= player_screen_y <= HEIGHT:
            player_draw_rect = pygame.Rect(
                game.player.rect.x,
                player_screen_y,
                PLAYER_SIZE,
                PLAYER_SIZE
            )

            glow_surf = pygame.Surface((PLAYER_SIZE + 10, PLAYER_SIZE + 10), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (255, 0, 128, 100), (5, 5, PLAYER_SIZE, PLAYER_SIZE))
            self.screen.blit(glow_surf, (player_draw_rect.x - 5, player_draw_rect.y - 5))

            pygame.draw.rect(self.screen, PLAYER_COLOR, player_draw_rect)
            pygame.draw.line(self.screen, (255, 255, 255),
                             (player_draw_rect.x + 5, player_draw_rect.y + 5),
                             (player_draw_rect.x + 15, player_draw_rect.y + 5), 2)
            pygame.draw.line(self.screen, (255, 255, 255),
                             (player_draw_rect.x + 5, player_draw_rect.y + 5),
                             (player_draw_rect.x + 5, player_draw_rect.y + 15), 2)

        score_text = font.render(f"Score: {game.score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (10, 10))

    def draw_game_over(self, game):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        game_over_text = big_font.render("GAME OVER", True, TITLE_COLOR)
        self.screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 120))

        final_score_text = font.render(f"Final Score: {game.score}", True, TEXT_COLOR)
        self.screen.blit(final_score_text, (WIDTH // 2 - final_score_text.get_width() // 2, HEIGHT // 2 - 50))

        high_score_text = font.render("High Scores:", True, TEXT_COLOR)
        self.screen.blit(high_score_text, (WIDTH // 2 - high_score_text.get_width() // 2, HEIGHT // 2))

        for i, (name, score) in enumerate(game.high_scores[:5]):
            text = font.render(f"{i + 1}. {name}: {score}", True, TEXT_COLOR)
            self.screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 + 40 + i * 30))

        restart_text = font.render("Press R to Restart", True, TEXT_COLOR)
        self.screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT - 100))

        menu_text = font.render("Press ENTER for Menu", True, TEXT_COLOR)
        self.screen.blit(menu_text, (WIDTH // 2 - menu_text.get_width()// 2, HEIGHT - 60))

    def draw_name_entry(self, game):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        title_text = font.render("NEW HIGH SCORE!", True, TITLE_COLOR)
        self.screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 2 - 100))

        score_text = font.render(f"Score: {game.score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 - 50))

        instruction_text = font.render("Enter your initials:", True, TEXT_COLOR)
        self.screen.blit(instruction_text, (WIDTH // 2 - instruction_text.get_width() // 2, HEIGHT // 2))

        name_box = pygame.Rect(WIDTH // 2 - 75, HEIGHT // 2 + 50, 150, 50)
        pygame.draw.rect(self.screen, (50, 50, 100), name_box)
        pygame.draw.rect(self.screen, TEXT_COLOR, name_box, 2)

        name_text = big_font.render(''.join(game.current_name), True, TEXT_COLOR)
        self.screen.blit(name_text, (WIDTH // 2 - name_text.get_width() // 2, HEIGHT // 2 + 55))

        if len(game.current_name) < 3 and game.name_cursor_blink < 30:
            cursor_x = WIDTH // 2 - 40 + len(game.current_name) * 25
            cursor_text = big_font.render("_", True, TEXT_COLOR)
            self.screen.blit(cursor_text, (cursor_x, HEIGHT // 2 + 55))

        submit_text = font.render("Press ENTER to submit", True, TEXT_COLOR)
        self.screen.blit(submit_text, (WIDTH // 2 - submit_text.get_width() // 2, HEIGHT // 2 + 120))