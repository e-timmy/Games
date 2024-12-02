import pygame
import pymunk
import pymunk.pygame_util
import sys
from game_state import GameState
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from level import LevelManager

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0, 900)

game_state = GameState()
level_manager = LevelManager(space, game_state)


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                screen_pos = pygame.mouse.get_pos()
                world_pos = level_manager.screen_to_world(*screen_pos)
                level_manager.shoot_bullet(
                    level_manager.player.body.position,
                    pymunk.Vec2d(*world_pos)
                )

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        level_manager.player.try_grab_rope(level_manager.ropes)
    else:
        level_manager.player.release_rope()

    if keys[pygame.K_a]:
        level_manager.player.move_left()
    elif keys[pygame.K_d]:
        level_manager.player.move_right()
    else:
        level_manager.player.stop_horizontal_movement()

    if keys[pygame.K_w] and not level_manager.player.holding_rope:
        level_manager.player.jump()

    return True


def update():
    dt = 1.0 / FPS
    for _ in range(4):
        space.step(dt / 4)
    level_manager.update(dt)


def draw():
    screen.fill((255, 255, 255))
    level_manager.draw(screen)
    pygame.display.flip()


def reset_game():
    game_state.reset()
    level_manager.setup_level()


def game_loop():
    handler = space.add_collision_handler(3, 1)  # 3: bullet, 1: wall
    handler.begin = level_manager.handle_bullet_wall_collision

    running = True
    while running:
        running = handle_events()
        if game_state.is_game_over():
            show_game_over_screen()
            reset_game()
        elif game_state.level_complete:
            show_level_complete_screen()
        else:
            update()
            draw()
        clock.tick(FPS)


def show_menu():
    menu_font = pygame.font.Font(None, 36)
    title = menu_font.render("Swing Platformer", True, (0, 0, 0))
    start_text = menu_font.render("Press SPACE to Start", True, (0, 0, 0))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return

        screen.fill((255, 255, 255))
        screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, SCREEN_HEIGHT // 3))
        screen.blit(start_text, (SCREEN_WIDTH // 2 - start_text.get_width() // 2, SCREEN_HEIGHT // 2))
        pygame.display.flip()
        clock.tick(FPS)


def show_game_over_screen():
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(128)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    font = pygame.font.Font(None, 36)
    game_over_text = font.render("Game Over", True, (255, 255, 255))
    continue_text = font.render("Press SPACE to continue", True, (255, 255, 255))

    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 3))
    screen.blit(continue_text, (SCREEN_WIDTH // 2 - continue_text.get_width() // 2, SCREEN_HEIGHT // 2))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False


def show_level_complete_screen():
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(128)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    font = pygame.font.Font(None, 36)
    level_complete_text = font.render(f"Level {game_state.current_level} Complete!", True, (255, 255, 255))
    continue_text = font.render("Press SPACE for next level", True, (255, 255, 255))

    screen.blit(level_complete_text, (SCREEN_WIDTH // 2 - level_complete_text.get_width() // 2, SCREEN_HEIGHT // 3))
    screen.blit(continue_text, (SCREEN_WIDTH // 2 - continue_text.get_width() // 2, SCREEN_HEIGHT // 2))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
                game_state.next_level()
                level_manager.setup_level()


if __name__ == "__main__":
    reset_game()
    while True:
        show_menu()
        game_loop()