import pygame
import pymunk
import pymunk.pygame_util
import sys
import math
from game_objects import Platform, Rope
from player import Player
from game_state import GameState
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, FPS

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

space = pymunk.Space()
space.gravity = (0, 900)

game_state = GameState()

player = Player(50, SCREEN_HEIGHT - 100)
space.add(player.body, player.shape)

platforms = [
    Platform(0, SCREEN_HEIGHT - 50, 200, 50),
    Platform(SCREEN_WIDTH - 200, SCREEN_HEIGHT - 50, 200, 50),
    Platform(0, 0, SCREEN_WIDTH, 10)  # Ceiling
]
for platform in platforms:
    space.add(platform.body, platform.shape)

# Create a single rope in the center of the screen
rope = Rope(space, (SCREEN_WIDTH // 2, 0), SCREEN_HEIGHT - 50)


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                player.grab_rope(rope)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click release
                player.release_rope(rope)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        player.move_left()
    elif keys[pygame.K_d]:
        player.move_right()
    else:
        player.stop_horizontal_movement()

    if keys[pygame.K_w]:
        if not player.holding_rope:
            player.jump()

    return True


def update():
    dt = 1.0 / FPS
    for _ in range(4):
        space.step(dt / 4)
    player.update(dt)
    if player.body.position.y > SCREEN_HEIGHT:
        game_state.set_game_over()


def draw():
    screen.fill((255, 255, 255))

    draw_options = pymunk.pygame_util.DrawOptions(screen)
    draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    # Draw ceiling
    pygame.draw.rect(screen, (100, 100, 100), (0, 0, SCREEN_WIDTH, 10))

    space.debug_draw(draw_options)
    rope.draw(screen)

    # Draw level info
    font = pygame.font.Font(None, 36)
    level_text = font.render(f"Level: {game_state.current_level}", True, (0, 0, 0))
    screen.blit(level_text, (10, 10))

    pygame.display.flip()


def reset_level():
    global player, platforms, space, rope

    # Clean up all physics objects
    clean_level()

    # Create new player
    player = Player(50, SCREEN_HEIGHT - 100)
    space.add(player.body, player.shape)

    # Create new platforms
    platforms = [
        Platform(0, SCREEN_HEIGHT - 50, 200, 50),
        Platform(SCREEN_WIDTH - 200, SCREEN_HEIGHT - 50, 200, 50),
        Platform(0, 0, SCREEN_WIDTH, 10)  # Ceiling
    ]
    for platform in platforms:
        space.add(platform.body, platform.shape)

    # Create new rope
    rope = Rope(space, (SCREEN_WIDTH // 2, 0), SCREEN_HEIGHT - 50)


def clean_level():
    # Remove all physics objects from the space
    for body in space.bodies:
        space.remove(body)
    for shape in space.shapes:
        space.remove(shape)
    for constraint in space.constraints:
        space.remove(constraint)


def reset_game():
    game_state.reset()
    reset_level()


def game_loop():
    running = True
    while running:
        running = handle_events()
        if game_state.is_game_over():
            show_game_over_screen()
            reset_game()
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


if __name__ == "__main__":
    reset_game()
    while True:
        show_menu()
        game_loop()