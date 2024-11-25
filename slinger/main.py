import pygame
import pymunk
import pymunk.pygame_util
import sys
import math
from game_objects import Player, Platform, Rope
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

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and player.rope is None:  # Left click
                mouse_pos = pygame.mouse.get_pos()
                angle = math.atan2(mouse_pos[1] - player.body.position.y, mouse_pos[0] - player.body.position.x)
                player.shoot_rope(angle, space)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click release
                player.holding_rope = False
                if player.rope and player.rope.player_joint:
                    space.remove(player.rope.player_joint)
                    player.rope.player_joint = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                player.jump()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player.move_left()
    elif keys[pygame.K_RIGHT]:
        player.move_right()
    else:
        player.stop_horizontal_movement()

    # Update holding_rope based on mouse state
    if pygame.mouse.get_pressed()[0]:  # Left mouse button
        if player.rope and player.rope.attached_to_ceiling:
            player.holding_rope = True
            if not player.rope.player_joint:
                player.rope.create_player_joint()
    else:
        player.holding_rope = False
        if player.rope and player.rope.player_joint:
            space.remove(player.rope.player_joint)
            player.rope.player_joint = None

    return True

def update():
    dt = 1.0 / FPS
    space.step(dt)
    player.update(dt)
    if player.body.position.y > SCREEN_HEIGHT and not game_state.level_complete:
        game_state.set_game_over()
    elif player.body.position.x > SCREEN_WIDTH and not game_state.level_complete:
        game_state.set_level_complete()

def draw():
    screen.fill((255, 255, 255))

    # Draw ceiling
    pygame.draw.rect(screen, (100, 100, 100), (0, 0, SCREEN_WIDTH, 10))

    # Only draw if level is not complete
    if not game_state.level_complete:
        space.debug_draw(draw_options)
        if player.rope:
            player.rope.draw(screen)

    # Draw level info
    font = pygame.font.Font(None, 36)
    level_text = font.render(f"Level: {game_state.current_level}", True, (0, 0, 0))
    screen.blit(level_text, (10, 10))

    if game_state.level_complete:
        level_complete_text = font.render("Level Complete! Press SPACE to continue", True, (0, 255, 0))
        screen.blit(level_complete_text, (SCREEN_WIDTH // 2 - level_complete_text.get_width() // 2, SCREEN_HEIGHT // 2))

    pygame.display.flip()

def reset_level():
    global player, platforms, space

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