import pygame
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_SPEED, PLAYER_SPRINT_SPEED, TILE_SIZE
from environment import Environment
from player import Player
from display import Display
from non_players import Npc1

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("The Dispossessed")
    clock = pygame.time.Clock()

    environment = Environment(300, 300)
    player = Player(environment.width // 2 * TILE_SIZE, environment.height // 2 * TILE_SIZE, environment)
    display = Display(screen)

    # Create npc1
    npc1 = Npc1(player.x + 50, player.y, environment)

    game_state = "menu"

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Get the time passed since last frame in seconds

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            player.is_sprinting = True
        else:
            player.is_sprinting = False

        if game_state == "playing":
            # Handle input and move player
            dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * (
                PLAYER_SPRINT_SPEED if player.is_sprinting else PLAYER_SPEED) * dt
            dy = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * (
                PLAYER_SPRINT_SPEED if player.is_sprinting else PLAYER_SPEED) * dt

            if dx != 0 or dy != 0:
                player.path = []
                player.move(dx, dy, environment)

            player.update(dt, environment)
            npc1.update(dt)  # Update npc1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and game_state == "menu":
                mouse_pos = pygame.mouse.get_pos()
                play_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 - 20, 100, 40)
                if play_button_rect.collidepoint(mouse_pos):
                    game_state = "playing"
            elif event.type == pygame.MOUSEBUTTONDOWN and game_state == "playing":
                mouse_pos = pygame.mouse.get_pos()
                target_x = player.x + (mouse_pos[0] - SCREEN_WIDTH // 2)
                target_y = player.y + (mouse_pos[1] - SCREEN_HEIGHT // 2)
                player.set_target(target_x, target_y, environment)

        if game_state == "menu":
            display.draw_menu()
        elif game_state == "playing":
            display.draw_environment(environment, player, [npc1])  # Pass npc1 to display

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()