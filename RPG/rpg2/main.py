import random

import pygame

from CharacterManager import NPCManager, MonsterManager
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_SPEED, PLAYER_SPRINT_SPEED, TILE_SIZE, NPC_COUNT, MONSTER_COUNT, CHARACTER_SCALE_FACTOR
from environment import Environment
from player import Player
from display import Display

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("The Dispossessed")
    clock = pygame.time.Clock()

    environment = Environment(300, 300)

    # Calculate quadrant boundaries
    mid_x = environment.width // 2
    mid_y = environment.height // 2

    # Environment, player and display
    environment = Environment(300, 300)
    display = Display(screen)
    player = Player(mid_x * TILE_SIZE + TILE_SIZE * 3, mid_y * TILE_SIZE - TILE_SIZE * 3, environment)

    # Replace individual character spawning with managers
    npc_manager = NPCManager(environment)
    monster_manager = MonsterManager(environment)

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

            # Update NPCs and Monsters
            npc_manager.update()
            monster_manager.update(player)

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
            display.draw_environment(environment, player, npc_manager.npcs,
                                     monster_manager.monsters + monster_manager.bosses)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()