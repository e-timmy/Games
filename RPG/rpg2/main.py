import random

import pygame
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_SPEED, PLAYER_SPRINT_SPEED, TILE_SIZE, NPC_COUNT, MONSTER_COUNT, CHARACTER_SCALE_FACTOR
from environment import Environment
from player import Player
from display import Display
from non_player_character import NPC1, NPC2, NPC3, Alien, Slime, Bat, Ghost, Spider


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("The Dispossessed")
    clock = pygame.time.Clock()

    environment = Environment(300, 300)

    # Calculate quadrant boundaries
    mid_x = environment.width // 2
    mid_y = environment.height // 2

    player = Player(mid_x * TILE_SIZE + TILE_SIZE * 3, mid_y * TILE_SIZE - TILE_SIZE * 3, environment)

    # Initialize player in the north-east quadrant, blocked in by a circular area
    player = Player(mid_x * TILE_SIZE + TILE_SIZE * 3, mid_y * TILE_SIZE - TILE_SIZE * 3, environment)
    display = Display(screen)

    # Initialize NPCs in specific quadrants
    npc1 = NPC1()  # North-east
    npc1.x = (mid_x + mid_x // 2) * TILE_SIZE
    npc1.y = mid_y // 2 * TILE_SIZE

    npc2 = NPC2()  # South-east
    npc2.x = (mid_x + mid_x // 2) * TILE_SIZE
    npc2.y = (mid_y + mid_y // 2) * TILE_SIZE

    npc3 = NPC3()  # South-west
    npc3.x = mid_x // 2 * TILE_SIZE
    npc3.y = (mid_y + mid_y // 2) * TILE_SIZE

    alien = Alien()  # North-west
    alien.x = mid_x // 2 * TILE_SIZE
    alien.y = mid_y // 2 * TILE_SIZE

    # Initialize Monsters - one type per quadrant, multiple of each
    # North-east quadrant - Slimes
    monsters = []
    for _ in range(3):
        slime = Slime()
        slime.x = random.randint(mid_x + mid_x // 4, environment.width) * TILE_SIZE
        slime.y = random.randint(0, mid_y - mid_y // 4) * TILE_SIZE
        monsters.append(slime)

    # South-east quadrant - Bats
    for _ in range(3):
        bat = Bat()
        bat.x = random.randint(mid_x + mid_x // 4, environment.width) * TILE_SIZE
        bat.y = random.randint(mid_y + mid_y // 4, environment.height) * TILE_SIZE
        monsters.append(bat)

    # South-west quadrant - Ghosts
    for _ in range(3):
        ghost = Ghost()
        ghost.x = random.randint(0, mid_x - mid_x // 4) * TILE_SIZE
        ghost.y = random.randint(mid_y + mid_y // 4, environment.height) * TILE_SIZE
        monsters.append(ghost)

    # North-west quadrant - Spiders
    for _ in range(3):
        spider = Spider()
        spider.x = random.randint(0, mid_x - mid_x // 4) * TILE_SIZE
        spider.y = random.randint(0, mid_y - mid_y // 4) * TILE_SIZE
        monsters.append(spider)

    # Initialize NPCs and Monsters list
    npcs = [npc1, npc2, npc3, alien]

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
            for npc in npcs:
                npc.update()
                npc.patrol_with_stops(environment)

            for monster in monsters:
                monster.update()
                monster.random_movement(environment)

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
            display.draw_environment(environment, player, npcs, monsters)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()