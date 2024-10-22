import pygame
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE, BLACK, WHITE, CHARACTER_SCALE_FACTOR

class Display:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)

    def draw_environment(self, environment, player, npcs, monsters):
        self.screen.fill(BLACK)

        visible_tiles_x = SCREEN_WIDTH // TILE_SIZE
        visible_tiles_y = SCREEN_HEIGHT // TILE_SIZE

        # Add buffer tiles for smoother scrolling (1 tile extra on each side)
        buffer_tiles = 2  # Adjustable buffer size

        # Calculate player's centered position in tile coordinates
        player_tile_x = player.x // TILE_SIZE
        player_tile_y = player.y // TILE_SIZE

        # Calculate the visible range around the player (with buffer)
        half_width = (visible_tiles_x // 2) + buffer_tiles
        half_height = (visible_tiles_y // 2) + buffer_tiles

        # Calculate the range of tiles to draw
        start_x = int(player_tile_x - half_width)
        end_x = int(player_tile_x + half_width + 1)
        start_y = int(player_tile_y - half_height)
        end_y = int(player_tile_y + half_height + 1)

        # Calculate offset for centering the view
        offset_x = -(player.x % TILE_SIZE)
        offset_y = -(player.y % TILE_SIZE)

        # Calculate base screen position (adjusted for buffer)
        base_screen_x = (SCREEN_WIDTH - visible_tiles_x * TILE_SIZE) // 2 - (buffer_tiles * TILE_SIZE)
        base_screen_y = (SCREEN_HEIGHT - visible_tiles_y * TILE_SIZE) // 2 - (buffer_tiles * TILE_SIZE)

        # Draw ground layer first
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                # Calculate screen position with buffer offset
                screen_x = (x - start_x) * TILE_SIZE + offset_x + base_screen_x
                screen_y = (y - start_y) * TILE_SIZE + offset_y + base_screen_y

                # Only draw if the tile is within the environment bounds
                if 0 <= x < environment.width and 0 <= y < environment.height:
                    ground_type = environment.ground_tiles[y][x]
                    if ground_type:
                        self.screen.blit(environment.tiles['ground'][ground_type], (screen_x, screen_y))

        # Draw object layer
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                screen_x = (x - start_x) * TILE_SIZE + offset_x + base_screen_x
                screen_y = (y - start_y) * TILE_SIZE + offset_y + base_screen_y

                if 0 <= x < environment.width and 0 <= y < environment.height:
                    object_type = environment.object_tiles[y][x]
                    if object_type:
                        self.screen.blit(environment.tiles['object'][object_type], (screen_x, screen_y))

        # Draw NPCs and Monsters
        for npc in npcs:
            npc_sprite = pygame.transform.scale(npc.get_current_sprite(),
                                                (npc.sprite_width * CHARACTER_SCALE_FACTOR,
                                                 npc.sprite_height * CHARACTER_SCALE_FACTOR))
            npc_screen_x = npc.x - player.x + SCREEN_WIDTH // 2 - npc_sprite.get_width() // 2
            npc_screen_y = npc.y - player.y + SCREEN_HEIGHT // 2 - npc_sprite.get_height() // 2
            self.screen.blit(npc_sprite, (npc_screen_x, npc_screen_y))

        for monster in monsters:
            # Get the sprite with appropriate scaling
            monster_sprite = monster.get_current_sprite()  # This now handles all scaling

            # Calculate screen position accounting for the scaled sprite size
            monster_screen_x = monster.x - player.x + SCREEN_WIDTH // 2 - monster_sprite.get_width() // 2
            monster_screen_y = monster.y - player.y + SCREEN_HEIGHT // 2 - monster_sprite.get_height() // 2

            self.screen.blit(monster_sprite, (monster_screen_x, monster_screen_y))

        # Draw player in center of screen
        player_screen_x = SCREEN_WIDTH // 2
        player_screen_y = SCREEN_HEIGHT // 2
        self.screen.blit(player.get_current_sprite(),
                         (player_screen_x - player.sprite_width // 2,
                          player_screen_y - player.sprite_height // 2))

    def draw_menu(self):
        self.screen.fill(BLACK)
        title = self.font.render("The Dispossessed", True, WHITE)
        play_button = self.font.render("Play", True, WHITE)

        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, SCREEN_HEIGHT // 3))
        self.screen.blit(play_button, (SCREEN_WIDTH // 2 - play_button.get_width() // 2, SCREEN_HEIGHT // 2))
