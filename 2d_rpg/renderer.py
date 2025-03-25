import pygame


class Renderer:
    """
    Handles rendering of game elements to the screen.
    """

    def __init__(self, screen, tile_size):
        """
        Initialize the renderer with the game screen and tile size.
        """
        self.screen = screen
        self.tile_size = tile_size

        # Define colors for each tile type
        self.colors = {
            'o': (150, 150, 150),  # Floor - gray
            '#': (80, 80, 80),  # Wall - dark gray
            '~': (64, 164, 223),  # Water - blue
            '.': (107, 142, 35),  # Grass - olive green
            'T': (34, 139, 34),  # Tree - forest green
        }

        # Define player color
        self.player_color = (220, 20, 60)  # Crimson red

    def render_map(self, game_map, start_x, end_x, start_y, end_y, camera_x, camera_y):
        """
        Render the visible portion of the game map to the screen.
        Only renders tiles within the given bounds.
        """
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                # Skip rendering if outside map bounds (shouldn't happen, but just in case)
                if y < 0 or y >= len(game_map) or x < 0 or x >= len(game_map[0]):
                    continue

                # Get the tile character and corresponding color
                tile = game_map[y][x]
                color = self.colors.get(tile, (0, 0, 0))  # Default to black for unknown tiles

                # Calculate screen position
                screen_x = int(x * self.tile_size - camera_x)
                screen_y = int(y * self.tile_size - camera_y)

                # Draw the tile
                self._draw_tile(screen_x, screen_y, color, tile)

    def _draw_tile(self, x, y, color, tile_type):
        """
        Draw a single tile with specific rendering based on tile type.
        """
        rect = pygame.Rect(x, y, self.tile_size, self.tile_size)

        # Basic rendering based on tile type
        if tile_type == 'o':  # Floor
            pygame.draw.rect(self.screen, color, rect)
        elif tile_type == '#':  # Wall
            pygame.draw.rect(self.screen, color, rect)
            # Add some detail to walls
            pygame.draw.rect(self.screen, (50, 50, 50), rect, 2)
        elif tile_type == '~':  # Water
            pygame.draw.rect(self.screen, color, rect)
            # Add some wave detail
            pygame.draw.line(self.screen, (100, 200, 255),
                             (x + 5, y + self.tile_size // 2),
                             (x + self.tile_size - 5, y + self.tile_size // 2),
                             2)
        elif tile_type == '.':  # Grass
            pygame.draw.rect(self.screen, color, rect)
        elif tile_type == 'T':  # Tree
            # Tree trunk
            pygame.draw.rect(self.screen, (101, 67, 33),
                             (x + self.tile_size // 3, y + self.tile_size // 2,
                              self.tile_size // 3, self.tile_size // 2))
            # Tree top (circle)
            pygame.draw.circle(self.screen, color,
                               (x + self.tile_size // 2, y + self.tile_size // 3),
                               self.tile_size // 3)
        else:  # Default rendering for unknown tiles
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

    def render_player(self, player, camera_x, camera_y):
        """
        Render the player character.
        """
        # Calculate screen position
        screen_x = int(player.x * self.tile_size - camera_x)
        screen_y = int(player.y * self.tile_size - camera_y)

        # Draw player
        # Circle for player body
        pygame.draw.circle(self.screen, self.player_color,
                           (screen_x + self.tile_size // 2, screen_y + self.tile_size // 2),
                           self.tile_size // 2 - 2)

        # Draw player health and energy bars
        self._draw_status_bars(screen_x, screen_y, player)

    def _draw_status_bars(self, x, y, player):
        """
        Draw health and energy bars above the player.
        """
        bar_width = self.tile_size
        bar_height = 4
        y_offset = -10  # Position above player

        # Health bar (red)
        health_percentage = player.health / player.max_health
        health_width = int(bar_width * health_percentage)

        pygame.draw.rect(self.screen, (100, 100, 100),
                         (x, y + y_offset, bar_width, bar_height))
        pygame.draw.rect(self.screen, (255, 0, 0),
                         (x, y + y_offset, health_width, bar_height))

        # Energy bar (blue) - below health bar
        energy_percentage = player.energy / player.max_energy
        energy_width = int(bar_width * energy_percentage)

        pygame.draw.rect(self.screen, (100, 100, 100),
                         (x, y + y_offset + bar_height + 1, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 0, 255),
                         (x, y + y_offset + bar_height + 1, energy_width, bar_height))