import pygame


class Player:
    """Represents the player character in the game."""

    def __init__(self, x, y):
        """Initialize the player with starting position."""
        self.x = x
        self.y = y
        self.speed = 1
        self.sprinting = False

    def move(self, dx, dy, map_data):
        """Move the player in the specified direction if possible."""
        new_x = self.x + dx * (self.speed * 2 if self.sprinting else self.speed)
        new_y = self.y + dy * (self.speed * 2 if self.sprinting else self.speed)

        # Check for map boundaries and collisions
        if 0 <= new_x < len(map_data[0]) and 0 <= new_y < len(map_data):
            if map_data[new_y][new_x] != '#':  # Not a wall
                self.x = new_x
                self.y = new_y

    def toggle_sprint(self, sprinting):
        """Toggle sprinting on or off."""
        self.sprinting = sprinting

    def draw(self, surface, camera_x, camera_y, tile_size):
        """Draw the player on the given surface."""
        screen_x = (self.x - camera_x) * tile_size
        screen_y = (self.y - camera_y) * tile_size

        # Draw player as a red rectangle
        pygame.draw.rect(surface, (255, 0, 0), (screen_x, screen_y, tile_size, tile_size))