class Camera:
    """Controls which portion of the map is visible on screen."""

    def __init__(self, map_width, map_height, screen_width, screen_height, tile_size):
        """Initialize the camera with map and screen dimensions."""
        self.x = 0
        self.y = 0
        self.map_width = map_width
        self.map_height = map_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tile_size = tile_size

        # Calculate how many tiles can fit on screen
        self.view_width = screen_width // tile_size
        self.view_height = screen_height // tile_size

    def center_on(self, x, y):
        """Center the camera on the given coordinates."""
        self.x = x - self.view_width // 2
        self.y = y - self.view_height // 2

        # Keep camera within map bounds
        self.x = max(0, min(self.map_width - self.view_width, self.x))
        self.y = max(0, min(self.map_height - self.view_height, self.y))

    def get_visible_area(self):
        """Get the map area currently visible on screen.

        Returns:
            A tuple of (start_x, start_y, end_x, end_y).
        """
        # Add a small buffer for smooth scrolling
        buffer = 2
        start_x = max(0, self.x - buffer)
        start_y = max(0, self.y - buffer)
        end_x = min(self.map_width, self.x + self.view_width + buffer)
        end_y = min(self.map_height, self.y + self.view_height + buffer)

        return (start_x, start_y, end_x, end_y)