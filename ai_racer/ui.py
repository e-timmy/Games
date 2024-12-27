import pygame


class UI:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 24)
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()
        self.margin = 20
        self.square_size = 20
        self.panel_height = 35
        self.item_width = 80  # Fixed width for each player's info

    def draw(self, screen):
        # Draw black background panel
        panel_rect = pygame.Rect(0, self.screen_height - self.panel_height,
                                 self.screen_width, self.panel_height)
        pygame.draw.rect(screen, (0, 0, 0), panel_rect)

        items_count = len(self.car_data)

        # Calculate total width needed for all items
        total_width = items_count * self.item_width

        # Calculate starting x position to center all items
        start_x = (self.screen_width - total_width) // 2

        for i, (color, number, laps) in enumerate(self.car_data):
            # Calculate x position for this item
            x = start_x + (i * self.item_width)
            y = self.screen_height - self.panel_height + 8

            # Draw colored square
            square_rect = pygame.Rect(x, y, self.square_size, self.square_size)
            pygame.draw.rect(screen, color, square_rect)
            pygame.draw.rect(screen, (255, 255, 255), square_rect, 1)

            # Draw lap counter
            text_surface = self.font.render(str(laps), True, (255, 255, 255))
            screen.blit(text_surface, (x + self.square_size + 5, y + 2))

    def update(self, car_data):
        self.car_data = car_data