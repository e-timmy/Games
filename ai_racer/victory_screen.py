import pygame


class VictoryScreen:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.font_large = pygame.font.Font(None, 74)
        self.font_small = pygame.font.Font(None, 36)
        self.overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        self.overlay.fill((0, 0, 0, 128))  # Semi-transparent black

    def draw(self, screen, winner_color, winner_number):
        # Draw semi-transparent overlay
        screen.blit(self.overlay, (0, 0))

        # Draw winner text
        winner_text = self.font_large.render(f"Player {winner_number} Wins!", True, winner_color)
        winner_rect = winner_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))

        # Draw white outline for better visibility
        outline_surface = self.font_large.render(f"Player {winner_number} Wins!", True, (255, 255, 255))
        outline_rect = outline_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))

        # Draw continue text
        continue_text = self.font_small.render("Press ENTER to continue", True, (255, 255, 255))
        continue_rect = continue_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 50))

        # Draw texts
        screen.blit(outline_surface, outline_rect.move(2, 2))  # Shadow effect
        screen.blit(winner_text, winner_rect)
        screen.blit(continue_text, continue_rect)