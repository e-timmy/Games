import pygame


class Menu:
    def __init__(self):
        self.font = pygame.font.Font(None, 74)
        self.small_font = pygame.font.Font(None, 36)

    def draw_main_menu(self, screen):
        screen.fill((0, 0, 0))
        title = self.font.render('PLATFORM CLIMBER', True, (255, 255, 255))
        start = self.small_font.render('Press SPACE to Start', True, (255, 255, 255))

        screen.blit(title, (400 - title.get_width() // 2, 200))
        screen.blit(start, (400 - start.get_width() // 2, 400))

    def draw_overlay(self, screen, text):
        overlay = pygame.Surface((800, 600))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))

        text_surface = self.font.render(text, True, (255, 255, 255))
        continue_text = self.small_font.render('Press SPACE to continue', True, (255, 255, 255))

        screen.blit(text_surface, (400 - text_surface.get_width() // 2, 250))
        screen.blit(continue_text, (400 - continue_text.get_width() // 2, 350))