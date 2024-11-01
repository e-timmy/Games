import pygame

from constants.game_constants import WINDOW_HEIGHT, WHITE, DEBUG_MODE


class UIManager:
    def __init__(self):
        self.font = pygame.font.Font(None, 36)
        self.debug_font = pygame.font.Font(None, 24)

    def draw_level_number(self, render_engine, camera, level_number, x_offset):
        level_text = self.font.render(f"Level {level_number}", True, WHITE)
        text_pos = camera.apply(x_offset + 50, 50)
        if text_pos[0] > -1000:
            render_engine.screen.blit(level_text, text_pos)

    def draw_debug_info(self, render_engine, camera, player, level_manager):
        if not DEBUG_MODE:
            return

        debug_info = [
            f"Camera X: {camera.x:.0f}",
            f"Transition: {level_manager.transitioning}",
            f"Current Level: {level_manager.current_level_number}"
        ]

        for i, text in enumerate(debug_info):
            debug_surface = self.debug_font.render(text, True, WHITE)
            render_engine.screen.blit(
                debug_surface,
                (10, WINDOW_HEIGHT - 120 + (i * 20))
            )
