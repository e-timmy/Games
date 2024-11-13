import pygame
from game_state import GameState


class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 74)
        self.font_small = pygame.font.Font(None, 36)

    def update(self):
        self.screen.fill((100, 200, 100))

        # Title
        text = self.font.render("Chicken Crossing", True, (255, 255, 255))
        text_rect = text.get_rect(center=(400, 200))
        self.screen.blit(text, text_rect)

        # Instruction
        instruction = self.font_small.render("Press SPACE to Start", True, (255, 255, 255))
        instruction_rect = instruction.get_rect(center=(400, 300))
        self.screen.blit(instruction, instruction_rect)

        # Controls
        controls = [
            "Controls:",
            "Arrow Keys Left/Right - Move",
            "Arrow Keys Up/Down - Change Lanes"
        ]

        for i, control in enumerate(controls):
            control_text = self.font_small.render(control, True, (255, 255, 255))
            control_rect = control_text.get_rect(center=(400, 400 + i * 30))
            self.screen.blit(control_text, control_rect)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            return GameState.PLAYING
        return GameState.MENU