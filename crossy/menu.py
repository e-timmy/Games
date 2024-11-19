import pygame
from game_state import GameState


class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 74)
        self.font_small = pygame.font.Font(None, 36)
        self.level_select_active = False
        self.selected_level = 1
        self.max_level = 20  # You can adjust this based on your game's total levels
        self.prev_up = False
        self.prev_down = False

    def update(self):
        self.screen.fill((100, 200, 100))

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            self.level_select_active = True
        else:
            self.level_select_active = False

        if self.level_select_active:
            return self.handle_level_select()
        else:
            return self.draw_main_menu()

    def draw_main_menu(self):
        # Title
        text = self.font.render("Chicken Crossing", True, (255, 255, 255))
        text_rect = text.get_rect(center=(400, 200))
        self.screen.blit(text, text_rect)

        # Instruction
        instruction = self.font_small.render("Press SPACE to Start", True, (255, 255, 255))
        instruction_rect = instruction.get_rect(center=(400, 300))
        self.screen.blit(instruction, instruction_rect)

        # Level select instruction
        level_select_text = self.font_small.render("Hold SHIFT for level select", True, (255, 255, 255))
        level_select_rect = level_select_text.get_rect(center=(400, 350))
        self.screen.blit(level_select_text, level_select_rect)

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

    def handle_level_select(self):
        # Draw semi-transparent overlay
        overlay = pygame.Surface((800, 600))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(200)
        self.screen.blit(overlay, (0, 0))

        # Title
        title = self.font.render("Level Select", True, (255, 255, 255))
        title_rect = title.get_rect(center=(400, 100))
        self.screen.blit(title, title_rect)

        # Selected level
        level_text = self.font.render(f"Level {self.selected_level}", True, (255, 255, 255))
        level_rect = level_text.get_rect(center=(400, 250))
        self.screen.blit(level_text, level_rect)

        # Instructions
        instructions = [
            "Use UP/DOWN arrows to change level",
            "Press ENTER to start at selected level",
            "Release SHIFT to cancel"
        ]

        for i, instruction in enumerate(instructions):
            inst_text = self.font_small.render(instruction, True, (255, 255, 255))
            inst_rect = inst_text.get_rect(center=(400, 350 + i * 40))
            self.screen.blit(inst_text, inst_rect)

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP] and not self.prev_up:
            self.selected_level = min(self.selected_level + 1, self.max_level)
        if keys[pygame.K_DOWN] and not self.prev_down:
            self.selected_level = max(self.selected_level - 1, 1)

        self.prev_up = keys[pygame.K_UP]
        self.prev_down = keys[pygame.K_DOWN]

        if keys[pygame.K_RETURN]:
            return GameState.PLAYING

        return GameState.MENU

    def get_selected_level(self):
        return self.selected_level