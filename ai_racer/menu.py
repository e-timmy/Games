import pygame
import pygame.freetype

class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = pygame.Color('lightskyblue3')
        self.text = text
        self.font = pygame.font.Font(None, 28)
        self.txt_surface = self.font.render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = pygame.Color('dodgerblue2') if self.active else pygame.Color('lightskyblue3')
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.active = False
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, self.color)

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        pygame.draw.rect(screen, self.color, self.rect, 2)

class PlayerConfigUI:
    def __init__(self, x, y, width, height, player_num):
        self.rect = pygame.Rect(x, y, width, height)
        self.player_num = player_num
        self.player_type = "AI"
        self.difficulty = InputBox(x + 70, y + 40, 50, 28, "1.0")
        self.type_button = Button(x + width - 60, y + 5, 50, 28, "AI", pygame.Color('blue'))
        self.can_be_human = player_num <= 2
        self.control_scheme = "arrows" if player_num == 1 else "wasd"

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.type_button.rect.collidepoint(event.pos) and self.can_be_human:
                if self.player_type == "Human":
                    self.player_type = "AI"
                    self.type_button.text = "AI"
                elif self.player_type == "AI":
                    self.player_type = "RL"
                    self.type_button.text = "RL"
                else:
                    self.player_type = "Human"
                    self.type_button.text = "HUM"
                self.type_button.txt_surface = self.type_button.font.render(
                    self.type_button.text, True, pygame.Color('white'))

    def draw(self, screen):
        pygame.draw.rect(screen, pygame.Color('darkgray'), self.rect)
        title = pygame.font.Font(None, 28).render(f"P{self.player_num}", True, pygame.Color('white'))
        screen.blit(title, (self.rect.x + 10, self.rect.y + 10))

        if self.can_be_human:
            self.type_button.draw(screen)

        if self.player_type == "AI":
            diff_label = pygame.font.Font(None, 24).render("Diff:", True, pygame.Color('white'))
            screen.blit(diff_label, (self.rect.x + 10, self.rect.y + 45))
            self.difficulty.draw(screen)
        else:
            control_text = "Controls: " + ("Arrows" if self.control_scheme == "arrows" else "WASD")
            ctrl_label = pygame.font.Font(None, 24).render(control_text, True, pygame.Color('white'))
            screen.blit(ctrl_label, (self.rect.x + 10, self.rect.y + 45))

class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = pygame.font.Font(None, 28)
        self.txt_surface = self.font.render(text, True, pygame.Color('white'))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_rect = self.txt_surface.get_rect(center=self.rect.center)
        screen.blit(self.txt_surface, text_rect)


class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()
        self.config_width = 200
        self.config_height = 80
        self.margin = 10
        self.columns = 2
        self.rows_visible = 5

        self.title_font = pygame.font.Font(None, 64)
        self.subtitle_font = pygame.font.Font(None, 32)

        self.configs = []
        self.add_initial_config()

        self.lap_input = InputBox(20, self.screen_height - 120, 80, 40, "0")
        self.lap_label = self.subtitle_font.render("Number of Laps (0 = infinite):", True, (255, 255, 255))

        button_y = self.screen_height - 60
        self.add_button = Button(20, button_y, 80, 40, "+", pygame.Color('green'))
        self.start_button = Button(self.screen_width - 100, button_y, 80, 40, "Start", pygame.Color('blue'))

    def get_config_position(self, index):
        # Calculate the total width needed for the grid
        total_width = self.columns * (self.config_width + self.margin)
        # Center the entire grid horizontally
        start_x = (self.screen_width - total_width) // 2

        # Calculate column and row
        col = index % self.columns
        row = index // self.columns

        # Calculate x and y positions
        x = start_x + col * (self.config_width + self.margin)
        y = self.margin + row * (self.config_height + self.margin) + 100  # 100 is offset for title

        return x, y

    def add_initial_config(self):
        # Use the standard positioning for the first config
        x, y = self.get_config_position(0)
        self.add_config_at_position(x, y)

    def add_config_at_position(self, x, y):
        new_config = PlayerConfigUI(x, y, self.config_width, self.config_height, len(self.configs) + 1)
        self.configs.append(new_config)

    def run(self):
        running = True
        while running:
            self.screen.fill((40, 40, 40))

            # Draw title
            title = self.title_font.render("Racing Game", True, (255, 255, 255))
            title_rect = title.get_rect(centerx=self.screen_width / 2, y=20)
            self.screen.blit(title, title_rect)

            # Draw subtitle
            subtitle = self.subtitle_font.render("Configure Players", True, (200, 200, 200))
            subtitle_rect = subtitle.get_rect(centerx=self.screen_width / 2, y=70)
            self.screen.blit(subtitle, subtitle_rect)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return [], 0

                for config in self.configs:
                    config.handle_event(event)

                self.lap_input.handle_event(event)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.add_button.rect.collidepoint(event.pos):
                        if len(self.configs) < 10:
                            x, y = self.get_config_position(len(self.configs))
                            if y + self.config_height < self.screen_height - 140:
                                self.add_config_at_position(x, y)

                    if self.start_button.rect.collidepoint(event.pos):
                        try:
                            lap_count = int(self.lap_input.text)
                            return self._create_player_configs(), lap_count
                        except ValueError:
                            return self._create_player_configs(), 0

            # Draw all configurations
            for config in self.configs:
                config.draw(self.screen)

            self.screen.blit(self.lap_label, (20, self.screen_height - 150))
            self.lap_input.draw(self.screen)

            if len(self.configs) < 10:
                self.add_button.draw(self.screen)
            self.start_button.draw(self.screen)

            pygame.display.flip()

        return []

    def _create_player_configs(self):
        player_configs = []
        for config_ui in self.configs:
            print(f"Creating config with player type: {config_ui.player_type}")
            if config_ui.player_type == "Human":
                player_configs.append(PlayerConfig(
                    "Human",
                    config_ui.control_scheme,
                    None
                ))
            elif config_ui.player_type == "RL":
                player_configs.append(PlayerConfig(
                    "RL",
                    None,
                    None
                ))
            else:  # AI
                player_configs.append(PlayerConfig(
                    "AI",
                    None,
                    config_ui.difficulty
                ))
        return player_configs

class PlayerConfig:
    def __init__(self, player_type="Human", control_scheme=None, difficulty=1.0):
        self.player_type = player_type
        self.control_scheme = control_scheme
        self.difficulty = difficulty


def get_key_mapping(key_config):
    key_mappings = {
        "arrows": (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT),
        "wasd": (pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d),
        "ijkl": (pygame.K_i, pygame.K_k, pygame.K_j, pygame.K_l),
        "8456": (pygame.K_KP8, pygame.K_KP5, pygame.K_KP4, pygame.K_KP6)
    }
    return key_mappings.get(key_config, key_mappings["arrows"])