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
        self.item_width = 80

        # UI elements
        self.speed_input = InputBox(self.screen_width - 120, 5, 60, 25, "1")
        self.menu_button = Button(10, 5, 60, 25, "Menu", pygame.Color('red'))

        # Initialize data
        self.car_data = []

    def handle_event(self, event):
        self.speed_input.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.menu_button.rect.collidepoint(event.pos):
                return True
        return False

    def get_speed_factor(self):
        try:
            return max(1, min(1000, int(self.speed_input.text)))
        except ValueError:
            return 1

    def update(self, car_data):
        self.car_data = car_data

    def draw(self, screen):
        # Draw top panel
        top_panel_rect = pygame.Rect(0, 0, self.screen_width, self.panel_height)
        pygame.draw.rect(screen, (0, 0, 0), top_panel_rect)

        # Draw menu button
        self.menu_button.draw(screen)

        # Draw speed input
        speed_label = self.font.render("Speed:", True, (255, 255, 255))
        screen.blit(speed_label, (self.screen_width - 180, 10))
        self.speed_input.draw(screen)

        # Draw bottom panel with car data
        bottom_panel_rect = pygame.Rect(0, self.screen_height - self.panel_height,
                                        self.screen_width, self.panel_height)
        pygame.draw.rect(screen, (0, 0, 0), bottom_panel_rect)

        if self.car_data:
            total_width = len(self.car_data) * self.item_width
            start_x = (self.screen_width - total_width) // 2

            for i, (color, number, laps) in enumerate(self.car_data):
                x = start_x + (i * self.item_width)
                y = self.screen_height - self.panel_height + 8

                square_rect = pygame.Rect(x, y, self.square_size, self.square_size)
                pygame.draw.rect(screen, color, square_rect)
                pygame.draw.rect(screen, (255, 255, 255), square_rect, 1)

                text_surface = self.font.render(str(laps), True, (255, 255, 255))
                screen.blit(text_surface, (x + self.square_size + 5, y + 2))


class Button:
    def __init__(self, x, y, w, h, text, color):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.text = text
        self.font = pygame.font.Font(None, 24)
        self.txt_surface = self.font.render(text, True, (255, 255, 255))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_rect = self.txt_surface.get_rect(center=self.rect.center)
        screen.blit(self.txt_surface, text_rect)


class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = pygame.Color('lightskyblue3')
        self.text = text
        self.font = pygame.font.Font(None, 24)
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