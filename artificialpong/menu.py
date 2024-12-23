import pygame

class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)
        self.difficulties = ["Easy", "Medium", "Hard"]
        self.selected = 0

    def draw(self):
        self.screen.fill((0, 0, 0))
        for i, diff in enumerate(self.difficulties):
            color = (255, 255, 255) if i == self.selected else (100, 100, 100)
            text = self.font.render(diff, True, color)
            self.screen.blit(text, (350, 250 + i * 50))
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected = (self.selected - 1) % len(self.difficulties)
                    elif event.key == pygame.K_DOWN:
                        self.selected = (self.selected + 1) % len(self.difficulties)
                    elif event.key == pygame.K_RETURN:
                        return self.difficulties[self.selected]
            self.draw()
        return None