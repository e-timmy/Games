import pygame

class Paddle:
    def __init__(self, x, y, side):
        self.rect = pygame.Rect(x, y, 20, 100)
        self.side = side
        self.speed = 5
        self.movement = 0

    def move(self, up, down):
        self.movement = 0
        if up and self.rect.top > 0:
            self.rect.y -= self.speed
            self.movement = -1
        if down and self.rect.bottom < 600:
            self.rect.y += self.speed
            self.movement = 1

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)

    def set_speed(self, speed):
        self.speed = speed