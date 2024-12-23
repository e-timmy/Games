import pygame
import random


class PowerUp:
    SHRINK_OPPONENT = 1
    SPEED_BOOST = 2
    MAGNETIC_PADDLE = 3

    COLORS = {
        SHRINK_OPPONENT: (255, 0, 0),  # Red
        SPEED_BOOST: (0, 255, 0),  # Green
        MAGNETIC_PADDLE: (0, 0, 255)  # Blue
    }

    def __init__(self):
        self.type = random.choice([PowerUp.SHRINK_OPPONENT, PowerUp.SPEED_BOOST, PowerUp.MAGNETIC_PADDLE])
        self.width = 20
        self.height = 20
        self.speed = random.uniform(1, 3)

        if random.choice([True, False]):
            self.rect = pygame.Rect(random.randint(100, 700), -self.height, self.width, self.height)
            self.direction = 1  # Moving down
        else:
            self.rect = pygame.Rect(random.randint(100, 700), 600, self.width, self.height)
            self.direction = -1  # Moving up

    def update(self):
        self.rect.y += self.speed * self.direction

    def draw(self, screen):
        pygame.draw.rect(screen, PowerUp.COLORS[self.type], self.rect)

    def collides_with(self, ball):
        return self.rect.colliderect(ball.rect)

    def is_off_screen(self):
        return self.rect.top > 600 or self.rect.bottom < 0