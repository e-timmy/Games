import pygame
import random
import math

class Ball:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.speed = 5
        self.angle = random.uniform(-math.pi / 4, math.pi / 4)
        self.dx = self.speed * math.cos(self.angle)
        self.dy = self.speed * math.sin(self.angle)
        self.speed_increase = 0.1

    def move(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

        # Only bounce off top and bottom
        if self.rect.top <= 0 or self.rect.bottom >= 600:
            self.dy *= -1

    def check_collision(self, left_paddle, right_paddle):
        collision = None
        if self.rect.colliderect(left_paddle.rect):
            self.handle_paddle_collision(left_paddle)
            collision = 'left'
        elif self.rect.colliderect(right_paddle.rect):
            self.handle_paddle_collision(right_paddle)
            collision = 'right'

        return collision

    def handle_paddle_collision(self, paddle):
        relative_intersect_y = (paddle.rect.y + paddle.rect.height / 2) - (self.rect.y + self.rect.height / 2)
        normalized_relative_intersect_y = relative_intersect_y / (paddle.rect.height / 2)
        bounce_angle = normalized_relative_intersect_y * (5 * math.pi / 12)

        if paddle.side == 'left':
            self.angle = -bounce_angle
        else:
            self.angle = math.pi - bounce_angle

        if paddle.movement != 0:
            self.angle += math.pi / 12 * paddle.movement

        self.speed += self.speed_increase
        self.dx = self.speed * math.cos(self.angle)
        self.dy = -self.speed * math.sin(self.angle)

    def reset(self):
        self.rect.center = (400, 300)
        self.speed = 5
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        if random.choice([True, False]):
            angle += math.pi
        self.dx = self.speed * math.cos(angle)
        self.dy = self.speed * math.sin(angle)

        min_horizontal_speed = 2
        while abs(self.dx) < min_horizontal_speed:
            if self.dx > 0:
                self.dx = min_horizontal_speed
            else:
                self.dx = -min_horizontal_speed

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)