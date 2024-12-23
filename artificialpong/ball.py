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
        self.magnetic_cooldown = 0

    def move(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

    def check_collision(self, left_paddle, right_paddle):
        if self.rect.top <= 0 or self.rect.bottom >= 600:
            self.dy *= -1

        collision = None
        if self.rect.colliderect(left_paddle.rect):
            self.handle_paddle_collision(left_paddle)
            collision = 'left'
        elif self.rect.colliderect(right_paddle.rect):
            self.handle_paddle_collision(right_paddle)
            collision = 'right'

        if self.rect.left <= 0 or self.rect.right >= 800:
            self.reset()

        return collision

    def handle_paddle_collision(self, paddle):
        relative_intersect_y = (paddle.rect.y + paddle.rect.height / 2) - (self.rect.y + self.rect.height / 2)
        normalized_relative_intersect_y = relative_intersect_y / (paddle.rect.height / 2)
        bounce_angle = normalized_relative_intersect_y * (5 * math.pi / 12)  # Max angle: 75 degrees

        if paddle.side == 'left':
            self.angle = -bounce_angle
        else:
            self.angle = math.pi - bounce_angle

        # Adjust angle based on paddle movement
        if paddle.movement != 0:
            self.angle += math.pi / 12 * paddle.movement  # Add or subtract up to 15 degrees

        self.speed_up()
        self.dx = self.speed * math.cos(self.angle)
        self.dy = -self.speed * math.sin(self.angle)  # Negative because pygame y-axis is inverted

        # Set magnetic cooldown
        self.magnetic_cooldown = 30  # About 0.5 seconds at 60 FPS

    def speed_up(self):
        self.speed += self.speed_increase

    def reset(self):
        self.rect.center = (400, 300)
        self.speed = 5
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        if random.choice([True, False]):
            angle += math.pi  # Occasionally start towards the left
        self.dx = self.speed * math.cos(angle)
        self.dy = self.speed * math.sin(angle)

        # Ensure significant horizontal movement
        min_horizontal_speed = 2
        while abs(self.dx) < min_horizontal_speed:
            if self.dx > 0:
                self.dx = min_horizontal_speed
            else:
                self.dx = -min_horizontal_speed

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)

    def update(self, left_paddle, right_paddle):
        self.move()

        if self.magnetic_cooldown > 0:
            self.magnetic_cooldown -= 1
            return

        # Apply magnetic effect
        for paddle in [left_paddle, right_paddle]:
            if paddle.is_magnetic():
                dx = paddle.rect.centerx - self.rect.centerx
                dy = paddle.rect.centery - self.rect.centery
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # Only apply magnetic effect if the ball is moving towards the opposite side
                is_moving_away = (paddle.side == 'left' and self.dx > 0) or (paddle.side == 'right' and self.dx < 0)

                if distance < 100 and is_moving_away:  # Magnetic effect range
                    force = 0.5 * (100 - distance) / 100  # Stronger effect when closer
                    self.dx += force * dx / distance
                    self.dy += force * dy / distance

        # Normalize speed to prevent the ball from getting too fast due to magnetic effect
        current_speed = math.sqrt(self.dx ** 2 + self.dy ** 2)
        if current_speed > self.speed:
            self.dx = self.dx / current_speed * self.speed
            self.dy = self.dy / current_speed * self.speed