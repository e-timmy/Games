import pygame


class Vehicle:
    def __init__(self, x, y, speed, vehicle_type='car'):
        self.x = x
        self.y = y
        self.speed = speed
        self.vehicle_type = vehicle_type

        # Set dimensions based on type
        if vehicle_type == 'car':
            self.width = 60
            self.height = 30
            self.color = (255, 0, 0)
        elif vehicle_type == 'truck':
            self.width = 90
            self.height = 40
            self.color = (0, 0, 255)
        elif vehicle_type == 'sports_car':
            self.width = 50
            self.height = 25
            self.color = (255, 165, 0)

    def update(self):
        self.x += self.speed

    def draw(self, screen):
        pygame.draw.rect(screen, self.color,
                         (self.x, self.y - self.height / 2,
                          self.width, self.height))

    def collides_with(self, player):
        return (self.x < player.x + player.width and
                self.x + self.width > player.x and
                self.y - self.height / 2 < player.y + player.height / 2 and
                self.y + self.height / 2 > player.y - player.height / 2)