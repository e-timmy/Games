import pygame
import math


class Vehicle:
    def __init__(self, x, y, speed, vehicle_type='car'):
        self.x = x
        self.y = y
        self.speed = speed
        self.vehicle_type = vehicle_type
        self.animation_frame = 0
        self.animation_speed = 0.2

        # Set dimensions and properties based on type
        if self.vehicle_type == 'car':
            self.width = 60
            self.height = 30
            self.base_color = (255, 0, 0)
            self.wheel_radius = 6
        elif self.vehicle_type == 'truck':
            self.width = 90
            self.height = 40
            self.base_color = (0, 0, 255)
            self.wheel_radius = 8
        elif self.vehicle_type == 'sports_car':
            self.width = 50
            self.height = 25
            self.base_color = (255, 165, 0)
            self.wheel_radius = 5

    def update(self):
        self.x += self.speed
        self.animation_frame += self.animation_speed * abs(self.speed)

    def draw(self, screen):
        # Calculate base position
        direction = 1 if self.speed > 0 else -1
        y_center = self.y - self.height / 2

        # Add very slight bounce animation (reduced from 1 to 0.5)
        bounce_offset = math.sin(self.animation_frame) * 0.5

        if self.vehicle_type == 'car':
            # Main body
            pygame.draw.rect(screen, self.base_color,
                             (self.x, y_center + bounce_offset,
                              self.width, self.height * 0.7))

            # Top part of car
            roof_width = self.width * 0.6
            roof_start = self.x + (self.width - roof_width) * (0.7 if direction > 0 else 0.3)
            pygame.draw.rect(screen, self.base_color,
                             (roof_start, y_center - self.height * 0.2 + bounce_offset,
                              roof_width, self.height * 0.4))

        elif self.vehicle_type == 'truck':
            # Cabin
            cabin_width = self.width * 0.3
            cabin_start = self.x if direction > 0 else self.x + self.width - cabin_width
            pygame.draw.rect(screen, self.base_color,
                             (cabin_start, y_center + bounce_offset,
                              cabin_width, self.height * 0.7))

            # Cargo area
            cargo_start = self.x + cabin_width if direction > 0 else self.x
            cargo_width = self.width - cabin_width
            pygame.draw.rect(screen, self.base_color,
                             (cargo_start, y_center - self.height * 0.1 + bounce_offset,
                              cargo_width, self.height * 0.8))

        elif self.vehicle_type == 'sports_car':
            # Sleek body
            points = [
                (self.x + (0 if direction > 0 else self.width), y_center + bounce_offset),
                (self.x + (self.width if direction > 0 else 0), y_center + bounce_offset),
                (self.x + (self.width * 0.8 if direction > 0 else self.width * 0.2),
                 y_center - self.height * 0.3 + bounce_offset),
                (self.x + (self.width * 0.2 if direction > 0 else self.width * 0.8),
                 y_center - self.height * 0.3 + bounce_offset),
            ]
            pygame.draw.polygon(screen, self.base_color, points)
            pygame.draw.rect(screen, self.base_color,
                             (self.x, y_center + self.height * 0.2 + bounce_offset,
                              self.width, self.height * 0.5))

        # Draw wheels with rotation animation
        wheel_positions = [(self.x + self.width * 0.2, y_center + self.height * 0.6),
                           (self.x + self.width * 0.8, y_center + self.height * 0.6)]

        for wx, wy in wheel_positions:
            pygame.draw.circle(screen, (30, 30, 30), (int(wx), int(wy + bounce_offset)), self.wheel_radius)
            # Add spoke animation
            spoke_angle = math.radians((self.animation_frame * 30) % 360)
            spoke_end_x = wx + math.cos(spoke_angle) * (self.wheel_radius - 2)
            spoke_end_y = wy + bounce_offset + math.sin(spoke_angle) * (self.wheel_radius - 2)
            pygame.draw.line(screen, (100, 100, 100), (wx, wy + bounce_offset),
                             (spoke_end_x, spoke_end_y), 2)

    def collides_with(self, player):
        return (self.x < player.x + player.width and
                self.x + self.width > player.x and
                self.y - self.height / 2 < player.y + player.height / 2 and
                self.y + self.height / 2 > player.y - player.height / 2)
