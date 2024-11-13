import pygame
import math


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.target_y = y
        self.width = 30
        self.height = 30
        self.speed = 3
        self.lane_index = -1
        self.color = (255, 200, 0)
        self.is_moving = False
        self.move_speed = 5  # Pixels per frame for vertical movement
        self.celebrating = False
        self.celebration_time = 0
        self.celebration_height = 0

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.target_y = y
        self.lane_index = -1
        self.is_moving = False
        self.celebrating = False
        self.celebration_time = 0
        self.celebration_height = 0

    def start_celebration(self):
        self.celebrating = True
        self.celebration_time = 0

    def update(self, lanes):
        if self.celebrating:
            # Jumping celebration animation
            self.celebration_time += 1
            # Use sine wave for smooth up and down motion
            self.celebration_height = math.sin(self.celebration_time * 0.1) * 30
            return

        keys = pygame.key.get_pressed()

        # Horizontal movement
        if keys[pygame.K_LEFT]:
            self.x = max(0, self.x - self.speed)
        if keys[pygame.K_RIGHT]:
            self.x = min(800 - self.width, self.x + self.speed)

        # Vertical movement with smooth transition
        if not self.is_moving:
            if keys[pygame.K_UP]:
                if self.lane_index == -1:  # Starting position
                    self.lane_index = len(lanes) - 1
                    self.target_y = lanes[self.lane_index]
                    self.is_moving = True
                elif self.lane_index > 0:
                    self.lane_index -= 1
                    self.target_y = lanes[self.lane_index]
                    self.is_moving = True
                elif self.lane_index == 0:  # Move to finish
                    self.lane_index = -2
                    self.target_y = lanes[0] - 100
                    self.is_moving = True

            elif keys[pygame.K_DOWN]:
                if self.lane_index == -2:  # If at finish, move back to top lane
                    self.lane_index = 0
                    self.target_y = lanes[0]
                    self.is_moving = True
                elif self.lane_index < len(lanes) - 1:
                    self.lane_index += 1
                    self.target_y = lanes[self.lane_index]
                    self.is_moving = True

        # Smooth movement towards target
        if self.is_moving:
            diff = self.target_y - self.y
            if abs(diff) < self.move_speed:
                self.y = self.target_y
                self.is_moving = False
            else:
                move_dir = 1 if diff > 0 else -1
                self.y += self.move_speed * move_dir

    def draw(self, screen):
        # Calculate actual y position including celebration jump height
        display_y = self.y - self.celebration_height
        y_center = display_y - self.height / 2

        # Body
        pygame.draw.ellipse(screen, self.color,
                            (self.x, y_center, self.width, self.height))

        # Head
        head_size = self.width * 0.4
        pygame.draw.circle(screen, self.color,
                           (self.x + self.width * 0.8, y_center - head_size / 2),
                           int(head_size / 2))

        # Beak
        beak_points = [
            (self.x + self.width * 0.9, y_center - head_size / 2),
            (self.x + self.width * 1.1, y_center - head_size / 2),
            (self.x + self.width * 0.9, y_center - head_size / 4)
        ]
        pygame.draw.polygon(screen, (255, 100, 0), beak_points)

        # Eye
        pygame.draw.circle(screen, (0, 0, 0),
                           (int(self.x + self.width * 0.8),
                            int(y_center - head_size / 2)),
                           2)

        # Feet (bouncing when celebrating)
        foot_color = (255, 100, 0)
        foot_bounce = abs(math.sin(self.celebration_time * 0.2) * 5) if self.celebrating else 0
        pygame.draw.line(screen, foot_color,
                         (self.x + self.width * 0.3, y_center + self.height),
                         (self.x + self.width * 0.3, y_center + self.height + 5 + foot_bounce),
                         3)
        pygame.draw.line(screen, foot_color,
                         (self.x + self.width * 0.7, y_center + self.height),
                         (self.x + self.width * 0.7, y_center + self.height + 5 + foot_bounce),
                         3)