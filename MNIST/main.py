import pygame
import math
import random

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
GRASS = (34, 139, 34)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Racing Game")
clock = pygame.time.Clock()

# Track definition - walls as line segments [(x1, y1, x2, y2), ...]
track_outer = [
    (100, 100, 700, 100),  # Top
    (700, 100, 700, 500),  # Right
    (700, 500, 100, 500),  # Bottom
    (100, 500, 100, 100),  # Left
]

track_inner = [
    (200, 200, 600, 200),  # Top
    (600, 200, 600, 400),  # Right
    (600, 400, 200, 400),  # Bottom
    (200, 400, 200, 200),  # Left
]

# Waypoints for AI
waypoints = [
    (150, 150),  # Top-left corner
    (650, 150),  # Top-right corner
    (650, 450),  # Bottom-right corner
    (150, 450),  # Bottom-left corner
]


class Car:
    def __init__(self, x, y, color, is_ai=False):
        self.x = x
        self.y = y
        self.width = 20
        self.height = 10
        self.color = color
        self.angle = 0  # radians
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.1
        self.deceleration = 0.05
        self.steering_speed = 0.1
        self.friction = 0.02
        self.is_ai = is_ai
        self.current_waypoint = 0
        self.waypoint_threshold = 30  # How close the AI needs to get to a waypoint
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.crashed = False

        # For difficulty scaling
        if is_ai:
            self.ai_difficulty = "medium"  # easy, medium, hard
            self.set_difficulty(self.ai_difficulty)

    def set_difficulty(self, difficulty):
        if difficulty == "easy":
            self.max_speed = 3
            self.acceleration = 0.05
            self.steering_speed = 0.05
            self.waypoint_threshold = 50
        elif difficulty == "medium":
            self.max_speed = 4
            self.acceleration = 0.08
            self.steering_speed = 0.08
            self.waypoint_threshold = 30
        elif difficulty == "hard":
            self.max_speed = 5
            self.acceleration = 0.1
            self.steering_speed = 0.1
            self.waypoint_threshold = 20

    def accelerate(self):
        self.speed += self.acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed

    def decelerate(self):
        self.speed -= self.deceleration
        if self.speed < -self.max_speed / 2:
            self.speed = -self.max_speed / 2

    def steer_left(self):
        self.angle -= self.steering_speed

    def steer_right(self):
        self.angle += self.steering_speed

    def apply_friction(self):
        if self.speed > 0:
            self.speed -= self.friction
            if self.speed < 0:
                self.speed = 0
        elif self.speed < 0:
            self.speed += self.friction
            if self.speed > 0:
                self.speed = 0

    def move(self):
        if self.crashed:
            self.speed = 0
            return

        self.apply_friction()

        # Move based on angle and speed
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Update rectangle for collision detection
        self.rect.center = (self.x, self.y)

    def ai_control(self):
        if self.crashed:
            return

        target_x, target_y = waypoints[self.current_waypoint]

        # Calculate angle to waypoint
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)

        # Normalize angles for comparison
        while target_angle < 0:
            target_angle += 2 * math.pi
        while self.angle < 0:
            self.angle += 2 * math.pi
        while self.angle >= 2 * math.pi:
            self.angle -= 2 * math.pi

        # Calculate difference between current and target angles
        diff = target_angle - self.angle
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi

        # Steering based on angle difference
        if diff > 0.05:
            self.steer_right()
        elif diff < -0.05:
            self.steer_left()

        # Calculate distance to waypoint
        distance = math.sqrt(dx * dx + dy * dy)

        # Speed control
        if distance > 100:
            self.accelerate()
        elif distance > 50:
            if self.speed < self.max_speed * 0.8:
                self.accelerate()
        else:
            if self.speed > self.max_speed * 0.6:
                self.decelerate()
            else:
                self.accelerate()

        # Change waypoint if close enough
        if distance < self.waypoint_threshold:
            self.current_waypoint = (self.current_waypoint + 1) % len(waypoints)

    def draw(self, surface):
        # Create a rotated rectangle
        car_rect = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(car_rect, self.color, (0, 0, self.width, self.height))

        # Add a direction indicator
        pygame.draw.line(car_rect, BLACK, (self.width // 2, self.height // 2),
                         (self.width, self.height // 2), 2)

        # Rotate and blit the car
        rotated_car = pygame.transform.rotate(car_rect, -self.angle * 180 / math.pi)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        surface.blit(rotated_car, rect.topleft)

        # Draw waypoint marker for AI
        if self.is_ai:
            target_x, target_y = waypoints[self.current_waypoint]
            pygame.draw.circle(surface, YELLOW, (target_x, target_y), 5, 1)


def check_collision(car, walls):
    # Simplified collision detection (can be improved)
    car_corners = [
        (car.x + math.cos(car.angle) * car.width / 2 - math.sin(car.angle) * car.height / 2,
         car.y + math.sin(car.angle) * car.width / 2 + math.cos(car.angle) * car.height / 2),
        (car.x + math.cos(car.angle) * car.width / 2 + math.sin(car.angle) * car.height / 2,
         car.y + math.sin(car.angle) * car.width / 2 - math.cos(car.angle) * car.height / 2),
        (car.x - math.cos(car.angle) * car.width / 2 + math.sin(car.angle) * car.height / 2,
         car.y - math.sin(car.angle) * car.width / 2 - math.cos(car.angle) * car.height / 2),
        (car.x - math.cos(car.angle) * car.width / 2 - math.sin(car.angle) * car.height / 2,
         car.y - math.sin(car.angle) * car.width / 2 + math.cos(car.angle) * car.height / 2)
    ]

    # Check if any corner is outside track bounds
    for corner_x, corner_y in car_corners:
        # Check if outside outer track
        if corner_x < 100 or corner_x > 700 or corner_y < 100 or corner_y > 500:
            return True
        # Check if inside inner track
        if 200 < corner_x < 600 and 200 < corner_y < 400:
            return True

    return False


# Create player and AI cars
player = Car(150, 150, RED)
ai_car = Car(150, 180, BLUE, is_ai=True)

running = True
while running:
    # Keep game running at right speed
    clock.tick(FPS)

    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                ai_car.set_difficulty("easy")
            elif event.key == pygame.K_2:
                ai_car.set_difficulty("medium")
            elif event.key == pygame.K_3:
                ai_car.set_difficulty("hard")

    # Handle player input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player.accelerate()
    if keys[pygame.K_DOWN]:
        player.decelerate()
    if keys[pygame.K_LEFT]:
        player.steer_left()
    if keys[pygame.K_RIGHT]:
        player.steer_right()

    # AI control
    ai_car.ai_control()

    # Update positions
    player.move()
    ai_car.move()

    # Check collisions
    player.crashed = check_collision(player, track_outer + track_inner)
    ai_car.crashed = check_collision(ai_car, track_outer + track_inner)

    # Drawing
    screen.fill(GRASS)

    # Draw track
    pygame.draw.rect(screen, GRAY, (100, 100, 600, 400))
    pygame.draw.rect(screen, GRASS, (200, 200, 400, 200))

    # Draw walls
    for x1, y1, x2, y2 in track_outer + track_inner:
        pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 2)

    # Draw waypoints (for visualization)
    for i, (wx, wy) in enumerate(waypoints):
        pygame.draw.circle(screen, YELLOW if i == ai_car.current_waypoint else WHITE, (wx, wy), 3)

    # Draw cars
    player.draw(screen)
    ai_car.draw(screen)

    # Draw UI
    font = pygame.font.SysFont('Arial', 16)
    speed_text = font.render(f"Speed: {abs(player.speed):.1f}", True, WHITE)
    ai_text = font.render(f"AI Difficulty: {ai_car.ai_difficulty} (Press 1-3 to change)", True, WHITE)
    crashed_text = font.render("CRASHED!" if player.crashed else "", True, RED)

    screen.blit(speed_text, (10, 10))
    screen.blit(ai_text, (10, 30))
    screen.blit(crashed_text, (WIDTH // 2 - 50, 10))

    # Flip display
    pygame.display.flip()

pygame.quit()