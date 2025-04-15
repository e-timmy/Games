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

# All walls combined
all_walls = track_outer + track_inner

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
        self.bounce_cooldown = 0  # Prevent multiple bounces in a row
        self.bounce_factor = 0.5  # How much speed is retained after bouncing
        self.mass = 1.0  # For collision physics
        self.collision_radius = max(self.width, self.height) / 1.5  # For car-car collisions

        # Add these properties to the Car class __init__
        self.is_drifting = False
        self.drift_factor = 0.0  # Range 0-1, how much the car is drifting
        self.drift_control = 0.8  # How much steering control during drift (lower = more slide)
        self.drift_momentum = 0  # Additional momentum in drift direction
        self.drift_max_momentum = 3  # Max drift momentum
        self.drift_decay = 0.05  # How quickly drift ends naturally
        self.drift_boost = 0  # Speed boost after successful drift

        # For difficulty scaling
        if is_ai:
            self.ai_difficulty = "medium"  # easy, medium, hard
            self.set_difficulty(self.ai_difficulty)

    def handle_drift(self):
        keys = pygame.key.get_pressed()

        # Start drift with spacebar (for player car only)
        if not self.is_ai:
            if keys[pygame.K_SPACE] and self.speed > 2:  # Minimum speed required for drifting
                self.is_drifting = True

                # Increase drift factor gradually
                self.drift_factor = min(1.0, self.drift_factor + 0.05)

                # Build momentum in drift direction
                if keys[pygame.K_LEFT]:
                    self.drift_momentum = min(self.drift_max_momentum, self.drift_momentum + 0.2)
                elif keys[pygame.K_RIGHT]:
                    self.drift_momentum = max(-self.drift_max_momentum, self.drift_momentum - 0.2)
            else:
                # End drift and apply boost if successful
                if self.is_drifting and abs(self.drift_momentum) > 1.0:
                    self.drift_boost = abs(self.drift_momentum) / 2

                self.is_drifting = False
                self.drift_factor = max(0, self.drift_factor - self.drift_decay)
                self.drift_momentum *= 0.9  # Gradually reduce drift momentum

        # Apply drift effects
        if self.drift_factor > 0:
            # Reduced speed during drift
            self.speed = max(2, self.speed * (1 - self.drift_factor * 0.2))

            # Apply drift momentum to movement
            drift_angle = self.angle + (self.drift_momentum * 0.1)

            # Calculate separate movement components
            # Forward movement (reduced during drift)
            forward_x = math.cos(self.angle) * self.speed * (1 - self.drift_factor * 0.5)
            forward_y = math.sin(self.angle) * self.speed * (1 - self.drift_factor * 0.5)

            # Sideways movement (based on drift momentum)
            side_x = math.cos(drift_angle) * self.speed * self.drift_factor
            side_y = math.sin(drift_angle) * self.speed * self.drift_factor

            # Combine movements
            self.x += forward_x + side_x
            self.y += forward_y + side_y

            # Apply less steering control during drift
            self.steering_speed = self.steering_speed * self.drift_control

            # Visual effects (smoke particles would be here)
        else:
            # Apply drift boost if available
            if self.drift_boost > 0:
                self.speed += self.drift_boost * 0.1
                self.drift_boost *= 0.9
                if self.drift_boost < 0.1:
                    self.drift_boost = 0

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

    def get_velocity_components(self):
        vx = math.cos(self.angle) * self.speed
        vy = math.sin(self.angle) * self.speed
        return vx, vy

    def set_velocity_from_components(self, vx, vy):
        self.speed = math.sqrt(vx * vx + vy * vy)
        if self.speed > 0.1:  # Only change angle if moving
            self.angle = math.atan2(vy, vx)
            while self.angle < 0:
                self.angle += 2 * math.pi
            while self.angle >= 2 * math.pi:
                self.angle -= 2 * math.pi

    def move(self):
        self.apply_friction()

        if self.bounce_cooldown > 0:
            self.bounce_cooldown -= 1

        # Store previous position for collision resolution
        prev_x, prev_y = self.x, self.y

        # Handle drifting mechanics
        if self.is_drifting or self.drift_factor > 0:
            self.handle_drift()
        else:
            # Normal movement
            self.x += math.cos(self.angle) * self.speed
            self.y += math.sin(self.angle) * self.speed

        # Handle collisions with walls
        if self.bounce_cooldown == 0:
            collision_wall = self.check_wall_collision(all_walls)
            if collision_wall:
                self.handle_wall_bounce(collision_wall, prev_x, prev_y)

    def check_wall_collision(self, walls):
        # Get car corners for collision detection
        car_corners = [
            (self.x + math.cos(self.angle) * self.width / 2 - math.sin(self.angle) * self.height / 2,
             self.y + math.sin(self.angle) * self.width / 2 + math.cos(self.angle) * self.height / 2),
            (self.x + math.cos(self.angle) * self.width / 2 + math.sin(self.angle) * self.height / 2,
             self.y + math.sin(self.angle) * self.width / 2 - math.cos(self.angle) * self.height / 2),
            (self.x - math.cos(self.angle) * self.width / 2 + math.sin(self.angle) * self.height / 2,
             self.y - math.sin(self.angle) * self.width / 2 - math.cos(self.angle) * self.height / 2),
            (self.x - math.cos(self.angle) * self.width / 2 - math.sin(self.angle) * self.height / 2,
             self.y - math.sin(self.angle) * self.width / 2 + math.cos(self.angle) * self.height / 2)
        ]

        # Check if any corner is outside track bounds
        for wall in walls:
            x1, y1, x2, y2 = wall
            # Check if car is outside outer track walls or inside inner track walls
            for corner_x, corner_y in car_corners:
                # Outer track bounds
                if x1 == x2:  # Vertical wall
                    if (min(y1, y2) <= corner_y <= max(y1, y2)) and abs(corner_x - x1) < 5:
                        return wall
                elif y1 == y2:  # Horizontal wall
                    if (min(x1, x2) <= corner_x <= max(x1, x2)) and abs(corner_y - y1) < 5:
                        return wall
        return None

    def handle_wall_bounce(self, wall, prev_x, prev_y):
        x1, y1, x2, y2 = wall

        # Determine if it's a horizontal or vertical wall
        if x1 == x2:  # Vertical wall
            # Reverse x component of velocity
            vx, vy = self.get_velocity_components()
            vx = -vx * self.bounce_factor
            self.set_velocity_from_components(vx, vy)
        elif y1 == y2:  # Horizontal wall
            # Reverse y component of velocity
            vx, vy = self.get_velocity_components()
            vy = -vy * self.bounce_factor
            self.set_velocity_from_components(vx, vy)

        # Move car back to previous position and apply new velocity
        self.x, self.y = prev_x, prev_y
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Set cooldown to prevent multiple bounces
        self.bounce_cooldown = 5

    def check_car_collision(self, other_car):
        # Simple circle-circle collision
        dist = math.sqrt((self.x - other_car.x) ** 2 + (self.y - other_car.y) ** 2)
        return dist < self.collision_radius + other_car.collision_radius

    def handle_car_collision(self, other_car):
        if self.bounce_cooldown == 0 and other_car.bounce_cooldown == 0:
            # Calculate normal vector (displacement from other car to this car)
            nx = self.x - other_car.x
            ny = self.y - other_car.y
            dist = math.sqrt(nx * nx + ny * ny)

            # Avoid division by zero
            if dist < 0.1:
                nx, ny = 1, 0
            else:
                nx, ny = nx / dist, ny / dist

            # Get velocity components
            v1x, v1y = self.get_velocity_components()
            v2x, v2y = other_car.get_velocity_components()

            # Calculate relative velocity along normal
            v1n = v1x * nx + v1y * ny
            v2n = v2x * nx + v2y * ny

            # Elastic collision formula (conservation of momentum and energy)
            # Using coefficient of restitution (elasticity) of 0.8
            elasticity = 0.8
            m1, m2 = self.mass, other_car.mass

            # Calculate new normal velocities
            v1n_new = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2) * elasticity
            v2n_new = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2) * elasticity

            # Convert normal velocity back to vector form
            v1n_dx = v1n_new * nx
            v1n_dy = v1n_new * ny
            v2n_dx = v2n_new * nx
            v2n_dy = v2n_new * ny

            # Tangential component remains unchanged
            v1tx = v1x - v1n * nx
            v1ty = v1y - v1n * ny
            v2tx = v2x - v2n * nx
            v2ty = v2y - v2n * ny

            # Set new velocities
            self.set_velocity_from_components(v1n_dx + v1tx, v1n_dy + v1ty)
            other_car.set_velocity_from_components(v2n_dx + v2tx, v2n_dy + v2ty)

            # Move cars apart to prevent sticking
            overlap = self.collision_radius + other_car.collision_radius - dist
            if overlap > 0:
                self.x += nx * overlap / 2
                self.y += ny * overlap / 2
                other_car.x -= nx * overlap / 2
                other_car.y -= ny * overlap / 2

            # Set cooldown to prevent rapid repeated collisions
            self.bounce_cooldown = 5
            other_car.bounce_cooldown = 5

    def ai_control(self):
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

        # Draw drift effects (tire tracks/smoke)
        if self.drift_factor > 0.2 and self.speed > 1:
            for i in range(2):
                # Calculate tire positions
                tire_x = self.x - math.cos(self.angle + math.pi / 4) * 5
                tire_y = self.y - math.sin(self.angle + math.pi / 4) * 5

                # Draw skid marks
                track_size = int(3 + self.drift_factor * 3)
                track_alpha = int(100 + self.drift_factor * 155)
                track_color = (50, 50, 50, track_alpha)

                smoke = pygame.Surface((track_size * 2, track_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(smoke, track_color, (track_size, track_size), track_size)
                surface.blit(smoke, (tire_x - track_size, tire_y - track_size))

        # Draw boost effect
        if self.drift_boost > 0:
            boost_length = int(self.drift_boost * 5)
            boost_width = int(3 + self.drift_boost)
            boost_angle = self.angle + math.pi  # Opposite direction of car

            boost_x = self.x - math.cos(boost_angle) * 10
            boost_y = self.y - math.sin(boost_angle) * 10

            boost_end_x = boost_x - math.cos(boost_angle) * boost_length
            boost_end_y = boost_y - math.sin(boost_angle) * boost_length

            pygame.draw.line(surface, (255, 140, 0), (boost_x, boost_y),
                             (boost_end_x, boost_end_y), boost_width)

        # Original waypoint marker code
        if self.is_ai:
            target_x, target_y = waypoints[self.current_waypoint]
            pygame.draw.circle(surface, YELLOW, (target_x, target_y), 5, 1)

# Create player and AI cars
player = Car(150, 150, RED)
ai_car = Car(150, 180, BLUE, is_ai=True)

# List of all cars for collision handling
all_cars = [player, ai_car]

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

    # Handle car-car collisions
    for i, car1 in enumerate(all_cars):
        for car2 in all_cars[i + 1:]:
            if car1.check_car_collision(car2):
                car1.handle_car_collision(car2)

    # Drawing
    screen.fill(GRASS)

    # Draw track
    pygame.draw.rect(screen, GRAY, (100, 100, 600, 400))
    pygame.draw.rect(screen, GRASS, (200, 200, 400, 200))

    # Draw walls
    for x1, y1, x2, y2 in all_walls:
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

    screen.blit(speed_text, (10, 10))
    screen.blit(ai_text, (10, 30))

    # Flip display
    pygame.display.flip()

pygame.quit()