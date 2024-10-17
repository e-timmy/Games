import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Rotation Minigame with Ball")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Set up the clock
clock = pygame.time.Clock()


class Ball:
    def __init__(self, x, y, radius):
        self.position = pygame.math.Vector2(x, y)
        self.radius = radius
        self.velocity = pygame.math.Vector2(0, 0)
        self.sub_steps = 3  # Number of physics updates per frame

    def update(self, container_points):
        # Perform multiple sub-steps for better physics accuracy
        dt = 1.0 / self.sub_steps
        for _ in range(self.sub_steps):
            self._update_step(container_points, dt)

    def _update_step(self, container_points, dt):
        # Light gravity
        gravity = pygame.math.Vector2(0, 0.05) * dt
        self.velocity += gravity

        # Minimal friction
        self.velocity *= 0.999

        # Calculate next position
        next_pos = self.position + self.velocity

        # Check collisions with container edges
        for i in range(len(container_points)):
            p1 = container_points[i]
            p2 = container_points[(i + 1) % len(container_points)]

            # Check collision with each edge
            if self.line_circle_collision(p1, p2, next_pos):
                # Calculate normal vector of the edge
                edge = p2 - p1
                normal = pygame.math.Vector2(-edge.y, edge.x).normalize()

                # Calculate reflection vector using the incoming velocity
                # R = V - 2(V · N)N where V is velocity vector and N is normal vector
                dot_product = self.velocity.dot(normal)
                reflection = self.velocity - 2 * dot_product * normal

                # Apply bounce with energy retention
                self.velocity = reflection * 0.9  # Bounce coefficient

                # Add a small boost to maintain energy
                self.velocity *= 1.1

                # Position correction to prevent sticking
                penetration = self.radius - (next_pos - self.get_closest_point(p1, p2, next_pos)).length()
                if penetration > 0:
                    next_pos += normal * penetration * 1.1

        # Update position
        self.position = next_pos

        # Safety check - if somehow ball gets out, force it back in
        if not self.is_inside_container(container_points):
            self.force_inside_container(container_points)


    def line_circle_collision(self, p1, p2, circle_pos):
        # Get closest point on line segment
        closest = self.get_closest_point(p1, p2, circle_pos)

        # Check if distance is less than radius
        return (circle_pos - closest).length() < self.radius

    def get_closest_point(self, p1, p2, point):
        line_vec = p2 - p1
        point_vec = point - p1
        line_length = line_vec.length()
        if line_length == 0:
            return pygame.math.Vector2(p1)

        t = max(0, min(1, point_vec.dot(line_vec) / (line_length * line_length)))
        return p1 + line_vec * t

    def is_inside_container(self, container_points):
        # Simple point-in-polygon test
        intersections = 0
        ray_end = self.position + pygame.math.Vector2(1000, 0)  # Cast ray to the right

        for i in range(len(container_points)):
            p1 = container_points[i]
            p2 = container_points[(i + 1) % len(container_points)]

            if self.ray_intersects_segment(self.position, ray_end, p1, p2):
                intersections += 1

        return intersections % 2 == 1

    def ray_intersects_segment(self, ray_start, ray_end, p1, p2):
        # Check if a ray intersects with a line segment
        def ccw(A, B, C):
            return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

        return ccw(ray_start, p1, p2) != ccw(ray_end, p1, p2) and \
            ccw(ray_start, ray_end, p1) != ccw(ray_start, ray_end, p2)

    def force_inside_container(self, container_points):
        # Find center of container
        center = pygame.math.Vector2(0, 0)
        for p in container_points:
            center += p
        center /= len(container_points)

        # Move ball towards center
        to_center = center - self.position
        if to_center.length() > 0:
            self.position = center - to_center.normalize() * (self.radius * 2)
            self.velocity *= 0.5

    def draw(self, surface):
        pygame.draw.circle(surface, RED, (int(self.position.x), int(self.position.y)), self.radius)


class Container:
    def __init__(self, center_x, center_y, size):
        self.center = pygame.math.Vector2(center_x, center_y)
        self.size = size
        self.rotation = 0
        self.points = self.calculate_points()

    def calculate_points(self):
        half_size = self.size / 2
        points = [
            pygame.math.Vector2(-half_size, -half_size),
            pygame.math.Vector2(half_size, -half_size),
            pygame.math.Vector2(half_size, half_size),
            pygame.math.Vector2(-half_size, half_size)
        ]

        angle = math.radians(self.rotation)
        for point in points:
            x = point.x * math.cos(angle) - point.y * math.sin(angle)
            y = point.x * math.sin(angle) + point.y * math.cos(angle)
            point.x = x + self.center.x
            point.y = y + self.center.y

        return points

    def update_rotation(self, new_rotation):
        self.rotation = new_rotation
        self.points = self.calculate_points()

    def draw(self, surface):
        points = [(int(p.x), int(p.y)) for p in self.points]
        pygame.draw.lines(surface, WHITE, True, points, 2)


# Set up game loop
rotation = 0
target_rotation = 0
rotation_speed = 2
rotation_count = 0
running = True

# Create the ball and container
container_size = min(width, height) - 200
container = Container(width // 2, height // 2, container_size)
ball = Ball(width // 2, height // 2, 10)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                rotation_count -= 1
            elif event.key == pygame.K_LEFT:
                rotation_count += 1

    target_rotation = rotation_count * 90

    if rotation != target_rotation:
        if rotation < target_rotation:
            rotation = min(rotation + rotation_speed, target_rotation)
        else:
            rotation = max(rotation - rotation_speed, target_rotation)
        container.update_rotation(rotation)

    screen.fill(BLACK)

    ball.update(container.points)
    container.draw(screen)
    ball.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()