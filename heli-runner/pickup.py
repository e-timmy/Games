import pygame
import math


class Pickup:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 15
        self.rotation = 0
        self.rotation_speed = 2

    def update(self):
        self.rotation += self.rotation_speed

    def draw(self, screen, offset):
        pass  # To be implemented by child classes

    def collides_with(self, player):
        dx = self.x - player.body.position.x
        dy = self.y - player.body.position.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance < self.radius + player.size / 2


class ShieldPickup(Pickup):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.color = (0, 191, 255)  # Deep Sky Blue
        self.radius = 20  # Increased size for better visibility

    def draw(self, screen, offset):
        adjusted_x = int(self.x + offset)
        adjusted_y = int(self.y)

        # Draw the spinning shield with increased visibility
        # Outer ring
        pygame.draw.circle(screen, self.color, (adjusted_x, adjusted_y), self.radius, 2)

        # Inner rotating star
        points = []
        for i in range(8):
            angle = math.radians(self.rotation + i * 45)
            point_x = adjusted_x + int((self.radius - 5) * math.cos(angle))
            point_y = adjusted_y + int((self.radius - 5) * math.sin(angle))
            points.append((point_x, point_y))

        pygame.draw.polygon(screen, self.color, points, 2)

        # Center dot
        pygame.draw.circle(screen, self.color, (adjusted_x, adjusted_y), 3)

class AutoPilotPickup(Pickup):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.color = (255, 255, 255)  # White color
        self.radius = 20

    def draw(self, screen, offset):
        adjusted_x = int(self.x + offset)
        adjusted_y = int(self.y)

        # Draw outer circle
        pygame.draw.circle(screen, self.color, (adjusted_x, adjusted_y), self.radius, 2)

        # Draw "A" symbol
        angle = math.radians(self.rotation)
        points = []
        for i in range(3):
            point_angle = angle + (2 * math.pi * i / 3)
            point_x = adjusted_x + int((self.radius - 5) * math.cos(point_angle))
            point_y = adjusted_y + int((self.radius - 5) * math.sin(point_angle))
            points.append((point_x, point_y))

        pygame.draw.lines(screen, self.color, True, points, 2)
        center_x = sum(p[0] for p in points) // 3
        center_y = sum(p[1] for p in points) // 3
        pygame.draw.circle(screen, self.color, (center_x, center_y), 3)


class ScatterShotPickup(Pickup):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.color = (255, 50, 50)  # Bright Red
        self.radius = 20  # Same size as shield for consistency

    def draw(self, screen, offset):
        adjusted_x = int(self.x + offset)
        adjusted_y = int(self.y)

        # Draw outer circle
        pygame.draw.circle(screen, self.color, (adjusted_x, adjusted_y), self.radius, 2)

        # Draw rotating bullet pattern
        for i in range(4):
            angle = math.radians(self.rotation + i * 90)
            start_x = adjusted_x + int((self.radius - 8) * math.cos(angle))
            start_y = adjusted_y + int((self.radius - 8) * math.sin(angle))
            end_x = adjusted_x + int((self.radius - 2) * math.cos(angle))
            end_y = adjusted_y + int((self.radius - 2) * math.sin(angle))
            pygame.draw.line(screen, self.color, (start_x, start_y), (end_x, end_y), 2)

        # Center dot
        pygame.draw.circle(screen, self.color, (adjusted_x, adjusted_y), 3)


class SlowDownPickup(Pickup):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.color = (0, 255, 0)  # Bright Green
        self.radius = 20  # Same size as other pickups for consistency

    def draw(self, screen, offset):
        adjusted_x = int(self.x + offset)
        adjusted_y = int(self.y)

        # Draw outer circle
        pygame.draw.circle(screen, self.color, (adjusted_x, adjusted_y), self.radius, 2)

        # Draw rotating hourglass pattern
        for i in range(2):
            angle = math.radians(self.rotation + i * 180)
            top_x = adjusted_x + int((self.radius - 5) * math.cos(angle))
            top_y = adjusted_y + int((self.radius - 5) * math.sin(angle))
            bottom_x = adjusted_x + int((self.radius - 5) * math.cos(angle + math.pi))
            bottom_y = adjusted_y + int((self.radius - 5) * math.sin(angle + math.pi))
            pygame.draw.line(screen, self.color, (top_x, top_y), (bottom_x, bottom_y), 2)

        # Center dot
        pygame.draw.circle(screen, self.color, (adjusted_x, adjusted_y), 3)


class GuidedBullet:
    def __init__(self, x, y, target):
        self.x = x
        self.y = y
        self.target = target
        self.speed = 5
        self.radius = 3
        self.color = (255, 0, 0)  # Red
        self.offset = 2

    def update(self, obstacles, floor_points, ceiling_points):
        if self.target is None or not self.target.is_active:
            return True  # Remove bullet if target is gone

        dx = self.target.x - self.x
        dy = self.target.y - self.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < self.speed:
            return True  # Bullet has reached its target

        direction = (dx / distance, dy / distance)

        new_x = self.x + direction[0] * self.speed
        new_y = self.y + direction[1] * self.speed

        # Check for collisions with cave walls
        if self.collides_with_cave(new_x, new_y, floor_points, ceiling_points):
            # Find a new path
            new_direction = self.find_new_direction(obstacles, floor_points, ceiling_points)
            if new_direction:
                new_x = self.x + new_direction[0] * self.speed
                new_y = self.y + new_direction[1] * self.speed
            else:
                return True  # Remove bullet if no path found

        self.x = new_x
        self.y = new_y
        return False

    def draw(self, screen, offset):
        pygame.draw.circle(screen, self.color, (int(self.x + offset), int(self.y)), self.radius)

    def collides_with_cave(self, x, y, floor_points, ceiling_points):
        for points in [floor_points, ceiling_points]:
            for i in range(len(points) - 1):
                x1, y1, _ = points[i]
                x2, y2, _ = points[i + 1]
                if x1 <= x <= x2:
                    wall_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                    if abs(y - wall_y) < self.radius:
                        return True
        return False

    def find_new_direction(self, obstacles, floor_points, ceiling_points):
        angles = [0, 45, -45, 90, -90, 135, -135, 180]
        for angle in angles:
            rad = math.radians(angle)
            dx = math.cos(rad)
            dy = math.sin(rad)
            new_x = self.x + dx * self.speed
            new_y = self.y + dy * self.speed
            if not self.collides_with_cave(new_x, new_y, floor_points, ceiling_points):
                return (dx, dy)
        return None

class GuidedScattershot:
    def __init__(self, player, obstacles):
        self.bullets = []
        for obstacle in obstacles:
            self.bullets.append(GuidedBullet(player.body.position.x, player.body.position.y, obstacle))

    def update(self, obstacles, floor_points, ceiling_points):
        for bullet in self.bullets[:]:
            if bullet.update(obstacles, floor_points, ceiling_points):
                self.bullets.remove(bullet)

    def draw(self, screen, offset):
        for bullet in self.bullets:
            bullet.draw(screen, offset)

    def is_active(self):
        return len(self.bullets) > 0