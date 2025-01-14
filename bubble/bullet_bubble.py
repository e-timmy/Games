import pygame
import math
import random
from bubble import Bubble
from settings import (BULLET_SPEED, BULLET_WOBBLE_AMOUNT, MIN_BUBBLE_SPEED,
                     MAX_BUBBLE_SPEED, WOBBLE_AMOUNT, SCREEN_WIDTH, SCREEN_HEIGHT,
                     PROPULSION_SPEED, MIN_PROPULSION_SIZE, MAX_PROPULSION_SIZE)

class BulletBubble(Bubble):
    def __init__(self, x, y, size, angle, color):
        # Calculate velocity components based on angle
        dx = BULLET_SPEED * math.cos(math.radians(angle))
        dy = BULLET_SPEED * math.sin(math.radians(angle))
        super().__init__(x, y, size, dx, dy)
        self.color = color
        self.is_bullet = True
        self.has_split_bubble = False
        # Less wobble while in bullet mode
        self.wobble_amount = BULLET_WOBBLE_AMOUNT

    def convert_to_regular_bubble(self):
        """Convert bullet to regular floating bubble after splitting"""
        self.has_split_bubble = True
        # Randomize direction slightly
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(MIN_BUBBLE_SPEED, MAX_BUBBLE_SPEED)
        self.dx = speed * math.cos(angle)
        self.dy = speed * math.sin(angle)
        # Reset wobble to normal bubble amount
        self.wobble_amount = WOBBLE_AMOUNT

    def split_bubble(self, target_bubble):
        """Calculate split sizes and create SplittingBubbles"""

        try:
            # Vector from bubble center to bullet impact
            dx = self.x - target_bubble.x
            dy = self.y - target_bubble.y

            # Normalize the vector
            dist = math.sqrt(dx * dx + dy * dy)
            if dist == 0:
                return None

            dx /= dist
            dy /= dist

            # Calculate intersection point
            intersection_x = target_bubble.x + dx * target_bubble.size
            intersection_y = target_bubble.y + dy * target_bubble.size

            # Calculate split ratio based on where bullet hit
            # 0.5 means perfect center hit, closer to 0 or 1 means edge hit
            split_ratio = 0.5 + (dist - target_bubble.size) / (2 * target_bubble.size)
            split_ratio = max(0.1, min(0.9, split_ratio))  # Clamp between 0.1 and 0.9

            # Calculate areas for the two new bubbles
            original_area = math.pi * target_bubble.size ** 2
            area1 = original_area * split_ratio
            area2 = original_area * (1 - split_ratio)

            # Calculate radii for new bubbles
            radius1 = math.sqrt(area1 / math.pi)
            radius2 = math.sqrt(area2 / math.pi)

            # Calculate perpendicular vector for separation
            perp_dx = -dy
            perp_dy = dx

            # Calculate angles for SplittingBubbles
            impact_angle = math.atan2(dy, dx)

            # Create two SplittingBubbles
            splitting_bubble1 = SplittingBubble(
                target_bubble.x, target_bubble.y, radius1, target_bubble.color,
                impact_angle, impact_angle + math.pi,
                              perp_dx * target_bubble.dx, perp_dy * target_bubble.dy
            )

            splitting_bubble2 = SplittingBubble(
                target_bubble.x, target_bubble.y, radius2, target_bubble.color,
                impact_angle + math.pi, impact_angle + 2 * math.pi,
                -perp_dx * target_bubble.dx, -perp_dy * target_bubble.dy
            )

            # Only set has_split_bubble flag if everything succeeded
            self.convert_to_regular_bubble()

            return splitting_bubble1, splitting_bubble2

        except Exception as e:
            print(f"Error in split_bubble: {e}")
            return None

    def update(self):
        # Use parent class update which includes wobble
        super().update()


class SplittingBubble:
    def __init__(self, x, y, radius, color, start_angle, end_angle, dx, dy):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.dx = dx
        self.dy = dy
        self.animation_time = 0
        self.animation_duration = 0.3  # seconds

    def update(self, dt):
        # Limit maximum dt to prevent large jumps
        dt = min(dt, 0.1)

        self.animation_time += dt
        progress = min(self.animation_time / self.animation_duration, 1)

        # Move slightly based on initial velocity
        new_x = self.x + self.dx * dt
        new_y = self.y + self.dy * dt

        # Bounce off screen edges
        if new_x - self.radius <= 0 or new_x + self.radius >= SCREEN_WIDTH:
            self.dx *= -1
            new_x = max(self.radius, min(SCREEN_WIDTH - self.radius, new_x))
        if new_y - self.radius <= 0 or new_y + self.radius >= SCREEN_HEIGHT:
            self.dy *= -1
            new_y = max(self.radius, min(SCREEN_HEIGHT - self.radius, new_y))

        self.x = new_x
        self.y = new_y

        # Adjust angles to create closing effect
        angle_adjustment = progress * math.pi
        self.start_angle += angle_adjustment
        self.end_angle -= angle_adjustment

    def is_animation_complete(self):
        return self.animation_time >= self.animation_duration

    def draw(self, screen):
        # Draw the arc
        rect = pygame.Rect(
            self.x - self.radius,
            self.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )

        # Draw filled semi-circle
        points = [
            (self.x, self.y),  # Center point
        ]

        # Add points along the arc
        num_points = 20  # Number of points to approximate the arc
        for i in range(num_points + 1):
            angle = self.start_angle + (self.end_angle - self.start_angle) * (i / num_points)
            px = self.x + math.cos(angle) * self.radius
            py = self.y + math.sin(angle) * self.radius
            points.append((px, py))

        # Draw filled polygon
        if len(points) >= 3:
            pygame.draw.polygon(screen, self.color, points)

    def to_bubble(self):
        """Convert to a regular Bubble instance"""
        from bubble import Bubble
        new_bubble = Bubble(self.x, self.y, self.radius, self.dx, self.dy)
        new_bubble.color = self.color
        return new_bubble


class PropulsionBubble(Bubble):
    def __init__(self, x, y, size, player_angle, player_dx, player_dy):
        # Calculate velocity opposite to player's movement direction
        # Base velocity on both the player's facing direction and current movement
        thrust_angle = math.radians(player_angle)

        # Combine thrust direction with current movement
        total_dx = -player_dx - PROPULSION_SPEED * math.cos(thrust_angle)
        total_dy = -player_dy - PROPULSION_SPEED * math.sin(thrust_angle)

        # Add some randomness to the direction
        spread = random.uniform(-0.3, 0.3)
        final_dx = total_dx + spread
        final_dy = total_dy + spread

        super().__init__(x, y, size, final_dx, final_dy)
        self.color = (200, 200, 255)  # Light blue color
        self.wobble_amount = WOBBLE_AMOUNT * 0.5  # Reduced wobble for propulsion bubbles

    def update(self):
        super().update()
        # Normal bubble behavior - no special updates needed as it's now a regular game entity

    def draw(self, screen):
        super().draw(screen)  # Use regular bubble drawing