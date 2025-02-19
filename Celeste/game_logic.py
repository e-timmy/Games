import random
import math
from dataclasses import dataclass
from typing import List, Dict, Any

# Constants (previously in settings.py)
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
GRAVITY = 0.8
JUMP_SPEED = -15
MOVE_SPEED = 5
BOOST_SPEED = 15
BOOST_DURATION = 10
CLIMB_SPEED = 4
CLIMB_TIME_LIMIT = 180


@dataclass
class Platform:
    x: float
    y: float
    width: float
    height: float
    is_moving: bool = False
    start_x: float = 0
    end_x: float = 0
    speed: float = 0
    direction: int = 1

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'is_moving': self.is_moving
        }


class LevelGenerator:
    def __init__(self):
        self.min_platform_length = 100
        self.max_platform_length = 250
        self.platform_height = 20
        self.vertical_spacing = 120
        self.horizontal_spacing = 200
        self.difficulty = 1
        self.max_difficulty = 5

    def generate_level(self) -> List[Platform]:
        platforms = []

        # Ground
        platforms.append(Platform(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40))

        # Walls
        platforms.append(Platform(0, 0, 40, SCREEN_HEIGHT - 140))
        platforms.append(Platform(SCREEN_WIDTH - 40, 100, 40, SCREEN_HEIGHT - 100))

        # Generate platforms
        current_height = SCREEN_HEIGHT - 200
        current_x = 200

        while current_height > 200:
            platform_length = random.randint(
                max(80, self.min_platform_length - self.difficulty * 15),
                max(120, self.max_platform_length - self.difficulty * 20)
            )

            if current_x + platform_length > SCREEN_WIDTH - 100:
                platform_length = SCREEN_WIDTH - current_x - 100

            new_platform = Platform(
                current_x,
                current_height,
                platform_length,
                self.platform_height
            )

            if random.random() < 0.2 and self.difficulty > 1:
                new_platform.is_moving = True
                new_platform.start_x = max(50, current_x - 100)
                new_platform.end_x = min(SCREEN_WIDTH - 50 - platform_length, current_x + 100)
                new_platform.speed = random.randint(1, 3)

            platforms.append(new_platform)

            current_height -= random.randint(
                self.vertical_spacing,
                self.vertical_spacing + self.difficulty * 10
            )

            if current_x < SCREEN_WIDTH / 2:
                current_x += random.randint(
                    self.horizontal_spacing,
                    self.horizontal_spacing + self.difficulty * 20
                )
            else:
                current_x = random.randint(100, SCREEN_WIDTH // 2 - 100)

        return platforms


class GameState:
    def __init__(self):
        self.level_generator = LevelGenerator()
        self.reset()

    def reset(self):
        self.platforms = self.level_generator.generate_level()
        self.player_x = 100
        self.player_y = SCREEN_HEIGHT - 150
        self.player_width = 40
        self.player_height = 60
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.on_wall = False
        self.is_climbing = False
        self.is_boosting = False
        self.boost_timer = 0
        self.climb_timer = 0
        self.facing_right = True
        self.win = False

    def update(self, input_data: Dict[str, Any]):
        # Handle input
        keys = input_data
        print(f"Received input: {keys}")  # Debug log

        # Define fixed time step
        dt = 1 / 60  # 60 FPS

        # Movement
        if not self.is_boosting and not self.is_climbing:
            target_vel_x = (keys.get('right', 0) - keys.get('left', 0)) * MOVE_SPEED
            # Smooth acceleration
            self.vel_x = self.vel_x + (target_vel_x - self.vel_x) * 0.3

            if self.vel_x > 0:
                self.facing_right = True
            elif self.vel_x < 0:
                self.facing_right = False

        # Jumping
        if keys.get('z', 0) and self.on_ground:
            self.vel_y = JUMP_SPEED
            self.on_ground = False

        # Climbing
        if keys.get('c', 0) and self.on_wall and self.climb_timer < CLIMB_TIME_LIMIT:
            self.is_climbing = True
            target_vel_y = (keys.get('down', 0) - keys.get('up', 0)) * CLIMB_SPEED
            self.vel_y = self.vel_y + (target_vel_y - self.vel_y) * 0.3
            self.vel_x = 0
            self.climb_timer += 1
        else:
            self.is_climbing = False
            if self.on_ground:
                self.climb_timer = 0

        # Boosting
        if keys.get('x', 0) and not self.is_boosting:
            self.start_boost(keys)

        # Update boost
        if self.is_boosting:
            self.boost_timer -= 1
            if self.boost_timer <= 0:
                self.is_boosting = False
                # Smooth transition from boost
                self.vel_x *= 0.5
                self.vel_y *= 0.5
            else:
                self.vel_x = self.boost_direction_x * BOOST_SPEED
                self.vel_y = self.boost_direction_y * BOOST_SPEED

        # Apply gravity if not climbing or boosting
        if not self.is_climbing and not self.is_boosting:
            self.vel_y += GRAVITY * dt

        # Apply air resistance
        if not self.on_ground and not self.is_boosting:
            self.vel_x *= 0.99
            if abs(self.vel_x) < 0.01:
                self.vel_x = 0

        # Update position with fixed time step
        self.player_x += self.vel_x * dt
        self.player_y += self.vel_y * dt

        # Update moving platforms
        for platform in self.platforms:
            if platform.is_moving:
                platform.x += platform.speed * platform.direction * dt
                if platform.x <= platform.start_x or platform.x >= platform.end_x:
                    platform.direction *= -1

        # Check collisions and update state
        self.check_collisions()

        # Check win condition
        self.check_win_condition()

        print(f"Player position: ({self.player_x}, {self.player_y})")  # Debug log

    def start_boost(self, keys):
        dx = keys.get('right', 0) - keys.get('left', 0)
        dy = keys.get('down', 0) - keys.get('up', 0)

        if dx == 0 and dy == 0:
            dx = 1 if self.facing_right else -1

        length = (dx ** 2 + dy ** 2) ** 0.5
        if length != 0:
            self.boost_direction_x = dx / length
            self.boost_direction_y = dy / length

        self.is_boosting = True
        self.boost_timer = BOOST_DURATION

    def check_collisions(self):
        self.on_ground = False
        self.on_wall = False

        for platform in self.platforms:
            # Create player rect
            player_rect = {
                'left': self.player_x,
                'right': self.player_x + self.player_width,
                'top': self.player_y,
                'bottom': self.player_y + self.player_height
            }

            # Create platform rect
            platform_rect = {
                'left': platform.x,
                'right': platform.x + platform.width,
                'top': platform.y,
                'bottom': platform.y + platform.height
            }

            # Check collision
            if (player_rect['right'] > platform_rect['left'] and
                    player_rect['left'] < platform_rect['right'] and
                    player_rect['bottom'] > platform_rect['top'] and
                    player_rect['top'] < platform_rect['bottom']):

                # Vertical collision
                if self.vel_y > 0:
                    self.player_y = platform_rect['top'] - self.player_height
                    self.on_ground = True
                    self.vel_y = 0
                elif self.vel_y < 0:
                    self.player_y = platform_rect['bottom']
                    self.vel_y = 0

                # Horizontal collision
                if self.vel_x > 0:
                    self.player_x = platform_rect['left'] - self.player_width
                    self.on_wall = True
                    self.vel_x = 0
                elif self.vel_x < 0:
                    self.player_x = platform_rect['right']
                    self.on_wall = True
                    self.vel_x = 0

        # Keep player in bounds
        self.player_x = max(0, min(self.player_x, SCREEN_WIDTH - self.player_width))
        self.player_y = max(0, min(self.player_y, SCREEN_HEIGHT - self.player_height))

    def check_win_condition(self):
        # Check if player is in win zone (top right area)
        if (self.player_x > SCREEN_WIDTH - 80 and
                self.player_y < 100):
            self.win = True

    def get_state(self) -> Dict[str, Any]:
        return {
            'player': {
                'x': self.player_x,
                'y': self.player_y,
                'width': self.player_width,
                'height': self.player_height,
                'vel_x': self.vel_x,
                'vel_y': self.vel_y,
                'on_ground': self.on_ground,
                'on_wall': self.on_wall,
                'is_climbing': self.is_climbing,
                'is_boosting': self.is_boosting,
                'facing_right': self.facing_right
            },
            'platforms': [p.to_dict() for p in self.platforms],
            'win': self.win
        }
