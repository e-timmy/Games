import pygame
import math
import random
from constants import (TILE_SIZE, CHARACTER_SCALE_FACTOR, BOSS_SCALE_FACTOR, BOSS_HEALTH_MULTIPLIER,
                       BOSS_DAMAGE_MULTIPLIER, BOSS_MOVEMENT_AREA_SIZE)
# Load sprite sheet
sprite_sheet = pygame.image.load('assets/town_rpg_pack/graphics/characters/characters.png')


class NonPlayerCharacter:
    def __init__(self, sprite_position):
        self.x = 0
        self.y = 0
        self.prev_x = 0
        self.prev_y = 0
        self.direction = 0
        self.animation_frame = 1
        self.animation_timer = 0
        self.moving = False
        self.speed = 1  # Reduced speed for NPCs

        # Get the correct row and column for this character
        self.row, self.col = sprite_position

        # Size of each sprite
        self.sprite_width = 16
        self.sprite_height = 16

        # Calculate sprite position in sprite sheet
        self.row_offset = self.row * 4
        self.col_offset = self.col * 3

    def get_current_sprite(self):
        sprite_x = (self.col_offset + self.animation_frame) * self.sprite_width
        sprite_y = (self.row_offset + self.direction) * self.sprite_height

        sprite = pygame.Surface((self.sprite_width, self.sprite_height))
        sprite.set_colorkey((0, 0, 0))

        sprite.blit(sprite_sheet, (0, 0),
                    (sprite_x, sprite_y, self.sprite_width, self.sprite_height))

        return sprite

    def update(self):
        if self.moving:
            self.animation_timer += 1
            if self.animation_timer >= 15:  # Reduced animation speed for NPCs
                self.animation_timer = 0
                self.animation_frame = (self.animation_frame + 1) % 3
        else:
            self.animation_frame = 1

    def move(self, dx, dy, environment):
        new_x = self.x + dx
        new_y = self.y + dy

        if dx != 0 or dy != 0:
            # Normalize diagonal movement
            length = math.sqrt(dx * dx + dy * dy)
            if length != 0:
                dx = dx / length * self.speed
                dy = dy / length * self.speed

            # Check for collision
            if not environment.is_collision(new_x, new_y):
                self.prev_x = self.x
                self.prev_y = self.y
                self.x = new_x
                self.y = new_y
                self.moving = True

                # Set direction based on movement
                if abs(dx) > abs(dy):
                    self.direction = 1 if dx < 0 else 2
                else:
                    self.direction = 3 if dy < 0 else 0
            else:
                # If collision, stop moving
                self.moving = False
        else:
            self.moving = False


class NPC(NonPlayerCharacter):
    def __init__(self, sprite_position):
        super().__init__(sprite_position)
        self.dialogue = []

        # Patrolling variables
        self.patrol_timer = 0
        self.max_patrol_time = random.randint(180, 300)  # 3-5 seconds of movement
        self.pause_timer = 0
        self.max_pause_time = random.randint(60, 180)  # 1-3 seconds of pause
        self.horizontal_direction = random.choice([-1, 1])
        self.vertical_direction = random.choice([-1, 1])
        self.is_paused = False

    def patrol_with_stops(self, environment):
        # Check if currently paused
        if self.is_paused:
            self.pause_timer -= 1
            self.moving = False

            # End pause
            if self.pause_timer <= 0:
                self.is_paused = False
                self.patrol_timer = random.randint(180, 300)
                # Choose movement direction
                movement_type = random.choice(['horizontal', 'vertical'])
                if movement_type == 'horizontal':
                    self.horizontal_direction = random.choice([-1, 1])
                    self.vertical_direction = 0
                else:
                    self.vertical_direction = random.choice([-1, 1])
                    self.horizontal_direction = 0
            return

        # Decrement patrol timer
        self.patrol_timer -= 1

        # Move
        dx = self.horizontal_direction * self.speed
        dy = self.vertical_direction * self.speed
        self.move(dx, dy, environment)

        # Update direction based on movement
        if dx != 0:
            self.direction = 1 if dx < 0 else 2  # Left or Right
        elif dy != 0:
            self.direction = 3 if dy < 0 else 0  # Up or Down

        # Check if patrol time is over
        if self.patrol_timer <= 0:
            self.is_paused = True
            self.pause_timer = random.randint(60, 180)


class Monster(NonPlayerCharacter):
    def __init__(self, sprite_position, base_speed, base_health, base_perception):
        super().__init__(sprite_position)
        self.base_speed = base_speed
        self.base_health = base_health
        self.base_perception = base_perception

        # Base monster attributes
        self.speed = base_speed
        self.health = base_health
        self.max_health = base_health
        self.perception_range = base_perception
        self.damage = 10
        self.is_boss = False

        # Movement area
        self.area_center_x = self.x
        self.area_center_y = self.y
        self.area_size = 150

        # Initialize movement variables
        self.change_direction_timer = random.randint(30, 90)
        self.current_dx = 0
        self.current_dy = 0

        # Scaling factor
        self.quadrant_multiplier = 1.0

    def get_current_sprite(self):
        # Get base sprite from parent class without any scaling
        base_sprite = super().get_current_sprite()

        # Apply character scale factor
        scaled_width = self.sprite_width * CHARACTER_SCALE_FACTOR
        scaled_height = self.sprite_height * CHARACTER_SCALE_FACTOR

        # If it's a boss, apply additional scaling
        if self.is_boss:
            scaled_width *= BOSS_SCALE_FACTOR
            scaled_height *= BOSS_SCALE_FACTOR

        return pygame.transform.scale(base_sprite, (int(scaled_width), int(scaled_height)))

    def set_as_boss(self):
        self.is_boss = True
        self.health *= BOSS_HEALTH_MULTIPLIER
        self.max_health *= BOSS_HEALTH_MULTIPLIER
        self.damage *= BOSS_DAMAGE_MULTIPLIER
        self.area_size = BOSS_MOVEMENT_AREA_SIZE
        self.perception_range *= 1.5  # Bosses have 50% larger perception range

    def random_movement(self, environment):
        # Change direction periodically
        self.change_direction_timer -= 1
        if self.change_direction_timer <= 0:
            # More controlled random movement
            movement_type = random.choice(['horizontal', 'vertical'])

            if movement_type == 'horizontal':
                self.current_dx = random.choice([-1, 1]) * random.uniform(0.5, 1)
                self.current_dy = 0
            else:
                self.current_dy = random.choice([-1, 1]) * random.uniform(0.5, 1)
                self.current_dx = 0

            # Stronger pull towards center for bosses
            if self.is_boss:
                pull_strength = 0.1
                dx_to_center = self.area_center_x - self.x
                dy_to_center = self.area_center_y - self.y
                distance_to_center = math.sqrt(dx_to_center ** 2 + dy_to_center ** 2)

                if distance_to_center > self.area_size:
                    self.current_dx += (dx_to_center / distance_to_center) * pull_strength
                    self.current_dy += (dy_to_center / distance_to_center) * pull_strength

            # Reset timer with some randomness
            self.change_direction_timer = random.randint(30, 90)

        # Move
        self.move(self.current_dx, self.current_dy, environment)


# Individual NPC Classes
class NPC1(NPC):
    def __init__(self):
        super().__init__((0, 0))
        self.dialogue = ["Welcome to our town!"]


class NPC2(NPC):
    def __init__(self):
        super().__init__((0, 1))
        self.dialogue = ["Beautiful weather today!"]


class NPC3(NPC):
    def __init__(self):
        super().__init__((0, 2))
        self.dialogue = ["Need any help?"]


class Alien(NPC):
    def __init__(self):
        super().__init__((0, 3))
        self.dialogue = ["Take me to your leader!"]


# Individual Monster Classes
# Specific Monster Classes with base stats
class Slime(Monster):
    def __init__(self):
        # Lowest base stats
        super().__init__((1, 0),
                         base_speed=0.3,      # Slowest
                         base_health=50,      # Lowest health
                         base_perception=100) # Shortest perception


class Bat(Monster):
    def __init__(self):
        super().__init__((1, 1),
                         base_speed=0.5,      # Slightly faster
                         base_health=75,      # Slightly more health
                         base_perception=150) # Slightly longer perception


class Ghost(Monster):
    def __init__(self):
        super().__init__((1, 2),
                         base_speed=0.7,      # Faster
                         base_health=100,     # More health
                         base_perception=200) # Longer perception


class Spider(Monster):
    def __init__(self):
        # Highest base stats
        super().__init__((1, 3),
                         base_speed=1.0,      # Fastest
                         base_health=125,     # Highest health
                         base_perception=250) # Longest perception