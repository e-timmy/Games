import pygame
import math
import random
from constants import CHARACTER_SCALE_FACTOR

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
    def __init__(self, sprite_position, speed):
        super().__init__(sprite_position)
        self.health = 100
        self.damage = 10
        self.speed = speed

        # Movement area
        self.area_center_x = self.x
        self.area_center_y = self.y
        self.area_size = 150

        # Random movement
        self.change_direction_timer = random.randint(30, 90)  # Random initial timer
        self.max_direction_time = 60  # 1 second at 60 FPS

        # Initialize with zero movement
        self.current_dx = 0
        self.current_dy = 0

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

            # Constrain to movement area
            if (abs(self.x - self.area_center_x) > self.area_size or
                    abs(self.y - self.area_center_y) > self.area_size):
                # Pull back towards center if near edge
                self.current_dx += (self.area_center_x - self.x) / self.area_size
                self.current_dy += (self.area_center_y - self.y) / self.area_size

            # Reset timer with some randomness
            self.change_direction_timer = random.randint(30, 90)

        # Move
        self.move(self.current_dx, self.current_dy, environment)

        # Update the movement area center
        self.area_center_x = self.x
        self.area_center_y = self.y


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
class Slime(Monster):
    def __init__(self):
        super().__init__((1, 0), speed=0.5)
        self.health = 50
        self.damage = 5


class Bat(Monster):
    def __init__(self):
        super().__init__((1, 1), speed=2)
        self.health = 30
        self.damage = 8


class Ghost(Monster):
    def __init__(self):
        super().__init__((1, 2), speed=1)
        self.health = 40
        self.damage = 15


class Spider(Monster):
    def __init__(self):
        super().__init__((1, 3), speed=1.5)
        self.health = 35
        self.damage = 12