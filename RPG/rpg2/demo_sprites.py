import pygame
import sys
import math
import random

# Initialize Pygame
pygame.init()

# Set up the display
WINDOW_SIZE = (800, 600)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Character Movement Patterns")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Load sprite sheet
sprite_sheet = pygame.image.load('assets/town_rpg_pack/graphics/characters/characters.png')


class NonPlayerCharacter:
    def __init__(self, sprite_position):
        self.x = random.randint(100, 700)
        self.y = random.randint(100, 500)
        self.direction = 0
        self.animation_frame = 1
        self.animation_timer = 0
        self.moving = False
        self.speed = 3

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
        sprite.set_colorkey(BLACK)

        sprite.blit(sprite_sheet, (0, 0),
                    (sprite_x, sprite_y, self.sprite_width, self.sprite_height))

        return pygame.transform.scale(sprite, (48, 48))

    def update(self):
        if self.moving:
            self.animation_timer += 1
            if self.animation_timer >= 10:
                self.animation_timer = 0
                self.animation_frame = (self.animation_frame + 1) % 3
        else:
            self.animation_frame = 1

    def move(self, dx, dy):
        if dx != 0 or dy != 0:
            # Normalize diagonal movement
            length = math.sqrt(dx * dx + dy * dy)
            if length != 0:
                dx = dx / length * self.speed
                dy = dy / length * self.speed

            self.x += dx
            self.y += dy
            self.moving = True

            # Set direction based on movement
            if abs(dx) > abs(dy):
                self.direction = 1 if dx < 0 else 2
            else:
                self.direction = 3 if dy < 0 else 0
        else:
            self.moving = False


class NPC(NonPlayerCharacter):
    def __init__(self, sprite_position):
        super().__init__(sprite_position)
        self.dialogue = []
        self.quest = None

        # Square walking pattern
        self.start_x = self.x
        self.start_y = self.y
        self.square_size = 100
        self.corner_pause_time = 0
        self.max_pause_time = 60  # 1 second at 60 FPS
        self.current_corner = 0

    def patrol_square(self):
        if self.corner_pause_time > 0:
            self.corner_pause_time -= 1
            self.moving = False
            return

        # Define corners of the square
        corners = [
            (self.start_x, self.start_y),  # Top-left
            (self.start_x + self.square_size, self.start_y),  # Top-right
            (self.start_x + self.square_size, self.start_y + self.square_size),  # Bottom-right
            (self.start_x, self.start_y + self.square_size)  # Bottom-left
        ]

        target_x, target_y = corners[self.current_corner]

        # Move towards the target corner
        dx = target_x - self.x
        dy = target_y - self.y

        # If reached the corner
        if abs(dx) < 5 and abs(dy) < 5:
            self.x = target_x
            self.y = target_y
            self.corner_pause_time = self.max_pause_time
            self.current_corner = (self.current_corner + 1) % 4
            return

        # Normalize movement
        length = math.sqrt(dx * dx + dy * dy)
        if length != 0:
            dx = dx / length * self.speed
            dy = dy / length * self.speed

        self.x += dx
        self.y += dy
        self.moving = True

        # Set direction
        if abs(dx) > abs(dy):
            self.direction = 1 if dx < 0 else 2
        else:
            self.direction = 3 if dy < 0 else 0


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
        self.change_direction_timer = 0
        self.max_direction_time = 60  # 1 second at 60 FPS
        self.current_dx = 0
        self.current_dy = 0

    def random_movement(self):
        # Change direction periodically
        self.change_direction_timer -= 1
        if self.change_direction_timer <= 0:
            # Stay within movement area
            dx = random.uniform(-1, 1)
            dy = random.uniform(-1, 1)

            # Constrain to movement area
            if (abs(self.x - self.area_center_x) > self.area_size or
                    abs(self.y - self.area_center_y) > self.area_size):
                # Pull back towards center if near edge
                dx += (self.area_center_x - self.x) / self.area_size
                dy += (self.area_center_y - self.y) / self.area_size

            self.current_dx = dx
            self.current_dy = dy

            # Reset timer with some randomness
            self.change_direction_timer = random.randint(30, 90)

        # Move
        self.move(self.current_dx, self.current_dy)


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


# Create characters
npcs = [NPC1(), NPC2(), NPC3(), Alien()]
monsters = [Slime(), Bat(), Ghost(), Spider()]
all_characters = npcs + monsters

# Create character selection buttons
button_height = 50
buttons = []
character_names = ["NPC1", "NPC2", "NPC3", "Alien", "Slime", "Bat", "Ghost", "Spider"]
for i in range(8):
    buttons.append(pygame.Rect(10, 10 + i * (button_height + 5), 100, button_height))

# Game loop
clock = pygame.time.Clock()
current_character = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for i, button in enumerate(buttons):
                if button.collidepoint(mouse_pos):
                    current_character = i

    # Handle movement
    current_char = all_characters[current_character]
    if isinstance(current_char, NPC):
        current_char.patrol_square()
    elif isinstance(current_char, Monster):
        current_char.random_movement()

    # Update animation
    current_char.update()

    # Draw
    screen.fill(WHITE)

    # Draw buttons
    for i, button in enumerate(buttons):
        color = GRAY if i == current_character else BLACK
        pygame.draw.rect(screen, color, button)
        font = pygame.font.Font(None, 24)
        text = font.render(character_names[i], True, WHITE)
        text_rect = text.get_rect(center=button.center)
        screen.blit(text, text_rect)

    # Draw current character
    current_sprite = current_char.get_current_sprite()
    screen.blit(current_sprite,
                (current_char.x - current_sprite.get_width() // 2,
                 current_char.y - current_sprite.get_height() // 2))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()