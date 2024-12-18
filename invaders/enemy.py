import pygame
import random
from config import *
from bullet import Bullet
import math


class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, enemy_type='basic'):
        super().__init__()
        self.enemy_type = enemy_type
        self.size_factor = ENEMY_ROWS[enemy_type]['size_factor']
        self.width = int(ENEMY_WIDTH * self.size_factor)
        self.height = int(ENEMY_HEIGHT * self.size_factor)

        # Animation properties
        self.animation_frames = self.create_animation_frames()
        self.current_frame = 0
        self.animation_timer = 0
        self.animation_speed = 100  # milliseconds between frame changes

        self.image = self.animation_frames[0]
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.original_x = x
        self.original_y = y
        self.shoot_chance = ENEMY_ROWS[enemy_type]['shoot_chance']
        self.points = ENEMY_SCORES[enemy_type]
        self.animation_offset = random.random() * math.pi * 2  # Random start point for animations

    def create_animation_frames(self):
        frames = []
        if self.enemy_type == 'basic':
            frames = self.create_basic_animation()
        elif self.enemy_type == 'medium':
            frames = self.create_medium_animation()
        else:  # advanced
            frames = self.create_advanced_animation()
        return frames

    def create_basic_animation(self):
        frames = []
        wing_positions = [0, 2, 4, 2]  # Wing movement pattern

        for wing_pos in wing_positions:
            surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

            # Main body
            pygame.draw.rect(surf, GREEN, (self.width // 4, self.height // 2, self.width // 2, self.height // 2))

            # Triangular top
            pygame.draw.polygon(surf, GREEN, [
                (self.width // 2, 0),
                (self.width // 4, self.height // 2),
                (3 * self.width // 4, self.height // 2)
            ])

            # Animated wings
            pygame.draw.polygon(surf, GREEN, [
                (0, self.height - wing_pos),
                (self.width // 4, self.height - 5),
                (self.width // 4, self.height)
            ])
            pygame.draw.polygon(surf, GREEN, [
                (self.width, self.height - wing_pos),
                (3 * self.width // 4, self.height - 5),
                (3 * self.width // 4, self.height)
            ])

            frames.append(surf)
        return frames

    def create_medium_animation(self):
        frames = []
        wing_angles = [0, 15, 30, 15]  # Wing rotation angles

        for angle in wing_angles:
            surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

            # Main body
            pygame.draw.polygon(surf, GREEN, [
                (self.width // 2, 0),
                (self.width // 4, self.height),
                (3 * self.width // 4, self.height)
            ])

            # Rotating wings
            left_wing = pygame.Surface((self.width // 3, self.height // 3), pygame.SRCALPHA)
            pygame.draw.polygon(left_wing, GREEN, [(0, 0), (self.width // 3, 0), (self.width // 3, self.height // 3)])
            right_wing = pygame.transform.flip(left_wing, True, False)

            left_rotated = pygame.transform.rotate(left_wing, -angle)
            right_rotated = pygame.transform.rotate(right_wing, angle)

            surf.blit(left_rotated, (0, self.height // 2))
            surf.blit(right_rotated, (self.width - right_rotated.get_width(), self.height // 2))

            frames.append(surf)
        return frames

    def create_advanced_animation(self):
        frames = []
        engine_sizes = [2, 4, 6, 4]  # Engine "flame" sizes

        for engine_size in engine_sizes:
            surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

            # Sleek body
            pygame.draw.polygon(surf, GREEN, [
                (self.width // 2, 0),
                (0, self.height - 5),
                (self.width // 4, self.height - 5),
                (self.width // 2, self.height),
                (3 * self.width // 4, self.height - 5),
                (self.width, self.height - 5)
            ])

            # Animated engine "flames"
            flame_points = [
                [(self.width // 3, self.height), (self.width // 3 + engine_size, self.height + engine_size),
                 (self.width // 3 - engine_size, self.height + engine_size)],
                [(2 * self.width // 3, self.height), (2 * self.width // 3 + engine_size, self.height + engine_size),
                 (2 * self.width // 3 - engine_size, self.height + engine_size)]
            ]
            for points in flame_points:
                pygame.draw.polygon(surf, GREEN, points)

            frames.append(surf)
        return frames

    def update(self):
        # Update animation frame
        now = pygame.time.get_ticks()
        if now - self.animation_timer > self.animation_speed:
            self.animation_timer = now
            self.current_frame = (self.current_frame + 1) % len(self.animation_frames)
            self.image = self.animation_frames[self.current_frame]

        # Add subtle floating movement
        time_factor = pygame.time.get_ticks() / 1000  # Convert to seconds
        float_offset = math.sin(time_factor + self.animation_offset) * 2  # Subtle float effect
        self.rect.y = self.original_y + float_offset

    def shoot(self):
        return EnemyBullet(self.rect.centerx, self.rect.bottom)

    def should_shoot(self):
        return random.random() < self.shoot_chance


class EnemyBullet(Bullet):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.image.fill((255, 0, 0))
        self.speed = -BULLET_SPEED


class EnemyFormation:
    def __init__(self, enemies_per_row=ENEMIES_PER_ROW, enemy_speed=0.5, shoot_chance_multiplier=1.0):
        self.enemies = pygame.sprite.Group()
        self.moving_down = False
        self.direction = 1
        self.enemies_per_row = enemies_per_row
        self.speed = enemy_speed
        self.shoot_chance_multiplier = shoot_chance_multiplier
        self.shift_x = 0
        self.total_shift_y = 0
        self.current_move_shift_y = 0

        # Intro animation variables
        self.is_intro_animation = True
        self.intro_enemies = []  # List to store enemies in intro order
        self.current_intro_enemy = 0  # Index of current enemy being positioned
        self.intro_speed = 3  # Speed of intro animation

        self.setup_formation()

    def setup_formation(self):
        intro_x = SCREEN_WIDTH // 2  # All enemies start from center top
        intro_y = -ENEMY_HEIGHT  # Start above the screen

        for enemy_type, properties in ENEMY_ROWS.items():
            row = properties['row']
            for col in range(self.enemies_per_row):
                # Calculate final position
                final_x = ENEMY_FORMATION_START_X + (col * (ENEMY_WIDTH + ENEMY_SPACING))
                final_y = ENEMY_START_Y + (row * ENEMY_ROW_HEIGHT)

                enemy = Enemy(intro_x, intro_y, enemy_type)
                enemy.shoot_chance *= self.shoot_chance_multiplier
                enemy.final_x = final_x  # Store final position
                enemy.final_y = final_y
                self.enemies.add(enemy)
                self.intro_enemies.append(enemy)

    def update(self):
        if not self.enemies:
            return

        if self.is_intro_animation:
            self.intro_animation_update()
        else:
            self.normal_update()

    def intro_animation_update(self):
        if self.current_intro_enemy >= len(self.intro_enemies):
            self.is_intro_animation = False
            return

        current_enemy = self.intro_enemies[self.current_intro_enemy]

        # Move towards final x position
        if current_enemy.rect.x != current_enemy.final_x:
            direction = 1 if current_enemy.final_x > current_enemy.rect.x else -1
            current_enemy.rect.x += direction * self.intro_speed
            if (direction == 1 and current_enemy.rect.x >= current_enemy.final_x) or \
                    (direction == -1 and current_enemy.rect.x <= current_enemy.final_x):
                current_enemy.rect.x = current_enemy.final_x

        # Move towards final y position
        if current_enemy.rect.y != current_enemy.final_y:
            current_enemy.rect.y += self.intro_speed
            if current_enemy.rect.y >= current_enemy.final_y:
                current_enemy.rect.y = current_enemy.final_y

        # If enemy reached its position, move to next enemy
        if current_enemy.rect.x == current_enemy.final_x and current_enemy.rect.y == current_enemy.final_y:
            self.current_intro_enemy += 1

    def normal_update(self):
        if not self.enemies:
            return

        # Check formation edges
        leftmost = min([enemy.rect.left for enemy in self.enemies])
        rightmost = max([enemy.rect.right for enemy in self.enemies])

        # Update individual enemies (for animations)
        for enemy in self.enemies:
            enemy.update()

        # Update horizontal movement
        if not self.moving_down:
            self.shift_x += self.speed * self.direction

            # Check if formation needs to move down
            if (self.direction > 0 and rightmost + self.speed >= SCREEN_WIDTH) or \
                    (self.direction < 0 and leftmost - self.speed <= 0):
                self.moving_down = True
                self.current_move_shift_y = 0
                self.direction *= -1

        # Update vertical movement
        if self.moving_down:
            self.current_move_shift_y += self.speed
            if self.current_move_shift_y >= ENEMY_MOVE_DOWN:
                self.moving_down = False
                self.total_shift_y += ENEMY_MOVE_DOWN
                self.speed *= ENEMY_SPEED_INCREASE_FACTOR

        # Apply movement to all enemies
        current_shift_y = self.total_shift_y + (self.current_move_shift_y if self.moving_down else 0)
        for enemy in self.enemies:
            enemy.rect.x = enemy.original_x + self.shift_x
            enemy.rect.y = enemy.original_y + current_shift_y