import pygame
import os
from pygame.locals import *
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
WINDOW_SIZE = (800, 600)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Sword Game Test")

# Load the SVG files
idle_sword = pygame.image.load("sword_idle_animation.svg")
slash_left = pygame.image.load("sword_slash_left.svg")
slash_right = pygame.image.load("sword_slash_right.svg")
charging_sword = pygame.image.load("sword_charging.svg")

# Scale the images
SPRITE_SIZE = (100, 100)
idle_sword = pygame.transform.scale(idle_sword, SPRITE_SIZE)
slash_left = pygame.transform.scale(slash_left, SPRITE_SIZE)
slash_right = pygame.transform.scale(slash_right, SPRITE_SIZE)
charging_sword = pygame.transform.scale(charging_sword, SPRITE_SIZE)

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
DARK_RED = (139, 0, 0)
BLUE = (74, 144, 226)


class ForceWave:
    def __init__(self, x, y, direction, charge_level):
        self.x = x
        self.y = y
        self.direction = direction  # "left" or "right"
        self.charge_level = charge_level
        self.width = 20
        self.height = 80 * (1 + charge_level / 200)  # Height increases with charge, but more modest
        self.distance = 0
        self.max_distance = 400 * (1 + charge_level / 100)  # Distance increases with charge
        self.speed = 12
        self.active = True

    def update(self):
        if not self.active:
            return

        if self.direction == "right":
            self.distance += self.speed
        else:
            self.distance -= self.speed

        if abs(self.distance) > self.max_distance:
            self.active = False

    def get_rect(self):
        if self.direction == "right":
            return pygame.Rect(self.x + self.distance,
                               self.y - self.height / 2,
                               self.width,
                               self.height)
        else:
            return pygame.Rect(self.x + self.distance,
                               self.y - self.height / 2,
                               self.width,
                               self.height)

    def draw(self, screen):
        if not self.active:
            return

        rect = self.get_rect()
        alpha = int(255 * (1 - abs(self.distance) / self.max_distance))
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (*BLUE, alpha), (0, 0, rect.width, rect.height))
        screen.blit(s, rect)


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 5
        self.last_direction = "right"
        self.slashing = False
        self.slash_timer = 0
        self.slash_duration = 20
        self.charging = False
        self.charge_level = 0
        self.max_charge = 100
        self.charge_rate = 2
        self.force_waves = []
        self.quick_slash_threshold = 300  # milliseconds
        self.space_pressed_time = 0

    def move(self, keys):
        if not self.charging:  # Only restrict movement during charging
            if keys[K_LEFT]:
                self.x -= self.speed
                self.last_direction = "left"
            if keys[K_RIGHT]:
                self.x += self.speed
                self.last_direction = "right"
            if keys[K_UP]:
                self.y -= self.speed
            if keys[K_DOWN]:
                self.y += self.speed

            # Keep player in bounds
            self.x = max(0, min(self.x, WINDOW_SIZE[0] - SPRITE_SIZE[0]))
            self.y = max(0, min(self.y, WINDOW_SIZE[1] - SPRITE_SIZE[1]))

    def start_charging(self):
        self.space_pressed_time = pygame.time.get_ticks()
        # Don't start charging immediately - wait to see if it's a quick slash
        pygame.time.set_timer(pygame.USEREVENT, self.quick_slash_threshold)

    def begin_charge(self):
        self.charging = True
        self.charge_level = 0

    def release_charge(self):
        current_time = pygame.time.get_ticks()
        pygame.time.set_timer(pygame.USEREVENT, 0)  # Cancel the timer

        if not self.charging:
            self.quick_slash()
        elif self.charge_level > 20:  # Minimum charge threshold
            self.charged_slash()

        self.charging = False
        self.charge_level = 0

    def quick_slash(self):
        self.slashing = True
        self.slash_timer = self.slash_duration

    def charged_slash(self):
        self.slashing = True
        self.slash_timer = self.slash_duration
        wave = ForceWave(self.x + SPRITE_SIZE[0] / 2,
                         self.y + SPRITE_SIZE[1] / 2,
                         self.last_direction,
                         self.charge_level)
        self.force_waves.append(wave)

    def update(self):
        if self.charging:
            self.charge_level = min(self.charge_level + self.charge_rate, self.max_charge)

        if self.slashing:
            self.slash_timer -= 1
            if self.slash_timer <= 0:
                self.slashing = False

        # Update force waves
        for wave in self.force_waves:
            wave.update()
        # Remove inactive waves
        self.force_waves = [wave for wave in self.force_waves if wave.active]

    def draw(self, screen):
        # Draw the charge bar first if charging
        if self.charging:
            self.draw_charge_bar(screen)

        # Draw the sword sprite
        if self.charging:
            screen.blit(charging_sword, (self.x, self.y))
        elif self.slashing:
            if self.last_direction == "left":
                screen.blit(slash_left, (self.x, self.y))
            else:
                screen.blit(slash_right, (self.x, self.y))
        else:
            # Add bobbing animation when idle
            bob_offset = abs(pygame.time.get_ticks() % 1000 - 500) / 50
            screen.blit(idle_sword, (self.x, self.y - bob_offset))

    def draw_charge_bar(self, screen):
        if self.charging and self.charge_level > 0:
            bar_width = 60
            bar_height = 8
            x = self.x + (SPRITE_SIZE[0] - bar_width) // 2
            y = self.y - 20

            # Background
            pygame.draw.rect(screen, (50, 50, 50),
                             (x, y, bar_width, bar_height))
            # Charge level
            charge_width = int(bar_width * (self.charge_level / self.max_charge))
            pygame.draw.rect(screen, (74, 144, 226),
                             (x, y, charge_width, bar_height))


class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 40
        self.health = 100
        self.rect = pygame.Rect(x, y, self.size, self.size)

    def take_damage(self, amount):
        self.health -= amount

    def is_alive(self):
        return self.health > 0

    def draw(self, screen):
        health_percentage = self.health / 100
        color = (
            int(DARK_RED[0] + (RED[0] - DARK_RED[0]) * health_percentage),
            int(DARK_RED[1] + (RED[1] - DARK_RED[1]) * health_percentage),
            int(DARK_RED[2] + (RED[2] - DARK_RED[2]) * health_percentage)
        )
        pygame.draw.rect(screen, color, self.rect)


# Game setup
player = Player(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
targets = [Target(random.randint(0, WINDOW_SIZE[0] - 40),
                  random.randint(0, WINDOW_SIZE[1] - 40)) for _ in range(5)]

# Game loop
running = True
clock = pygame.time.Clock()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_SPACE:
                player.start_charging()
        elif event.type == KEYUP:
            if event.key == K_SPACE:
                player.release_charge()
        elif event.type == pygame.USEREVENT:
            # Timer hit - start charging if space is still held
            if pygame.key.get_pressed()[K_SPACE]:
                player.begin_charge()
            pygame.time.set_timer(pygame.USEREVENT, 0)  # Cancel the timer

    # Update
    keys = pygame.key.get_pressed()
    player.move(keys)
    player.update()

    # Handle combat
    for wave in player.force_waves:
        if wave.active:
            wave_rect = wave.get_rect()
            for target in targets:
                if target.rect.colliderect(wave_rect):
                    # Damage based on charge level and distance
                    base_damage = 20  # Increased base damage for force waves
                    damage = (base_damage + player.charge_level / 2) * (1 - abs(wave.distance) / wave.max_distance)
                    target.take_damage(damage)
                    # Only damage once per wave-target collision
                    wave.active = False

    if player.slashing:
        slash_rect = pygame.Rect(player.x, player.y, SPRITE_SIZE[0], SPRITE_SIZE[1])
        for target in targets:
            if target.rect.colliderect(slash_rect):
                target.take_damage(10)  # Fixed damage for quick slash

    # Remove dead targets
    targets = [target for target in targets if target.is_alive()]

    # Render
    screen.fill(WHITE)

    # Draw force waves
    for wave in player.force_waves:
        wave.draw(screen)

    # Draw targets
    for target in targets:
        target.draw(screen)

    # Draw player (sword)
    player.draw(screen)

    # Update display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()