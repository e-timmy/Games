import random
import math
import pygame

from player import Player
from bubble import Bubble
from settings import *


class GameState:
    def __init__(self):
        self.player = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, PLAYER_START_SIZE)
        self.bubbles = []
        self.total_bubbles_spawned = 0
        self.spawn_initial_bubbles()
        self.game_over = False

    def spawn_initial_bubbles(self):
        attempts = 0
        while len(self.bubbles) < INITIAL_BUBBLE_COUNT and attempts < 100:
            if self.spawn_bubble():
                self.total_bubbles_spawned += 1
            attempts += 1

    def spawn_bubble(self):
        if self.total_bubbles_spawned >= MAX_TOTAL_BUBBLES:
            return False

        size = random.randint(MIN_BUBBLE_SIZE, MAX_BUBBLE_SIZE)
        x = random.randint(size, SCREEN_WIDTH - size)
        y = random.randint(size, SCREEN_HEIGHT - size)

        if any(math.hypot(b.x - x, b.y - y) < MIN_BUBBLE_DISTANCE for b in self.bubbles):
            return False

        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(MIN_BUBBLE_SPEED, MAX_BUBBLE_SPEED)
        dx = speed * math.cos(angle)
        dy = speed * math.sin(angle)

        self.bubbles.append(Bubble(x, y, size, dx, dy))
        return True

    def handle_player_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player.rotate(-PLAYER_ROTATION_SPEED)
        if keys[pygame.K_RIGHT]:
            self.player.rotate(PLAYER_ROTATION_SPEED)
        if keys[pygame.K_UP]:
            self.player.thrust()

    def handle_bubble_interactions(self):
        # Check player against other bubbles
        for bubble in self.bubbles:
            if self.player.collides_with(bubble):
                if self.player.size > bubble.size:
                    self.player.absorb(bubble)
                else:
                    bubble.absorb(self.player)
                    if self.player.size <= MIN_BUBBLE_SIZE:
                        self.game_over = True
                        return

        # Check all bubble pairs for collision
        for i, bubble1 in enumerate(self.bubbles):
            for bubble2 in self.bubbles[i + 1:]:
                if bubble1.collides_with(bubble2):
                    if bubble1.size > bubble2.size:
                        bubble1.absorb(bubble2)
                    else:
                        bubble2.absorb(bubble1)

    def remove_absorbed_bubbles(self):
        self.bubbles = [b for b in self.bubbles if b.size > MIN_BUBBLE_SIZE]

    def update(self):
        if not self.game_over:
            self.handle_player_input()
            self.player.update()
            for bubble in self.bubbles:
                bubble.update()

            self.handle_bubble_interactions()
            self.remove_absorbed_bubbles()

            # Spawn new bubbles if needed
            if (len(self.bubbles) < INITIAL_BUBBLE_COUNT and
                    self.total_bubbles_spawned < MAX_TOTAL_BUBBLES):
                if self.spawn_bubble():
                    self.total_bubbles_spawned += 1

    def draw(self, screen):
        for bubble in self.bubbles:
            bubble.draw(screen)
        self.player.draw(screen)

        if self.game_over:
            font = pygame.font.Font(None, 74)
            text = font.render('Game Over!', True, (255, 0, 0))
            text_rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            screen.blit(text, text_rect)