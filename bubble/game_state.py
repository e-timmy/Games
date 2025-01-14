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
        self.bullets = []
        self.splitting_bubbles = []
        self.total_bubbles_spawned = 0
        self.spawn_initial_bubbles()
        self.game_over = False
        self.last_update_time = pygame.time.get_ticks()

    def handle_player_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player.rotate(-PLAYER_ROTATION_SPEED)
        if keys[pygame.K_RIGHT]:
            self.player.rotate(PLAYER_ROTATION_SPEED)
        if keys[pygame.K_UP]:
            new_bubbles = self.player.thrust()
            if new_bubbles:
                # Add propulsion bubbles directly to main bubble list
                self.bubbles.extend(new_bubbles)
                self.total_bubbles_spawned += len(new_bubbles)  # Update total bubbles count
        if keys[pygame.K_SPACE]:
            bullet = self.player.shoot()
            if bullet:
                self.bullets.append(bullet)

    def update(self):
        current_time = pygame.time.get_ticks()
        dt = (current_time - self.last_update_time) / 1000.0
        self.last_update_time = current_time

        if not self.game_over:
            self.handle_player_input()
            self.player.update()

            # Update all bubbles
            for bubble in self.bubbles:
                bubble.update()
            for bullet in self.bullets:
                bullet.update()

            self.handle_bubble_splitting()
            self.update_splitting_bubbles(dt)
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
        for bullet in self.bullets:
            bullet.draw(screen)
        for splitting_bubble in self.splitting_bubbles:
            splitting_bubble.draw(screen)
        self.player.draw(screen)

        if self.game_over:
            font = pygame.font.Font(None, 74)
            text = font.render('Game Over!', True, (255, 0, 0))
            text_rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            screen.blit(text, text_rect)

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

    def handle_bubble_splitting(self):
        # Check bullet collisions with bubbles
        for bullet in self.bullets[:]:  # Use slice copy to safely modify list while iterating
            if not bullet.has_split_bubble:
                for bubble in self.bubbles[:]:  # Same here
                    if bullet.collides_with(bubble):
                        # Split the bubble
                        new_splitting_bubbles = bullet.split_bubble(bubble)
                        if new_splitting_bubbles:
                            self.bubbles.remove(bubble)
                            self.splitting_bubbles.extend(new_splitting_bubbles)
                            # Note: has_split_bubble is now set inside split_bubble() on success
                            break
                        else:
                            # If split failed, convert bullet to regular bubble
                            bullet.convert_to_regular_bubble()

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
        # Move split bullets to regular bubbles list
        for bullet in self.bullets[:]:
            if bullet.has_split_bubble:
                self.bubbles.append(bullet)
                self.bullets.remove(bullet)

    def update_splitting_bubbles(self, dt):
        for splitting_bubble in self.splitting_bubbles[:]:
            splitting_bubble.update(dt)
            if splitting_bubble.is_animation_complete():
                self.bubbles.append(splitting_bubble.to_bubble())
                self.splitting_bubbles.remove(splitting_bubble)