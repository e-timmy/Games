import pygame
import random
from enum import Enum
from game_objects import Player3D, Projectile3D
from math_3d import Vector3, Matrix4x4, project_point
from math import pi
import numpy as np


class GameState(Enum):
    START_SCREEN = 1
    PLAYING = 2
    GAME_OVER = 3


class GameManager3D:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.state = GameState.START_SCREEN
        self.score = 0
        self.high_score = 0
        self.font = pygame.font.Font(None, 64)
        self.small_font = pygame.font.Font(None, 32)
        self.difficulty_level = 1
        self.score_for_next_level = 10  # Score needed for first difficulty increase
        self.projectile_colors = [
            (255, 0, 0),  # Regular red
            (255, 165, 0),  # Orange
            (255, 255, 0),  # Yellow
            (0, 255, 0),  # Green
            (0, 0, 255)  # Blue
        ]
        self.reset_game()

        self.projection_matrix = Matrix4x4.perspective(
            fov=pi / 3,
            aspect=width / height,
            near=0.1,
            far=100.0
        )

    def reset_game(self):
        self.player = Player3D(0, 0)
        self.projectiles = []
        self.splatter_particles = []
        self.spawn_timer = 0
        self.base_spawn_delay = 45
        self.spawn_delay = self.base_spawn_delay
        self.grid_points = self._create_grid()
        self.grid_scroll = 0
        self.score = 0
        self.difficulty_level = 1
        self.score_for_next_level = 10

    def _create_grid(self):
        grid_points = []
        for x in range(-5, 6, 1):
            for y in range(-5, 6, 1):
                grid_points.append(Vector3(x/2.5, y/2.5, -20))
        return grid_points

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if self.state == GameState.START_SCREEN:
                    self.state = GameState.PLAYING
                elif self.state == GameState.GAME_OVER:
                    self.reset_game()
                    self.state = GameState.PLAYING

    def update_difficulty(self):
        old_level = self.difficulty_level
        self.difficulty_level = 1 + (self.score // 10)  # Increase difficulty every 10 points

        if self.difficulty_level != old_level:
            # Update game parameters based on new difficulty
            self.spawn_delay = max(15, self.base_spawn_delay - (self.difficulty_level * 3))
            print(f"Difficulty increased to level {self.difficulty_level}! Speed increased!")

    def spawn_projectile(self):
        # Difficulty-based spawn patterns
        if self.difficulty_level >= 3 and random.random() < 0.3:
            # Spawn cluster of projectiles
            self._spawn_projectile_cluster()
        else:
            # Regular single projectile spawn
            self._spawn_single_projectile()

    def _spawn_projectile_cluster(self):
        # Spawn 3 projectiles in a triangle formation
        center_x = random.uniform(-0.2, 0.2)
        center_y = random.uniform(-0.2, 0.2)

        for i in range(3):
            angle = (2 * pi * i / 3) + random.uniform(-0.2, 0.2)
            offset = 0.1  # Distance from center of formation

            x = center_x + np.cos(angle) * offset
            y = center_y + np.sin(angle) * offset
            z = -15

            target_x = 0
            target_y = 0
            target_z = self.player.position.z

            dx = target_x - x
            dy = target_y - y
            dz = target_z - z

            magnitude = (dx * dx + dy * dy + dz * dz) ** 0.5
            base_speed = 8 + (self.difficulty_level * 0.5)  # Speed increases with difficulty
            speed = base_speed * random.uniform(0.9, 1.1)  # Slight speed variation

            velocity = Vector3(
                (dx / magnitude) * speed,
                (dy / magnitude) * speed,
                (dz / magnitude) * speed
            )

            # Use different colors based on difficulty
            color_idx = min(self.difficulty_level - 1, len(self.projectile_colors) - 1)
            projectile = Projectile3D(Vector3(x, y, z), velocity, self.projectile_colors[color_idx])
            self.projectiles.append(projectile)

    def _spawn_single_projectile(self):
        x = random.uniform(-0.2, 0.2)
        y = random.uniform(-0.2, 0.2)
        z = -15

        target_x = 0
        target_y = 0
        target_z = self.player.position.z

        dx = target_x - x
        dy = target_y - y
        dz = target_z - z

        magnitude = (dx * dx + dy * dy + dz * dz) ** 0.5
        base_speed = 8 + (self.difficulty_level * 0.5)
        speed = base_speed * random.uniform(0.9, 1.1)

        velocity = Vector3(
            (dx / magnitude) * speed,
            (dy / magnitude) * speed,
            (dz / magnitude) * speed
        )

        color_idx = min(self.difficulty_level - 1, len(self.projectile_colors) - 1)
        projectile = Projectile3D(Vector3(x, y, z), velocity, self.projectile_colors[color_idx])
        self.projectiles.append(projectile)

    def handle_input(self, keys, delta_time):
        if self.state != GameState.PLAYING:
            return

        move_speed = 2.0 * delta_time
        if keys[pygame.K_LEFT]:
            self.player.position.x += move_speed  # Fixed direction
        if keys[pygame.K_RIGHT]:
            self.player.position.x -= move_speed  # Fixed direction
        if keys[pygame.K_UP]:
            self.player.position.y += move_speed  # Fixed direction
        if keys[pygame.K_DOWN]:
            self.player.position.y -= move_speed  # Fixed direction

        self.player.position.x = max(-1, min(1, self.player.position.x))
        self.player.position.y = max(-1, min(1, self.player.position.y))

    def update(self, delta_time):
        if self.state != GameState.PLAYING:
            return

        self.update_difficulty()

        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_delay:
            self.spawn_timer = 0
            self.spawn_projectile()
            self.score += 1

        self.grid_scroll += delta_time * 5
        if self.grid_scroll >= 20:
            self.grid_scroll = 0

        for particle in self.splatter_particles[:]:
            particle.update(delta_time)
            if particle.lifetime <= 0:
                self.splatter_particles.remove(particle)

        for projectile in self.projectiles[:]:
            projectile.update(delta_time)

            if self.check_collision(projectile):
                self.high_score = max(self.score, self.high_score)
                self.state = GameState.GAME_OVER
                return

            if projectile.has_splattered:
                screen_pos = project_point(Vector3(projectile.position.x,
                                                   projectile.position.y,
                                                   0),
                                           self.projection_matrix)
                screen_x = int((screen_pos.x + 1) * self.width / 2)
                screen_y = int((screen_pos.y + 1) * self.height / 2)
                new_particles = projectile.create_splatter_particles(screen_x, screen_y)
                self.splatter_particles.extend(new_particles)
                self.projectiles.remove(projectile)
            elif not projectile.active:
                self.projectiles.remove(projectile)

    def draw(self, screen):
        screen.fill((20, 20, 40))

        if self.state == GameState.START_SCREEN:
            title = self.font.render("3D GRAVITY DODGE", True, (255, 255, 255))
            start_text = self.small_font.render("Press SPACE to Start", True, (255, 255, 255))
            controls = self.small_font.render("Use Arrow Keys to Move", True, (255, 255, 255))
            screen.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 3))
            screen.blit(start_text, (self.width // 2 - start_text.get_width() // 2, self.height // 2))
            screen.blit(controls, (self.width // 2 - controls.get_width() // 2, self.height // 2 + 40))
            return

        if self.state == GameState.GAME_OVER:
            game_over = self.font.render("GAME OVER", True, (255, 0, 0))
            score_text = self.small_font.render(f"Score: {self.score}", True, (255, 255, 255))
            high_score_text = self.small_font.render(f"High Score: {self.high_score}", True, (255, 255, 255))
            level_text = self.small_font.render(f"Reached Level: {self.difficulty_level}", True, (255, 255, 255))
            restart_text = self.small_font.render("Press SPACE to Restart", True, (255, 255, 255))

            screen.blit(game_over, (self.width // 2 - game_over.get_width() // 2, self.height // 3))
            screen.blit(score_text, (self.width // 2 - score_text.get_width() // 2, self.height // 2))
            screen.blit(high_score_text, (self.width // 2 - high_score_text.get_width() // 2, self.height // 2 + 40))
            screen.blit(level_text, (self.width // 2 - level_text.get_width() // 2, self.height // 2 + 80))
            screen.blit(restart_text, (self.width // 2 - restart_text.get_width() // 2, self.height // 2 + 120))
            return

        # Draw game elements
        for point in self.grid_points:
            grid_pos = Vector3(point.x, point.y, (point.z + self.grid_scroll) % 20 - 20)
            screen_pos = project_point(grid_pos, self.projection_matrix)
            screen_x = int((screen_pos.x + 1) * self.width / 2)
            screen_y = int((screen_pos.y + 1) * self.height / 2)

            size = max(1, int(8 / (-grid_pos.z + 2)))
            brightness = max(20, min(150, int(255 / (-grid_pos.z + 15))))
            pygame.draw.circle(screen, (brightness, brightness, brightness),
                               (screen_x, screen_y), size)

        for projectile in self.projectiles:
            projectile.draw(screen, self.projection_matrix)

        for particle in self.splatter_particles:
            particle.draw(screen)

        self.player.draw(screen, self.projection_matrix)

        # Draw score and level
        score_text = self.small_font.render(f"Score: {self.score}", True, (255, 255, 255))
        level_text = self.small_font.render(f"Level: {self.difficulty_level}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        screen.blit(level_text, (10, 40))