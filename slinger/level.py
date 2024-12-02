import pygame
import pymunk
from constants import SCREEN_WIDTH, SCREEN_HEIGHT
from game_objects import Platform, Rope, Bullet
from player import Player


class LevelManager:
    def __init__(self, space, game_state):
        self.space = space
        self.game_state = game_state
        self.player = None
        self.platforms = []
        self.ropes = []
        self.bullets = []
        self.bullets_used = 0

        # Base dimensions (Level 1)
        self.base_width = SCREEN_WIDTH
        self.base_height = SCREEN_HEIGHT

        # Current level dimensions
        self.current_width = self.base_width
        self.current_height = self.base_height
        self.scale_factor = 1.0

        # View/Camera properties
        self.camera_scale = 1.0

        self.PLATFORM_WIDTH_RATIO = 0.3  # 30% of screen width
        self.PLAYER_SIZE_RATIO = 0.025  # 2.5% of screen width

    def setup_level(self):
        self.clean_level()

        # Update dimensions and scaling for new level
        self.scale_factor = 1.0 + (self.game_state.current_level - 1) * 0.5
        self.current_width = self.base_width * self.scale_factor
        self.current_height = self.base_height * self.scale_factor
        self.camera_scale = self.base_width / self.current_width

        # Update space gravity
        self.space.gravity = (0, 900 * self.scale_factor)

        # Calculate platform and gap widths
        platform_count = self.game_state.current_level + 1
        base_platform_width = self.base_width * self.PLATFORM_WIDTH_RATIO
        platform_width = base_platform_width * self.scale_factor

        total_platform_width = platform_width * platform_count
        total_gap_width = self.current_width - total_platform_width
        gap_width = total_gap_width / (platform_count - 1) if platform_count > 1 else 0

        # Create player with proportional size
        player_size = self.base_width * self.PLAYER_SIZE_RATIO * self.scale_factor
        player_start_x = 50 * self.scale_factor
        player_start_y = self.current_height - (100 * self.scale_factor)
        self.player = Player(player_start_x, player_start_y, player_size, self.scale_factor)
        self.space.add(self.player.body, self.player.shape)

        # Create platforms
        for i in range(platform_count):
            x = i * (platform_width + gap_width)
            y = self.current_height - (50 * self.scale_factor)
            platform = Platform(x, y, platform_width, 50 * self.scale_factor)
            self.platforms.append(platform)
            self.space.add(platform.body, platform.shape)

        # Add ceiling
        ceiling = Platform(0, 0, self.current_width, 10 * self.scale_factor)
        self.platforms.append(ceiling)
        self.space.add(ceiling.body, ceiling.shape)

    def clean_level(self):
        for body in list(self.space.bodies):
            self.space.remove(body)
        for shape in list(self.space.shapes):
            self.space.remove(shape)
        for constraint in list(self.space.constraints):
            self.space.remove(constraint)

        self.platforms.clear()
        self.ropes.clear()
        self.bullets.clear()
        self.bullets_used = 0
        self.player = None

    def world_to_screen(self, x, y):
        return (int(x * self.camera_scale), int(y * self.camera_scale))

    def screen_to_world(self, x, y):
        return (x / self.camera_scale, y / self.camera_scale)

    def handle_bullet_wall_collision(self, arbiter, space, data):
        bullet_shape = arbiter.shapes[0]
        wall_shape = arbiter.shapes[1]

        if wall_shape.body.position.y <= 10 * self.scale_factor:
            for bullet in self.bullets:
                if bullet.shape == bullet_shape:
                    bullet.explode()
                    new_rope = Rope(self.space, bullet.position,
                                    self.current_height - 50 * self.scale_factor,
                                    self.scale_factor, unfurling=True)
                    self.ropes.append(new_rope)
                    self.space.remove(bullet.body, bullet.shape)
                    self.bullets.remove(bullet)
                    return False
        return True

    def shoot_bullet(self, start_pos, target_pos):
        if self.bullets_used < self.game_state.bullets_per_level:
            new_bullet = Bullet(start_pos, target_pos, self.space, self.scale_factor)
            self.bullets.append(new_bullet)
            self.bullets_used += 1

    def update(self, dt):
        if self.player:
            self.player.update(dt)

            if self.player.body.position.y > self.current_height:
                self.game_state.set_game_over()
            elif self.player.body.position.x > self.current_width - (50 * self.scale_factor):
                self.game_state.set_level_complete()

        for bullet in self.bullets[:]:
            if bullet.update(dt, self.current_height):
                self.space.remove(bullet.body, bullet.shape)
                self.bullets.remove(bullet)

        for rope in self.ropes:
            rope.update(dt)

    def draw(self, screen):
        for platform in self.platforms:
            vertices = [self.world_to_screen(v.x + platform.body.position.x,
                                             v.y + platform.body.position.y)
                        for v in platform.shape.get_vertices()]
            pygame.draw.polygon(screen, (100, 100, 100), vertices)

        for rope in self.ropes:
            rope.draw(screen, self.camera_scale)

        for bullet in self.bullets:
            bullet.draw(screen, self.camera_scale)

        if self.player:
            vertices = [self.world_to_screen(v.x + self.player.body.position.x,
                                             v.y + self.player.body.position.y)
                        for v in self.player.shape.get_vertices()]
            pygame.draw.polygon(screen, (0, 128, 255), vertices)

        font = pygame.font.Font(None, 36)
        bullet_text = font.render(f"Bullets: {self.game_state.bullets_per_level - self.bullets_used}", True, (0, 0, 0))
        level_text = font.render(f"Level: {self.game_state.current_level}", True, (0, 0, 0))
        screen.blit(bullet_text, (10, 50))
        screen.blit(level_text, (10, 10))