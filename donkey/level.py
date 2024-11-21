import pygame
import pymunk
import random

from game_objects import Platform, Ball, create_ball


class Level:
    def __init__(self, number):
        self.number = number
        self.platforms = []
        self.balls = []
        self.ball_spawn_timer = 0
        self.finish_zone = None
        self.completed = False

        # Level scaling factors
        base_platforms = 4
        additional_platforms = (number - 1) * 2
        self.num_platforms = min(base_platforms + additional_platforms, 15)

        # Adjust platform spacing based on number of platforms
        self.total_height = 500  # Total playable height
        self.platform_spacing = self.total_height / (self.num_platforms + 1)

        # Player and game mechanics configuration
        self.player_size = 20
        # Make jump height 2.5x the platform spacing to ensure reachability
        self.jump_height = self.platform_spacing * 2.5

        # Ball configurations
        self.ball_config = {
            "interval": max(180 - (number - 1) * 20, 60),
            "speed": 3 + (number - 1),
            "max_balls": 1 if number == 1 else min(2 + number, 8),
            "size": 10
        }

    def generate_platforms(self, space):
        thickness = 16
        screen_width = 800
        screen_height = 600

        self.platforms = []

        # First platform (ground) positioning
        first_platform_y = screen_height - self.platform_spacing

        # Create first platform (ground)
        ground = Platform(space, (0, first_platform_y), (screen_width, first_platform_y), thickness)
        self.platforms.append(ground)

        # Create walls
        left_wall = Platform(space, (-20, 0), (-20, screen_height), thickness)
        right_wall = Platform(space, (screen_width + 20, 100), (screen_width + 20, screen_height), thickness)
        self.platforms.extend([left_wall, right_wall])

        # Generate platforms with smaller slope for better reachability
        last_y = first_platform_y

        for i in range(self.num_platforms):
            y = last_y - self.platform_spacing

            if i % 2 == 0:
                # Platform sloping down from left to right (reduced slope)
                start_x = -20
                end_x = random.randint(500, 700)
                start_y = y
                end_y = y + 10  # Reduced slope
            else:
                # Platform sloping down from right to left (reduced slope)
                start_x = screen_width + 20
                end_x = random.randint(100, 300)
                start_y = y
                end_y = y + 10  # Reduced slope

            platform = Platform(space, (start_x, start_y), (end_x, end_y), thickness)
            self.platforms.append(platform)
            last_y = y

        # Create finish zone above the highest platform
        highest_y = min([p.p1[1] for p in self.platforms[3:]], default=100)
        if self.platforms[-1].p1[0] < 400:
            self.finish_zone = pygame.Rect(750, highest_y - 50, 70, 30)
        else:
            self.finish_zone = pygame.Rect(0, highest_y - 50, 70, 30)

    def update(self, space):
        self.ball_spawn_timer += 1
        if self.ball_spawn_timer >= self.ball_config["interval"] and len(self.balls) < self.ball_config["max_balls"]:
            highest_y = min([p.p1[1] for p in self.platforms[3:]], default=100)

            if self.finish_zone.x > 400:
                spawn_pos = (850, highest_y + 20)
                velocity = (-self.ball_config["speed"] * 200, 0)
            else:
                spawn_pos = (-50, highest_y + 20)
                velocity = (self.ball_config["speed"] * 200, 0)

            new_ball = create_ball(space, self.ball_config["size"])
            new_ball.body.position = spawn_pos
            new_ball.body.velocity = velocity

            self.balls.append(new_ball)
            self.ball_spawn_timer = 0

        self.balls = [ball for ball in self.balls if
                      -100 < ball.body.position.x < 900 and
                      ball.body.position.y < 650]

    def draw(self, screen):
        for platform in self.platforms:
            platform.draw(screen)

        for ball in self.balls:
            ball.draw(screen)

        pygame.draw.rect(screen, (0, 255, 0), self.finish_zone)

    def get_jump_height(self):
        return self.jump_height

    def check_finish(self, player):
        player_rect = pygame.Rect(
            player.body.position.x - player.width / 2,
            player.body.position.y - player.height / 2,
            player.width,
            player.height
        )
        return self.finish_zone.colliderect(player_rect)