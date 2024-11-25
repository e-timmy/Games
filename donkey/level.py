import pygame
import pymunk
import random

from game_objects import Platform, Ball, create_ball, Ladder
from settings import WINDOW_WIDTH


class Level:
    def __init__(self, number):
        self.number = number
        self.platforms = []
        self.ladders = []
        self.balls = []
        self.ball_spawn_timer = 0
        self.finish_zone = None
        self.completed = False

        base_platforms = 4
        additional_platforms = (number - 1) * 2
        self.num_platforms = min(base_platforms + additional_platforms, 15)

        self.total_height = 500
        self.platform_spacing = self.total_height / (self.num_platforms + 1)

        self.player_size = 20
        ball_size = 15
        self.jump_height = ball_size * 2.5

        self.ball_config = {
            "interval": max(180 - (number - 1) * 20, 60),
            "speed": 3 + (number - 1),
            "max_balls": 1 if number == 1 else min(2 + number, 8),
            "size": ball_size
        }

    def generate_platforms(self, space):
        thickness = 20
        screen_width = 800
        screen_height = 600

        self.platforms = []
        self.ladders = []

        first_platform_y = screen_height - self.platform_spacing
        ground = Platform(space, (0, first_platform_y), (screen_width, first_platform_y), thickness)
        self.platforms.append(ground)

        left_wall = Platform(space, (-20, 0), (-20, screen_height), thickness)
        right_wall = Platform(space, (screen_width + 20, 100), (screen_width + 20, screen_height), thickness)
        self.platforms.extend([left_wall, right_wall])

        last_y = first_platform_y

        for i in range(self.num_platforms):
            y = last_y - self.platform_spacing

            platform_height = 40  # Height difference for slopes
            platform_min_length = 300  # Minimum platform length

            platform_height = 25  # Reduced height difference for gentler slopes

            if i % 2 == 0:  # Even platforms slope down from left to right
                start_x = 0  # Start at left edge
                end_x = screen_width - 100  # Extend almost to right edge
                start_y = y - platform_height  # Start higher
                end_y = y + platform_height  # End lower

                # Always add at least one ladder
                ladder_x = random.randint(end_x - 200, end_x - 100)
                if i == 0:
                    # Connect ground to first platform, extend ladder 20px above platform
                    self.ladders.append(Ladder(ladder_x, end_y - 20, first_platform_y))
                else:
                    # Connect to previous platform, extend ladder 20px above platform
                    self.ladders.append(Ladder(ladder_x, end_y - 20, last_y))

                # 30% chance for an additional ladder
                if random.random() < 0.3:
                    extra_ladder_x = random.randint(end_x - 400, end_x - 250)
                    if i == 0:
                        self.ladders.append(Ladder(extra_ladder_x, end_y - 20, first_platform_y))
                    else:
                        self.ladders.append(Ladder(extra_ladder_x, end_y - 20, last_y))

            else:  # Odd platforms slope up from left to right
                start_x = 100  # Start slightly in from left
                end_x = screen_width  # End at right edge
                start_y = y + platform_height  # Start lower
                end_y = y - platform_height  # End higher

                # Always add at least one ladder
                ladder_x = random.randint(start_x + 100, start_x + 200)
                self.ladders.append(Ladder(ladder_x, start_y - 20, last_y))

                # 30% chance for an additional ladder
                if random.random() < 0.3:
                    extra_ladder_x = random.randint(start_x + 250, start_x + 400)
                    self.ladders.append(Ladder(extra_ladder_x, start_y - 20, last_y))

            platform = Platform(space, (start_x, start_y), (end_x, end_y), thickness)
            self.platforms.append(platform)

            last_y = y

        # Place finish zone at top right of highest platform
        highest_platform = self.platforms[-1]
        if len(self.platforms) % 2 == 0:  # Even number of platforms
            finish_x = highest_platform.p2[0] - 70  # Place near right end
            finish_y = highest_platform.p2[1] - 30
        else:  # Odd number of platforms
            finish_x = screen_width - 70  # Place at right edge
            finish_y = highest_platform.p2[1] - 30

        self.finish_zone = pygame.Rect(finish_x, finish_y, 70, 30)

    def update(self, space):
        self.ball_spawn_timer += 1
        if self.ball_spawn_timer >= self.ball_config["interval"] and len(self.balls) < self.ball_config["max_balls"]:
            highest_platform = None
            highest_y = float('inf')
            for platform in self.platforms[3:]:
                platform_y = min(platform.p1[1], platform.p2[1])
                if platform_y < highest_y:
                    highest_y = platform_y
                    highest_platform = platform

            if highest_platform:
                # For alternating platforms, always spawn from the higher end
                platform_index = len(self.platforms) - self.platforms.index(highest_platform)
                if platform_index % 2 == 0:  # Even platforms (slope down left to right)
                    spawn_pos = (0, highest_platform.p1[1])  # Spawn at left edge (higher point)
                    velocity = (self.ball_config["speed"] * 100, 0)  # Roll right
                else:  # Odd platforms (slope down right to left)
                    spawn_pos = (WINDOW_WIDTH, highest_platform.p2[1])  # Spawn at right edge (higher point)
                    velocity = (-self.ball_config["speed"] * 100, 0)  # Roll left

                can_spawn = True
                for ball in self.balls:
                    if abs(ball.body.position.x - spawn_pos[0]) < 100:
                        can_spawn = False
                        break

                if can_spawn:
                    new_ball = create_ball(space, self.ball_config["size"])
                    new_ball.body.position = spawn_pos
                    new_ball.body.velocity = velocity
                    self.balls.append(new_ball)
                    self.ball_spawn_timer = 0

        kept_balls = []
        for ball in self.balls:
            if -100 < ball.body.position.x < 900 and ball.body.position.y < 650:
                kept_balls.append(ball)
            else:
                space.remove(ball.body, ball.shape)
        self.balls = kept_balls

    def draw(self, screen):
        for platform in self.platforms:
            platform.draw(screen)

        for ladder in self.ladders:
            ladder.draw(screen)

        for ball in self.balls:
            ball.draw(screen)

        pygame.draw.rect(screen, (0, 255, 0), self.finish_zone)

    def get_jump_height(self):
        return self.jump_height

    def get_ladders(self):
        return self.ladders

    def check_finish(self, player):
        player_rect = pygame.Rect(
            player.body.position.x - player.width / 2,
            player.body.position.y - player.height / 2,
            player.width,
            player.height
        )
        return self.finish_zone.colliderect(player_rect)