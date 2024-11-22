import pygame
import pymunk
import random

from game_objects import Platform, Ball, create_ball, Ladder


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
        ball_size = 15  # Define ball size
        self.jump_height = ball_size * 2.5  # Jump height relative to ball size

        self.ball_config = {
            "interval": max(180 - (number - 1) * 20, 60),
            "speed": 3 + (number - 1),
            "max_balls": 1 if number == 1 else min(2 + number, 8),
            "size": ball_size
        }

    def generate_platforms(self, space):
        thickness = 16
        screen_width = 800
        screen_height = 600

        self.platforms = []
        self.ladders = []

        # First platform (ground) positioning
        first_platform_y = screen_height - self.platform_spacing
        ground = Platform(space, (0, first_platform_y), (screen_width, first_platform_y), thickness)
        self.platforms.append(ground)

        # Create walls
        left_wall = Platform(space, (-20, 0), (-20, screen_height), thickness)
        right_wall = Platform(space, (screen_width + 20, 100), (screen_width + 20, screen_height), thickness)
        self.platforms.extend([left_wall, right_wall])

        last_y = first_platform_y

        for i in range(self.num_platforms):
            y = last_y - self.platform_spacing

            if i % 2 == 0:
                start_x = -20
                end_x = random.randint(600, 750)
                start_y = y
                end_y = y + 30

                # Add ladder connecting this platform to the one below
                ladder_x = random.randint(end_x - 200, end_x - 50)
                if i == 0:
                    # For the first platform, connect to ground platform
                    self.ladders.append(Ladder(ladder_x, end_y, first_platform_y))
                else:
                    # For others, connect to the platform below
                    self.ladders.append(Ladder(ladder_x, end_y, last_y))
            else:
                start_x = screen_width + 20
                end_x = random.randint(50, 200)
                start_y = y
                end_y = y + 30

                # Add ladder connecting this platform to the one below
                ladder_x = random.randint(end_x + 50, end_x + 200)
                self.ladders.append(Ladder(ladder_x, end_y, last_y))

            platform = Platform(space, (start_x, start_y), (end_x, end_y), thickness)
            self.platforms.append(platform)

            last_y = y

        # Finish zone placement
        highest_platform = self.platforms[-1]
        if highest_platform.p1[0] < 400:
            finish_x = highest_platform.p2[0] - 90
            finish_y = highest_platform.p2[1] - 30
        else:
            finish_x = highest_platform.p2[0]
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
                if highest_platform.p1[0] < highest_platform.p2[0]:
                    spawn_pos = (-50, highest_y - 15)
                    velocity = (self.ball_config["speed"] * 100, 0)
                else:
                    spawn_pos = (850, highest_y - 15)
                    velocity = (-self.ball_config["speed"] * 100, 0)

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