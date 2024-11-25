import pymunk
import pygame
import math


class Player:
    def __init__(self, space, position, size, jump_height):
        self.width = size
        self.height = size
        mass = 1
        moment = pymunk.moment_for_box(mass, (self.width, self.height))
        self.body = pymunk.Body(mass, moment)
        self.body.position = position

        self.body.angular_velocity_limit = 15
        self.body.angular_damping = 0.9

        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))
        self.shape.friction = 0.7
        self.shape.elasticity = 0
        self.shape.collision_type = 3

        space.add(self.body, self.shape)

        gravity = abs(space.gravity[1])
        self.jump_speed = math.sqrt(2 * gravity * (jump_height + 5))

        self.move_speed = 300
        self.on_ground = False
        self.on_ladder = False
        self.current_ladder = None
        self.jump_cooldown = 0
        self.JUMP_COOLDOWN_TIME = 5

    def update(self, keys, space, ladders):
        near_ladder = False
        nearest_ladder = None

        for ladder in ladders:
            if (abs(self.body.position.x - (ladder.x + ladder.width / 2)) < 20 and
                    ladder.y1 - self.height <= self.body.position.y <= ladder.y2):
                near_ladder = True
                nearest_ladder = ladder
                break

        if near_ladder and (keys[pygame.K_UP] or keys[pygame.K_DOWN]):
            self.on_ladder = True
            self.current_ladder = nearest_ladder
            self.body.position = (nearest_ladder.x + nearest_ladder.width / 2, self.body.position.y)
            self.body.velocity = (0, 0)

        if self.on_ladder:
            self.body.gravity = (0, 0)

            move_y = 0
            if keys[pygame.K_UP]:
                move_y = -4
            elif keys[pygame.K_DOWN]:
                move_y = 4

            if move_y != 0:
                new_y = self.body.position.y + move_y

                if new_y <= self.current_ladder.y1:
                    # Position player at the ladder end
                    self.body.position = (self.body.position.x, self.current_ladder.y1)

                    if keys[pygame.K_UP]:  # When player is trying to go up
                        self.on_ladder = False
                        self.body.gravity = space.gravity

                        # Stronger upward boost and slight horizontal movement based on direction
                        boost_y = -150  # Stronger upward boost
                        boost_x = 0

                        if keys[pygame.K_LEFT]:
                            boost_x = -50
                        elif keys[pygame.K_RIGHT]:
                            boost_x = 50

                        self.body.velocity = (boost_x, boost_y)  # Combined boost
                elif new_y >= self.current_ladder.y2:
                    self.body.position = (self.body.position.x, self.current_ladder.y2)
                    self.on_ladder = False
                    self.body.gravity = space.gravity
                else:
                    self.body.position = (self.body.position.x, new_y)

            if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
                self.on_ladder = False
                self.body.gravity = space.gravity
            else:
                self.body.position = (self.current_ladder.x + self.current_ladder.width / 2, self.body.position.y)
                self.body.velocity = (0, 0)
        else:
            self.body.gravity = space.gravity
            vel_x = 0
            if keys[pygame.K_LEFT]:
                vel_x = -self.move_speed
            elif keys[pygame.K_RIGHT]:
                vel_x = self.move_speed

            current_vel = self.body.velocity
            world_vel = (vel_x, current_vel.y)
            self.body.velocity = world_vel

        if self.body.position.x < 0:
            self.body.position = (0, self.body.position.y)
            self.body.velocity = (0, self.body.velocity.y)
        elif self.body.position.x > 800:
            self.body.position = (800, self.body.position.y)
            self.body.velocity = (0, self.body.velocity.y)

        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1

    def jump(self):
        if (self.on_ground or self.on_ladder) and self.jump_cooldown == 0:
            jump_impulse = (0, -self.jump_speed * self.body.mass)
            self.body.apply_impulse_at_world_point(jump_impulse, self.body.position)
            self.on_ground = False
            self.on_ladder = False
            self.jump_cooldown = self.JUMP_COOLDOWN_TIME

    def draw(self, screen):
        points = [self.body.local_to_world(v) for v in self.shape.get_vertices()]
        points = [(int(p.x), int(p.y)) for p in points]

        pygame.draw.polygon(screen, (255, 100, 100), points)
        pygame.draw.polygon(screen, (255, 0, 0), points, 2)

        center = (int(self.body.position.x), int(self.body.position.y))
        pygame.draw.circle(screen, (255, 255, 255), center, 2)


def collision_handler(arbiter, space, data):
    player = data['player']
    normal = arbiter.normal
    if abs(normal.y) > abs(normal.x):
        player.on_ground = True
    return True