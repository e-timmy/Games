import pygame
import pymunk
from constants import *


class Player:
    def __init__(self, space, x, y):
        self.space = space
        self.ducked = False
        self.jumps_performed = 0
        self.coyote_time = 0
        self.ground_contact_time = 0

        self.body = pymunk.Body(1, pymunk.moment_for_box(1, (PLAYER_WIDTH, PLAYER_HEIGHT)))
        self.body.position = (x, y)

        self.shape = pymunk.Poly.create_box(self.body, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.shape.friction = 0.5
        self.shape.collision_type = 1

        space.add(self.body, self.shape)

    def is_on_ground(self):
        height = PLAYER_DUCKED_HEIGHT if self.ducked else PLAYER_HEIGHT
        point = self.body.position + (0, height / 2 + 1)
        ground = self.space.point_query_nearest(point, 1, pymunk.ShapeFilter())
        return ground is not None and self.body.velocity.y > -0.1

    def can_double_jump(self):
        return self.jumps_performed == 1 and not self.is_on_ground() and self.ducked

    def jump(self):

        if self.is_on_ground() or self.coyote_time > 0:
            self.body.velocity = (self.body.velocity.x, JUMP_FORCE)
            self.jumps_performed = 1
            self.coyote_time = 0
        elif self.can_double_jump():
            self.body.velocity = (self.body.velocity.x, JUMP_FORCE * 0.8)  # Slightly weaker double jump
            self.jumps_performed = 2
            self.stand()

    def duck(self):

        if not self.ducked:
            self.ducked = True
            self.space.remove(self.shape)
            self.shape = pymunk.Poly.create_box(self.body, (PLAYER_WIDTH, PLAYER_DUCKED_HEIGHT))
            self.shape.friction = 0.5
            self.shape.collision_type = 1
            self.space.add(self.shape)

    def stand(self):

        if self.ducked:
            self.ducked = False
            self.space.remove(self.shape)
            self.shape = pymunk.Poly.create_box(self.body, (PLAYER_WIDTH, PLAYER_HEIGHT))
            self.shape.friction = 0.5
            self.shape.collision_type = 1
            self.space.add(self.shape)

    def update(self):
        was_on_ground = self.is_on_ground()

        if self.is_on_ground():
            self.ground_contact_time += 1
            if self.ground_contact_time > 5:  # Require 5 frames of ground contact
                self.jumps_performed = 0
                self.coyote_time = 5
        else:
            self.ground_contact_time = 0
            if was_on_ground:
                self.coyote_time = 5

        if not self.is_on_ground() and self.coyote_time > 0:
            self.coyote_time -= 1

        # Check for collision with platforms
        for shape in self.space.shapes:
            if shape.collision_type == COLLISION_TYPE_PLATFORM:
                if self.shape.shapes_collide(shape).points:
                    # If player is below the platform and moving upwards, ignore collision
                    if self.body.position.y > shape.body.position.y and self.body.velocity.y < 0:
                        continue
                    # Otherwise, handle collision normally
                    self.handle_collision(shape.parent)

    def handle_collision(self, platform):
        # Handle collision with platform
        # This method can be used to implement specific behaviors for different platform types
        pass

    def move_left(self):
        self.body.velocity = (-PLAYER_SPEED, self.body.velocity.y)

    def move_right(self):
        self.body.velocity = (PLAYER_SPEED, self.body.velocity.y)

    def stop_horizontal_movement(self):
        self.body.velocity = (0, self.body.velocity.y)

    def get_position(self):
        return self.body.position

    def draw(self, screen, camera_offset):
        height = PLAYER_DUCKED_HEIGHT if self.ducked else PLAYER_HEIGHT
        pos = self.body.position
        rect = pygame.Rect(
            pos.x - PLAYER_WIDTH / 2 + camera_offset[0],
            pos.y - height / 2 + camera_offset[1],
            PLAYER_WIDTH,
            height
        )

        pygame.draw.rect(screen, PLAYER_COLOR, rect)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)

        visor_height = height // 4
        visor_rect = pygame.Rect(
            rect.left,
            rect.top + visor_height,
            rect.width,
            visor_height
        )
        pygame.draw.rect(screen, (0, 255, 255), visor_rect)

        line_spacing = height // 5
        for i in range(1, 5):
            start = (rect.left, rect.top + i * line_spacing)
            end = (rect.right, rect.top + i * line_spacing)
            pygame.draw.line(screen, (0, 200, 200), start, end, 1)