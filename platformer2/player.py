import pygame
import pymunk
from constants import *


class Player:
    def __init__(self, space, x, y):
        self.ducked = False
        self.can_double_jump = False
        self.double_jump_timer = 0

        # Create main body
        self.body = pymunk.Body(1, pymunk.moment_for_box(1, (PLAYER_WIDTH, PLAYER_HEIGHT)))
        self.body.position = (x, y)

        self.shape = pymunk.Poly.create_box(self.body, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.shape.friction = 0.5
        self.shape.collision_type = 1

        space.add(self.body, self.shape)

        self.on_ground = False
        self.space = space

    def move_left(self):
        self.body.velocity = (-PLAYER_SPEED, self.body.velocity.y)

    def move_right(self):
        self.body.velocity = (PLAYER_SPEED, self.body.velocity.y)

    def stop_horizontal_movement(self):
        self.body.velocity = (0, self.body.velocity.y)

    def jump(self):
        if self.on_ground:
            self.body.velocity = (self.body.velocity.x, JUMP_FORCE)
            self.on_ground = False
            self.can_double_jump = True
        elif self.can_double_jump and self.ducked and self.double_jump_timer == 0:
            self.body.velocity = (self.body.velocity.x, JUMP_FORCE * 0.8)  # Slightly weaker double jump
            self.can_double_jump = False
            self.double_jump_timer = 10  # Set a small delay before allowing to stand up

    def duck(self):
        if not self.ducked:
            self.ducked = True
            # Remove old shape
            self.space.remove(self.shape)
            # Create new, shorter shape
            self.shape = pymunk.Poly.create_box(self.body, (PLAYER_WIDTH, PLAYER_DUCKED_HEIGHT))
            self.shape.friction = 0.5
            self.shape.collision_type = 1
            self.space.add(self.shape)

    def stand(self):
        if self.ducked and self.double_jump_timer == 0:
            self.ducked = False
            # Remove ducked shape
            self.space.remove(self.shape)
            # Create original shape
            self.shape = pymunk.Poly.create_box(self.body, (PLAYER_WIDTH, PLAYER_HEIGHT))
            self.shape.friction = 0.5
            self.shape.collision_type = 1
            self.space.add(self.shape)

    def update(self):
        # Check if player is on ground
        height = PLAYER_DUCKED_HEIGHT if self.ducked else PLAYER_HEIGHT
        point = self.body.position + (0, height / 2 + 2)
        ground = self.space.point_query_nearest(point, 5, pymunk.ShapeFilter())

        if ground:
            self.on_ground = True
            self.can_double_jump = False
            # Adjust player position if slightly below ground
            if self.body.position.y + height / 2 > ground.point.y:
                self.body.position = (self.body.position.x, ground.point.y - height / 2)
        else:
            self.on_ground = False

        if self.double_jump_timer > 0:
            self.double_jump_timer -= 1

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