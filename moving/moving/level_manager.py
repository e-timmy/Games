import pymunk
import pygame
from constants import *
from game_objects import Item, PowerUpType


class Level:
    def __init__(self, space, level_number):
        self.space = space
        self.level_number = level_number
        self.boundaries = []
        self.items = []
        self.x_offset = level_number * WINDOW_WIDTH

        # Descending wall properties
        self.descending_wall_body = None
        self.descending_wall_shape = None
        self.wall_height = 0
        self.target_wall_height = WINDOW_HEIGHT
        self.wall_descent_speed = 15
        self.wall_descending = False
        self.wall_started = False

        # Blocking wall properties
        self.blocking_wall_body = None
        self.blocking_wall_shape = None
        self.has_blocking_wall = True
        self.blocking_wall_height = WINDOW_HEIGHT
        self.blocking_wall_descending = False

        # Generate level content
        self.create_boundaries()
        self.generate_items()
        self.create_blocking_wall()

    def generate_items(self):
        if self.level_number == 0:
            item = Item(self.space,
                        (self.x_offset + WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2),
                        PowerUpType.JUMP)
            self.items.append(item)
        elif self.level_number == 1:
            item = Item(self.space,
                        (self.x_offset + WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2),
                        PowerUpType.SHOOT)
            self.items.append(item)
        elif self.level_number == 2:
            # Place item in top right corner
            item = Item(self.space,
                        (self.x_offset + WINDOW_WIDTH - 50, 50),
                        PowerUpType.GRAVITY)
            self.items.append(item)
        elif self.level_number == 3:
            # Create a challenging position for the fourth item
            # Place it on a high platform or in a difficult-to-reach corner
            item = Item(self.space,
                        (self.x_offset + WINDOW_WIDTH - 50, WINDOW_HEIGHT - 550),  # Very high up
                        PowerUpType.JUMP)  # You can change this to a new power type
            self.items.append(item)

            # Optionally, you could add some platforms here to make it even more challenging
            platform_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            platform_shape = pymunk.Segment(
                platform_body,
                (self.x_offset + WINDOW_WIDTH - 200, WINDOW_HEIGHT - 400),
                (self.x_offset + WINDOW_WIDTH - 100, WINDOW_HEIGHT - 400),
                WALL_THICKNESS / 2
            )
            platform_shape.friction = 1.0
            platform_shape.elasticity = 0.5
            platform_shape.collision_type = 3
            self.boundaries.append((platform_body, platform_shape))
            self.space.add(platform_body, platform_shape)
        else:
            # For any additional levels
            item = Item(self.space,
                        (self.x_offset + WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2),
                        PowerUpType.JUMP)
            self.items.append(item)

    def create_boundaries(self):
        # Create floor
        floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        floor_shape = pymunk.Segment(
            floor_body,
            (self.x_offset, WINDOW_HEIGHT - WALL_THICKNESS / 2),
            (self.x_offset + WINDOW_WIDTH, WINDOW_HEIGHT - WALL_THICKNESS / 2),
            WALL_THICKNESS / 2
        )
        floor_shape.friction = 1.0
        floor_shape.elasticity = 0.5
        floor_shape.collision_type = 3

        # Create roof
        roof_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        roof_shape = pymunk.Segment(
            roof_body,
            (self.x_offset, WALL_THICKNESS / 2),
            (self.x_offset + WINDOW_WIDTH, WALL_THICKNESS / 2),
            WALL_THICKNESS / 2
        )
        roof_shape.friction = 1.0
        roof_shape.elasticity = 0.5
        roof_shape.collision_type = 3

        self.boundaries = [(floor_body, floor_shape), (roof_body, roof_shape)]
        for body, shape in self.boundaries:
            self.space.add(body, shape)

    def create_blocking_wall(self):
        if self.has_blocking_wall:
            self.blocking_wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            self.blocking_wall_shape = pymunk.Segment(
                self.blocking_wall_body,
                (self.x_offset + WINDOW_WIDTH, 0),
                (self.x_offset + WINDOW_WIDTH, self.blocking_wall_height),
                WALL_THICKNESS / 2
            )
            self.blocking_wall_shape.friction = 1.0
            self.blocking_wall_shape.elasticity = 0.5
            self.blocking_wall_shape.collision_type = 3
            self.space.add(self.blocking_wall_body, self.blocking_wall_shape)

    def remove_blocking_wall(self):
        self.blocking_wall_descending = True

    def update_blocking_wall(self):
        if self.blocking_wall_descending:
            self.blocking_wall_height -= self.wall_descent_speed
            if self.blocking_wall_height <= 0:
                if self.blocking_wall_body and self.blocking_wall_shape:
                    self.space.remove(self.blocking_wall_body, self.blocking_wall_shape)
                    self.blocking_wall_body = None
                    self.blocking_wall_shape = None
                self.blocking_wall_height = 0
                self.has_blocking_wall = False
                self.blocking_wall_descending = False
            else:
                # Update the physical wall
                if self.blocking_wall_body and self.blocking_wall_shape:
                    self.space.remove(self.blocking_wall_body, self.blocking_wall_shape)
                self.create_blocking_wall()

    def start_wall_descent(self):
        if not self.wall_started:
            self.wall_descending = True
            self.wall_height = 0
            self.wall_started = True
            self.update_descending_wall()

    def update_descending_wall(self):
        if self.descending_wall_body is not None and self.descending_wall_shape is not None:
            self.space.remove(self.descending_wall_body, self.descending_wall_shape)
            self.descending_wall_body = None
            self.descending_wall_shape = None

        if self.wall_height > 0:
            self.descending_wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            self.descending_wall_shape = pymunk.Segment(
                self.descending_wall_body,
                (self.x_offset, 0),
                (self.x_offset, self.wall_height),
                WALL_THICKNESS / 2
            )
            self.descending_wall_shape.friction = 1.0
            self.descending_wall_shape.elasticity = 0.5
            self.descending_wall_shape.collision_type = 3
            self.space.add(self.descending_wall_body, self.descending_wall_shape)

    def update_wall(self):
        if self.wall_descending and self.wall_height < self.target_wall_height:
            self.wall_height += self.wall_descent_speed
            if self.wall_height >= self.target_wall_height:
                self.wall_height = self.target_wall_height
                self.wall_descending = False
            self.update_descending_wall()

        self.update_blocking_wall()

    def draw(self, screen, camera):
        # Draw floor and roof
        floor_start = camera.apply(self.x_offset, WINDOW_HEIGHT - WALL_THICKNESS)
        roof_start = camera.apply(self.x_offset, 0)

        if floor_start[0] > -1000:
            # Draw floor
            pygame.draw.rect(screen, GRAY,
                             (floor_start[0], floor_start[1],
                              WINDOW_WIDTH, WALL_THICKNESS))
            # Draw roof
            pygame.draw.rect(screen, GRAY,
                             (roof_start[0], roof_start[1],
                              WINDOW_WIDTH, WALL_THICKNESS))

        # Draw descending wall
        if self.wall_height > 0:
            wall_start = camera.apply(self.x_offset - WALL_THICKNESS / 2, 0)
            if wall_start[0] > -1000:  # Check if position is valid
                pygame.draw.rect(screen, GRAY,
                                 (wall_start[0], wall_start[1],
                                  WALL_THICKNESS, self.wall_height))

        # Draw blocking wall
        if self.has_blocking_wall and self.blocking_wall_height > 0:
            block_wall_start = camera.apply(self.x_offset + WINDOW_WIDTH - WALL_THICKNESS / 2, 0)
            if block_wall_start[0] > -1000:  # Check if position is valid
                pygame.draw.rect(screen, GRAY,
                                 (block_wall_start[0], block_wall_start[1],
                                  WALL_THICKNESS, self.blocking_wall_height))

        # Draw level boundary markers
        boundary_start = camera.apply(self.x_offset, 0)
        boundary_end = camera.apply(self.x_offset, WINDOW_HEIGHT)
        if boundary_start[0] > -1000:  # Check if position is valid
            pygame.draw.line(screen, RED, boundary_start, boundary_end, 3)

        right_boundary_start = camera.apply(self.x_offset + WINDOW_WIDTH, 0)
        right_boundary_end = camera.apply(self.x_offset + WINDOW_WIDTH, WINDOW_HEIGHT)
        if right_boundary_start[0] > -1000:  # Check if position is valid
            pygame.draw.line(screen, BLUE, right_boundary_start, right_boundary_end, 3)

        # Draw level number
        font = pygame.font.Font(None, 36)
        level_text = font.render(f"Level {self.level_number}", True, WHITE)
        text_pos = camera.apply(self.x_offset + 50, 50)
        if text_pos[0] > -1000:  # Check if position is valid
            screen.blit(level_text, text_pos)

        # Draw items
        for item in self.items:
            if not item.collected:
                item_pos = camera.apply(item.body.position.x, item.body.position.y)
                if item_pos[0] > -1000:  # Check if position is valid
                    pygame.draw.rect(screen, GREEN,
                                     (item_pos[0] - ITEM_SIZE / 2,
                                      item_pos[1] - ITEM_SIZE / 2,
                                      ITEM_SIZE, ITEM_SIZE))

    def clear(self):
        for item in self.items:
            if not item.collected:
                self.space.remove(item.body, item.shape)
        self.items.clear()

    def remove_completely(self):
        for body, shape in self.boundaries:
            self.space.remove(body, shape)
        self.boundaries.clear()
        if self.descending_wall_body and self.descending_wall_shape:
            self.space.remove(self.descending_wall_body, self.descending_wall_shape)
        if self.blocking_wall_body and self.blocking_wall_shape:
            self.space.remove(self.blocking_wall_body, self.blocking_wall_shape)
        self.clear()


class LevelManager:
    def __init__(self, space):
        self.space = space
        self.current_level_number = 0
        self.transitioning = False
        self.camera_offset = 0
        self.transition_speed = 10
        self.levels = []

        self.current_level = Level(space, 0)
        self.next_level = Level(space, 1)
        self.levels = [self.current_level, self.next_level]

        self.debug_font = pygame.font.Font(None, 24)
        self.transition_complete = False
        self.first_landing = False  # Track first landing

    def check_first_landing(self, player_state):
        # Check if player has landed for the first time
        if not self.first_landing and player_state == "GROUNDED":
            self.first_landing = True
            self.current_level.start_wall_descent()

    def check_level_complete(self, player_pos):
        level_boundary = (self.current_level_number + 1) * WINDOW_WIDTH
        if (not self.transitioning and player_pos.x >= level_boundary):
            self.start_transition()
            return True
        return False

    def update_transition(self):
        if self.transitioning:
            self.camera_offset += self.transition_speed
            if self.camera_offset >= WINDOW_WIDTH:
                self.complete_transition()
                # Start wall descent for the previous level
                if len(self.levels) >= 2:
                    self.levels[-2].start_wall_descent()

        # Update walls for all levels
        for level in self.levels:
            level.update_wall()

    def complete_transition(self):
        if len(self.levels) > 2:
            oldest_level = self.levels.pop(0)
            oldest_level.remove_completely()

        self.current_level = self.next_level
        self.current_level_number += 1

        self.next_level = Level(self.space, self.current_level_number + 1)
        self.levels.append(self.next_level)

        self.transitioning = False
        self.camera_offset = 0

    def get_item_from_shape(self, shape):
        for level in self.levels:
            for item in level.items:
                if item.shape == shape and not item.collected:
                    level.remove_blocking_wall()  # Remove blocking wall when item is collected
                    return item
        return None

    def draw_current_level(self, screen, camera):
        for level in self.levels:
            level.draw(screen, camera)

        debug_info = [
            f"Camera X: {camera.x:.0f}",
            f"Transition: {self.transitioning}",
            f"Camera Offset: {self.camera_offset:.0f}",
            f"Current Level: {self.current_level_number}",
            f"Level Boundary: {(self.current_level_number + 1) * WINDOW_WIDTH}"
        ]

        for i, text in enumerate(debug_info):
            debug_surface = self.debug_font.render(text, True, WHITE)
            screen.blit(debug_surface, (10, WINDOW_HEIGHT - 120 + (i * 20)))

    def start_transition(self):
        self.transitioning = True
        self.camera_offset = 0