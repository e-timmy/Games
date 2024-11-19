import pygame
import math


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.target_y = y
        self.width = 20
        self.height = 20
        self.speed = 3
        self.lane_index = -1
        self.original_color = (255, 200, 0)
        self.test_mode_color = (200, 200, 255)
        self.color = self.original_color
        self.is_moving = False
        self.move_speed = 5
        self.celebrating = False
        self.celebration_time = 0
        self.celebration_height = 0
        self.animation_frame = 0
        self.walking = False
        self.direction = 1
        self.test_mode = False
        self.on_log = False
        self.on_lilypad = False
        self.in_river = False
        self.river_lane_index = -1

    def toggle_test_mode(self, enabled):
        self.test_mode = enabled
        self.color = self.test_mode_color if enabled else self.original_color

    def update(self, lanes, river_top, river_bottom):
        self.animation_frame = (self.animation_frame + 0.2) % (2 * math.pi)

        if self.celebrating:
            self.celebration_time += 1
            self.celebration_height = math.sin(self.celebration_time * 0.1) * 20
            return

        keys = pygame.key.get_pressed()
        self.walking = False

        self.toggle_test_mode(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

        # Horizontal movement
        if keys[pygame.K_LEFT]:
            self.x = max(0, self.x - self.speed)
            self.walking = True
            self.direction = -1
        if keys[pygame.K_RIGHT]:
            self.x = min(800 - self.width, self.x + self.speed)
            self.walking = True
            self.direction = 1

        # Don't process vertical movement if already moving
        if self.is_moving:
            self._continue_movement()
            return

        # Handle vertical movement based on current position
        if keys[pygame.K_UP]:
            self._handle_upward_movement(lanes, river_top, river_bottom)
        elif keys[pygame.K_DOWN]:
            self._handle_downward_movement(lanes, river_top, river_bottom)

    def move_on_log(self, log_speed):
        self.x += log_speed
        # Ensure the player stays within the screen bounds
        self.x = max(0, min(self.x, 800 - self.width))

    def move_on_lilypad(self, lilypad_speed):
        self.x += lilypad_speed
        # Ensure the player stays within the screen bounds
        self.x = max(0, min(self.x, 800 - self.width))

    def _continue_movement(self):
        diff = self.target_y - self.y
        if abs(diff) < self.move_speed:
            self.y = self.target_y
            self.is_moving = False
        else:
            move_dir = 1 if diff > 0 else -1
            self.y += self.move_speed * move_dir

    def _handle_upward_movement(self, lanes, river_top, river_bottom):
        # Starting area (bottom of screen)
        if self.lane_index == -1 and len(lanes) > 0:
            self.lane_index = len(lanes) - 1
            self.target_y = lanes[self.lane_index]
            self.is_moving = True
            return

        # In road lanes
        if 0 <= self.lane_index < len(lanes):
            if self.lane_index > 0:
                self.lane_index -= 1
                self.target_y = lanes[self.lane_index]
                self.is_moving = True
            elif river_top is not None:  # Move to river if it exists
                self.lane_index = -2
                self.river_lane_index = 2  # Start at bottom river lane
                self.target_y = river_bottom - (river_bottom - river_top) / 6  # Center of bottom river lane
                self.is_moving = True
                self.in_river = True
            else:  # Move to finish line if no river
                self.lane_index = -3
                self.target_y = 0  # Top of the screen
                self.is_moving = True
            return

        # In river
        if self.in_river and river_top is not None:
            lane_height = (river_bottom - river_top) / 3  # Assuming max 3 lanes
            if self.river_lane_index > 0:  # If not in the top river lane
                self.river_lane_index -= 1
                self.target_y = river_top + (self.river_lane_index + 0.5) * lane_height
                self.is_moving = True
            else:  # At top river lane, move to finish
                self.lane_index = -3
                self.target_y = river_top - self.height
                self.is_moving = True
                self.in_river = False
                self.river_lane_index = -1
            return

        # Allow moving to finish line from any position if not already there
        if self.lane_index != -3:
            self.lane_index = -3
            self.target_y = 0  # Top of the screen
            self.is_moving = True

    def _handle_downward_movement(self, lanes, river_top, river_bottom):
        # From finish line to river or first road lane
        if self.lane_index == -3:
            if river_top is not None:
                self.lane_index = -2
                self.river_lane_index = 0
                lane_height = (river_bottom - river_top) / 3  # Assuming max 3 lanes
                self.target_y = river_top + (0.5 * lane_height)  # Center of first lane
                self.is_moving = True
                self.in_river = True
            elif len(lanes) > 0:
                self.lane_index = 0
                self.target_y = lanes[0]
                self.is_moving = True
            return

        # In river
        if self.in_river and river_bottom is not None:
            lane_height = (river_bottom - river_top) / 3  # Assuming max 3 lanes
            if self.river_lane_index < 2:  # If not in bottom river lane
                self.river_lane_index += 1
                self.target_y = river_top + (self.river_lane_index + 0.5) * lane_height
                self.is_moving = True
            else:  # At bottom river lane, move to road
                self.lane_index = 0
                self.target_y = lanes[0]
                self.is_moving = True
                self.in_river = False
                self.river_lane_index = -1
            return

        # In road lanes
        if 0 <= self.lane_index < len(lanes) - 1:
            self.lane_index += 1
            self.target_y = lanes[self.lane_index]
            self.is_moving = True
        elif self.lane_index == len(lanes) - 1:
            self.lane_index = -1
            self.target_y = 600 - self.height  # Move to bottom of the screen
            self.is_moving = True

    def draw(self, screen):
        # Calculate vertical offset for walking animation
        walk_bounce = math.sin(self.animation_frame * 2) * 2 if self.walking else 0

        # Calculate actual y position including celebration and walking
        display_y = self.y - self.celebration_height + walk_bounce
        y_center = display_y - self.height / 2

        # Body (simple circle)
        pygame.draw.circle(screen, self.color, (int(self.x + self.width / 2), int(y_center + self.height / 2)),
                           int(self.width / 2))

        # Beak pointing in movement direction
        beak_points = [
            (self.x + self.width / 2 + self.direction * 5, y_center + self.height / 2 - 3),
            (self.x + self.width / 2 + self.direction * 10, y_center + self.height / 2),
            (self.x + self.width / 2 + self.direction * 5, y_center + self.height / 2 + 3)
        ]
        pygame.draw.polygon(screen, (255, 100, 0), beak_points)5.75 &

        # Eye
        eye_x = self.x + self.width / 2 + self.direction * 2
        pygame.draw.circle(screen, (0, 0, 0),
                           (int(eye_x), int(y_center + self.height / 2 - 2)),
                           2)

        # Legs
        leg_offset = math.sin(self.animation_frame) * 3 if self.walking else 0
        leg_color = (255, 100, 0)
        pygame.draw.line(screen, leg_color,
                         (self.x + self.width * 0.3, y_center + self.height),
                         (self.x + self.width * 0.3 + leg_offset, y_center + self.height + 5),
                         2)
        pygame.draw.line(screen, leg_color,
                         (self.x + self.width * 0.7, y_center + self.height),
                         (self.x + self.width * 0.7 - leg_offset, y_center + self.height + 5),
                         2)

    def start_celebration(self):
        self.celebrating = True
        self.celebration_time = 0