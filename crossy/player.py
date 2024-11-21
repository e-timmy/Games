import random

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
        self.moving_up = False
        self.moving_down = False
        self.direction = 1
        self.vertical_direction = 0
        self.test_mode = False
        self.on_log = False
        self.on_lilypad = False
        self.in_river = False
        self.river_lane_index = -1
        self.move_cooldown = 0
        self.move_cooldown_max = 5

        # Death animation attributes
        self.is_dead = False
        self.death_type = None  # 'road' or 'river'
        self.death_animation_time = 0
        self.death_animation_duration = 60  # 1 second at 60 FPS
        self.flatten_scale = 1.0  # For road death
        self.sink_depth = 0  # For river death
        self.bubble_positions = []  # For river death

    def toggle_test_mode(self, enabled):
        self.test_mode = enabled
        self.color = self.test_mode_color if enabled else self.original_color

    def update(self, lanes, river_top, river_bottom):
        if self.is_dead:
            self.death_animation_time += 1
            if self.death_type == 'river':
                if len(self.bubble_positions) < 8 and random.random() < 0.1:
                    self.bubble_positions.append((
                        self.x + random.randint(-15, 15),
                        self.y + random.randint(0, 10)
                    ))
            # Return True when animation is complete
            return self.death_animation_time >= 30  # Match this to your animation duration

        self.animation_frame = (self.animation_frame + 0.2) % (2 * math.pi)

        if self.move_cooldown > 0:
            self.move_cooldown -= 1
            return

        if self.celebrating:
            self.celebration_time += 1
            self.celebration_height = math.sin(self.celebration_time * 0.1) * 20
            return

        keys = pygame.key.get_pressed()
        self.walking = False
        self.moving_up = False
        self.moving_down = False

        self.toggle_test_mode(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

        # Horizontal movement
        if keys[pygame.K_LEFT]:
            self.x = max(0, self.x - self.speed)
            self.walking = True
            self.direction = -1
        elif keys[pygame.K_RIGHT]:
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
            self.moving_up = True
            self.vertical_direction = -1
        elif keys[pygame.K_DOWN]:
            self._handle_downward_movement(lanes, river_top, river_bottom)
            self.moving_down = True
            self.vertical_direction = 1

    def _continue_movement(self):
        diff = self.target_y - self.y
        if abs(diff) < self.move_speed:
            self.y = self.target_y
            self.is_moving = False
            self.move_cooldown = self.move_cooldown_max
            self.vertical_direction = 0
        else:
            move_dir = 1 if diff > 0 else -1
            self.y += self.move_speed * move_dir
            self.vertical_direction = move_dir
            self.moving_up = move_dir == -1
            self.moving_down = move_dir == 1

    def move_on_log(self, log_speed):
        self.x += log_speed
        self.x = max(0, min(self.x, 800 - self.width))

    def move_on_lilypad(self, lilypad_speed):
        self.x += lilypad_speed
        self.x = max(0, min(self.x, 800 - self.width))

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
            # Move to initial starting position (50 pixels below road)
            self.lane_index = -1
            self.target_y = lanes[-1] + 50  # Safe distance below the last lane
            self.is_moving = True

    def draw(self, screen):

        if self.is_dead:
            if self.death_type == 'road':
                self._draw_road_death(screen)
            elif self.death_type == 'river':
                self._draw_river_death(screen)
            return

        # Reduced horizontal waddle for vertical movement
        vertical_waddle = math.sin(self.animation_frame * 2) * 1 if (
                    self.moving_up or self.moving_down or self.is_moving) else 0

        # Vertical bounce for horizontal movement (unchanged)
        horizontal_bounce = math.sin(self.animation_frame * 2) * 2 if self.walking else 0

        # Calculate display position
        display_x = self.x + self.width / 2 + vertical_waddle
        display_y = self.y - self.celebration_height + horizontal_bounce
        y_center = display_y - self.height / 2

        # Body (simple circle)
        pygame.draw.circle(screen, self.color, (int(display_x), int(y_center + self.height / 2)),
                           int(self.width / 2))

        # Determine beak and eye positions based on movement direction
        if self.vertical_direction != 0:
            # Vertical movement beak
            beak_points = [
                (display_x - 3, y_center + self.height / 2 + self.vertical_direction * 5),
                (display_x, y_center + self.height / 2 + self.vertical_direction * 10),
                (display_x + 3, y_center + self.height / 2 + self.vertical_direction * 5)
            ]
            pygame.draw.polygon(screen, (255, 100, 0), beak_points)

            # Two eyes for vertical movement (front view)
            eye_y = y_center + self.height / 2 + (self.vertical_direction * -2)
            pygame.draw.circle(screen, (0, 0, 0),
                               (int(display_x - 4), int(eye_y)),
                               2)
            pygame.draw.circle(screen, (0, 0, 0),
                               (int(display_x + 4), int(eye_y)),
                               2)
        else:
            # Horizontal movement beak
            beak_points = [
                (display_x + self.direction * 5, y_center + self.height / 2 - 3),
                (display_x + self.direction * 10, y_center + self.height / 2),
                (display_x + self.direction * 5, y_center + self.height / 2 + 3)
            ]
            pygame.draw.polygon(screen, (255, 100, 0), beak_points)

            # One eye for horizontal movement (side view)
            pygame.draw.circle(screen, (0, 0, 0),
                               (int(display_x + self.direction * 2),
                                int(y_center + self.height / 2 - 2)),
                               2)

        # Legs animation
        leg_color = (255, 100, 0)
        if self.vertical_direction != 0 or self.is_moving:
            # Vertical movement leg animation (smaller movement)
            leg_offset = math.sin(self.animation_frame * 2) * 1.5
            pygame.draw.line(screen, leg_color,
                             (display_x - 5, y_center + self.height),
                             (display_x - 5 - leg_offset, y_center + self.height + 5),
                             2)
            pygame.draw.line(screen, leg_color,
                             (display_x + 5, y_center + self.height),
                             (display_x + 5 + leg_offset, y_center + self.height + 5),
                             2)
        else:
            # Horizontal movement leg animation (unchanged)
            leg_offset = math.sin(self.animation_frame) * 3 if self.walking else 0
            pygame.draw.line(screen, leg_color,
                             (display_x - 7, y_center + self.height),
                             (display_x - 7 + leg_offset, y_center + self.height + 5),
                             2)
            pygame.draw.line(screen, leg_color,
                             (display_x + 7, y_center + self.height),
                             (display_x + 7 - leg_offset, y_center + self.height + 5),
                             2)

    def start_celebration(self):
        self.celebrating = True
        self.celebration_time = 0

    def die(self, death_type):
        self.is_dead = True
        self.death_type = death_type
        self.death_animation_time = 0
        if death_type == 'river':
            self.bubble_positions = [(self.x + random.randint(-10, 10), self.y)
                                     for _ in range(5)]

    def _draw_road_death(self, screen):
        display_x = self.x + self.width / 2
        display_y = self.y + self.height / 2

        # Draw flattened body
        flattened_height = self.height * self.flatten_scale
        pygame.draw.ellipse(screen, self.color,
                            (display_x - self.width / 2,
                             display_y - flattened_height / 2,
                             self.width, flattened_height))

        # Draw X eyes
        eye_color = (0, 0, 0)
        eye_size = 2
        left_eye_x = display_x - 4
        right_eye_x = display_x + 4
        eye_y = display_y - flattened_height / 4

        pygame.draw.line(screen, eye_color,
                         (left_eye_x - eye_size, eye_y - eye_size),
                         (left_eye_x + eye_size, eye_y + eye_size), 2)
        pygame.draw.line(screen, eye_color,
                         (left_eye_x + eye_size, eye_y - eye_size),
                         (left_eye_x - eye_size, eye_y + eye_size), 2)

        pygame.draw.line(screen, eye_color,
                         (right_eye_x - eye_size, eye_y - eye_size),
                         (right_eye_x + eye_size, eye_y + eye_size), 2)
        pygame.draw.line(screen, eye_color,
                         (right_eye_x + eye_size, eye_y - eye_size),
                         (right_eye_x - eye_size, eye_y + eye_size), 2)

    def _draw_river_death(self, screen):
        display_x = self.x + self.width / 2
        display_y = self.y + self.height / 2

        # Calculate shrink and fade progress
        progress = min(1.0, self.death_animation_time / 30)  # Complete in 0.5 seconds (30 frames)

        # Draw character shrinking and fading
        if progress < 1.0:
            # Calculate shrink factor and transparency
            shrink_factor = 1.0 - progress
            alpha = int(255 * (1 - progress))

            # Draw shrinking, fading character
            char_size = int(self.width * shrink_factor)
            char_surface = pygame.Surface((char_size * 2, char_size * 2), pygame.SRCALPHA)

            # Main body
            pygame.draw.circle(char_surface, (*self.color, alpha),
                               (char_size, char_size), char_size)

            # X eyes (scaled with character)
            eye_size = max(1, int(2 * shrink_factor))
            eye_offset = max(2, int(4 * shrink_factor))
            eye_color = (0, 0, 0, alpha)

            for x_offset in [-eye_offset, eye_offset]:
                pygame.draw.line(char_surface, eye_color,
                                 (char_size + x_offset - eye_size, char_size - eye_size),
                                 (char_size + x_offset + eye_size, char_size + eye_size), 2)
                pygame.draw.line(char_surface, eye_color,
                                 (char_size + x_offset + eye_size, char_size - eye_size),
                                 (char_size + x_offset - eye_size, char_size + eye_size), 2)

            screen.blit(char_surface,
                        (int(display_x - char_size),
                         int(display_y - char_size)))

        # Draw ripple effect
        ripple_size = 20 + (progress * 10)  # Ripple expands as character shrinks
        ripple_alpha = max(0, int(255 * (1 - progress)))
        ripple_surface = pygame.Surface((ripple_size * 2, ripple_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(ripple_surface, (255, 255, 255, ripple_alpha),
                           (ripple_size, ripple_size), ripple_size, 2)
        screen.blit(ripple_surface,
                    (int(display_x - ripple_size),
                     int(display_y - ripple_size)))

        # Draw bubbles
        for pos in self.bubble_positions[:]:
            bx, by = pos
            bubble_y = by - (self.death_animation_time * 1.5)
            bubble_x = bx + math.sin(self.death_animation_time * 0.1 + bx * 0.1) * 2

            bubble_alpha = max(0, int(255 * (1 - (bubble_y - self.y + 50) / 50)))
            if bubble_alpha > 0:
                bubble_surface = pygame.Surface((8, 8), pygame.SRCALPHA)
                pygame.draw.circle(bubble_surface, (255, 255, 255, bubble_alpha),
                                   (4, 4), 2)
                screen.blit(bubble_surface,
                            (int(bubble_x - 4), int(bubble_y - 4)))