import pygame
import math
from base_car import BaseCar


class AI2(BaseCar):
    def __init__(self, pos, color, number, track, difficulty=1.0):
        super().__init__(pos, color, number)
        self.track = track
        self.debug_mode = True  # Enable debug visualization
        self.debug_surface = pygame.Surface((track.outer_rect.width + 100, track.outer_rect.height + 100),
                                            pygame.SRCALPHA)
        self.waypoints = self._generate_racing_line()
        self.current_waypoint = 0
        self.difficulty = difficulty
        self.max_speed *= difficulty
        self.acceleration *= difficulty
        self.target_threshold = 40
        self.cornering_speed = self.max_speed * 0.75
        self.straight_speed = self.max_speed
        self.prev_angle_diff = 0  # For detecting sudden angle changes
        self.debug_log = []  # Store debug information
        self.frame_counter = 0
        self.log_frequency = 60  # Log every 60 frames

    def _generate_racing_line(self):
        """Generate an optimal racing line with proper corner anticipation"""
        waypoints = []
        outer = self.track.outer_rect
        track_width = self.track.track_width

        # Calculate optimal racing line parameters
        entry_offset = track_width * 0.6  # Reduced from 0.7
        apex_inset = track_width * 0.4  # Increased from 0.3

        # Define key points for each corner
        corners = [
            # Bottom right corner
            {
                'entry': (outer.right - track_width * 1.5, outer.bottom - entry_offset),
                'apex': (outer.right - apex_inset, outer.bottom - apex_inset),
                'exit': (outer.right - entry_offset, outer.bottom - track_width * 1.5)
            },
            # Top right corner
            {
                'entry': (outer.right - entry_offset, outer.top + track_width * 1.5),
                'apex': (outer.right - apex_inset, outer.top + apex_inset),
                'exit': (outer.right - track_width * 1.5, outer.top + entry_offset)
            },
            # Top left corner
            {
                'entry': (outer.left + track_width * 1.5, outer.top + entry_offset),
                'apex': (outer.left + apex_inset, outer.top + apex_inset),
                'exit': (outer.left + entry_offset, outer.top + track_width * 1.5)
            },
            # Bottom left corner
            {
                'entry': (outer.left + entry_offset, outer.bottom - track_width * 1.5),
                'apex': (outer.left + apex_inset, outer.bottom - apex_inset),
                'exit': (outer.left + track_width * 1.5, outer.bottom - entry_offset)
            }
        ]

        # Generate waypoints including approach and exit points
        for corner in corners:
            # Add approach point before entry
            approach_vector = (
                corner['entry'][0] - corner['apex'][0],
                corner['entry'][1] - corner['apex'][1]
            )
            approach_dist = math.sqrt(approach_vector[0] ** 2 + approach_vector[1] ** 2)
            approach_unit = (approach_vector[0] / approach_dist, approach_vector[1] / approach_dist)
            approach_point = (
                corner['entry'][0] + approach_unit[0] * track_width,
                corner['entry'][1] + approach_unit[1] * track_width
            )

            waypoints.extend([
                approach_point,
                corner['entry'],
                corner['apex'],
                corner['exit']
            ])

        return waypoints

    def _log_debug_info(self, angle_diff, distance, corner_factor, current_speed, target_speed):
        """Log debugging information to console"""
        if self.frame_counter % self.log_frequency == 0:
            debug_info = {
                'frame': self.frame_counter,
                'waypoint': self.current_waypoint,
                'angle_diff': round(angle_diff, 2),
                'distance': round(distance, 2),
                'corner_factor': round(corner_factor, 2),
                'speed': round(current_speed, 2),
                'target_speed': round(target_speed, 2),
                'position': (round(self.pos[0], 2), round(self.pos[1], 2)),
                'angle': round(self.angle, 2)
            }

            print(f"\nFrame {debug_info['frame']} - Waypoint {debug_info['waypoint']}")
            print(f"Angle Diff: {debug_info['angle_diff']}° (prev: {round(self.prev_angle_diff, 2)}°)")
            print(f"Distance: {debug_info['distance']} px")
            print(f"Corner Factor: {debug_info['corner_factor']}")
            print(f"Speed: {debug_info['speed']} / {debug_info['target_speed']}")
            print(f"Position: {debug_info['position']}")
            print(f"Car Angle: {debug_info['angle']}°")

            self.debug_log.append(debug_info)

    def _calculate_corner_factor(self):
        """Calculate how sharp the upcoming turn is"""
        next_wp = (self.current_waypoint + 1) % len(self.waypoints)
        next_next_wp = (self.current_waypoint + 2) % len(self.waypoints)

        current_pos = self.waypoints[self.current_waypoint]
        next_pos = self.waypoints[next_wp]
        next_next_pos = self.waypoints[next_next_wp]

        v1 = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
        v2 = (next_next_pos[0] - next_pos[0], next_next_pos[1] - next_pos[1])

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 * mag2 == 0:
            return 0

        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.acos(cos_angle)

        return angle / math.pi

    def handle_input(self, game_state):
        if not game_state.race_started:
            return

        self.frame_counter += 1
        target = self.waypoints[self.current_waypoint]

        # Calculate distance and angle to target
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        distance = math.hypot(dx, dy)

        # Calculate target angle considering car's current orientation
        target_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = self._normalize_angle(target_angle - self.angle)

        # Detect if we're turning the wrong way
        if abs(angle_diff) > 150:  # If target is almost behind us
            # Force a right turn if we're stuck
            angle_diff = -60 if angle_diff > 0 else 60

        corner_factor = self._calculate_corner_factor()
        current_speed = math.hypot(*self.vel)

        # Adjust target speed based on corner factor and angle difference
        base_target_speed = self.straight_speed * (1 - corner_factor * 0.4)
        angle_speed_factor = 1 - (abs(angle_diff) / 180) * 0.5
        target_speed = base_target_speed * angle_speed_factor

        # Speed control with smoother transitions
        if current_speed < target_speed * 0.95:
            self.accelerate()
        elif current_speed > target_speed * 1.05:
            self.decelerate()

        # Enhanced steering control
        turn_threshold = 3  # Reduced from 5
        if abs(angle_diff) > turn_threshold:
            # More aggressive turning for larger angles
            turn_rate = min(abs(angle_diff) / 30.0, 1.0) * self.turn_speed
            if angle_diff > 0:
                self.turn(turn_rate)
            else:
                self.turn(-turn_rate)

        # Log debug information
        self._log_debug_info(angle_diff, distance, corner_factor, current_speed, target_speed)
        self.prev_angle_diff = angle_diff

        # Switch to next waypoint when close enough
        if distance < self.target_threshold:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)

    def _normalize_angle(self, angle):
        """Normalize angle to be between -180 and 180 degrees"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def draw(self, screen):
        """Override draw method to add debug visualization"""
        super().draw(screen)

        if self.debug_mode:
            # Clear debug surface
            self.debug_surface.fill((0, 0, 0, 0))

            # Draw all waypoints
            for i, wp in enumerate(self.waypoints):
                color = (0, 255, 0) if i == self.current_waypoint else (200, 200, 200)
                pygame.draw.circle(screen, color, (int(wp[0]), int(wp[1])), 5)

                # Draw lines between waypoints
                next_wp = self.waypoints[(i + 1) % len(self.waypoints)]
                pygame.draw.line(screen, (100, 100, 100), wp, next_wp, 1)

            # Draw target line
            target = self.waypoints[self.current_waypoint]
            pygame.draw.line(screen, (255, 0, 0),
                             (int(self.pos[0]), int(self.pos[1])),
                             (int(target[0]), int(target[1])), 2)

            # Draw velocity vector
            vel_scale = 5
            pygame.draw.line(screen, (0, 255, 255),
                             (int(self.pos[0]), int(self.pos[1])),
                             (int(self.pos[0] + self.vel[0] * vel_scale),
                              int(self.pos[1] + self.vel[1] * vel_scale)), 2)

            # Draw cardinal direction
            direction_length = 40
            angle_rad = math.radians(self.angle)
            end_pos = (int(self.pos[0] + math.cos(angle_rad) * direction_length),
                       int(self.pos[1] + math.sin(angle_rad) * direction_length))
            pygame.draw.line(screen, (255, 255, 0),
                             (int(self.pos[0]), int(self.pos[1])), end_pos, 2)