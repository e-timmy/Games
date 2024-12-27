import pygame


from base_car import BaseCar
import math

class AI(BaseCar):
    def __init__(self, pos, color, number, track, difficulty=1.0):
        super().__init__(pos, color, number)
        self.waypoints = self._generate_waypoints(track)
        self.current_waypoint = 0
        self.difficulty = difficulty
        self.max_speed *= difficulty
        self.acceleration *= difficulty
        self.target_threshold = 60

    def _generate_waypoints(self, track):
        waypoints = []
        # Calculate the middle of the track
        track_center = track.track_width // 2
        inset = track.inset + track_center

        # Generate waypoints in anticlockwise order, with more points for smoother turning
        # Bottom right corner
        waypoints.append((track.outer_rect.right - inset, track.outer_rect.bottom - inset))
        waypoints.append((track.outer_rect.right - inset, track.outer_rect.bottom - inset - track_center))

        # Top right corner
        waypoints.append((track.outer_rect.right - inset, track.outer_rect.top + inset + track_center))
        waypoints.append((track.outer_rect.right - inset, track.outer_rect.top + inset))

        # Top left corner
        waypoints.append((track.outer_rect.left + inset, track.outer_rect.top + inset))
        waypoints.append((track.outer_rect.left + inset, track.outer_rect.top + inset + track_center))

        # Bottom left corner
        waypoints.append((track.outer_rect.left + inset, track.outer_rect.bottom - inset - track_center))
        waypoints.append((track.outer_rect.left + inset, track.outer_rect.bottom - inset))

        return waypoints

    def _normalize_angle(self, angle):
        """Normalize angle to be between -180 and 180 degrees"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _calculate_steering_angle(self, target):
        """Calculate the steering angle needed to reach the target"""
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]

        # Calculate target angle in degrees
        target_angle = math.degrees(math.atan2(dy, dx))

        # Calculate the difference between target angle and current angle
        angle_diff = self._normalize_angle(target_angle - self.angle)

        return angle_diff

    def handle_input(self, game_state):
        if not game_state.race_started:
            return

        # Get current target waypoint
        target = self.waypoints[self.current_waypoint]

        # Calculate distance to current waypoint
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        distance = math.hypot(dx, dy)

        # Calculate steering angle
        angle_diff = self._calculate_steering_angle(target)

        # Steering logic
        turn_threshold = 5
        if abs(angle_diff) > turn_threshold:
            # Reduce speed when turning sharply
            self.max_speed = self.difficulty * 6
            if angle_diff > 0:
                self.turn(self.turn_speed)
            else:
                self.turn(-self.turn_speed)
        else:
            # Resume normal speed on straights
            self.max_speed = self.difficulty * 8

        # Always try to maintain forward momentum
        speed = math.hypot(*self.vel)
        if speed < self.max_speed * 0.8:
            self.accelerate()

        # Switch to next waypoint when close enough
        if distance < self.target_threshold:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)

    def draw(self, screen):
        # Draw the car
        super().draw(screen)

        # Optionally, for debugging: draw waypoints and current target

        for i, wp in enumerate(self.waypoints):
            color = (255, 0, 0) if i == self.current_waypoint else (0, 255, 0)
            pygame.draw.circle(screen, color, (int(wp[0]), int(wp[1])), 5)
