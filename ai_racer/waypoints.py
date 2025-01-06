import pygame
import math


class WaypointVisualizer:
    def __init__(self, track):
        self.track = track
        self.waypoints = self._generate_waypoints()
        self.waypoint_groups = self._group_waypoints()
        self.activated_waypoints = set()
        self.flash_colors = {}
        self.activation_radius = 50
        self.visualization_enabled = True
        self.last_activated_group = None

    def _generate_waypoints(self):
        waypoints = []
        spacing = 30
        track_width = self.track.track_width
        outer_rect = self.track.outer_rect
        inner_rect = pygame.Rect(outer_rect.left + track_width, outer_rect.top + track_width,
                                 outer_rect.width - 2 * track_width, outer_rect.height - 2 * track_width)

        for x in range(outer_rect.left, outer_rect.right, spacing):
            for y in range(outer_rect.top, outer_rect.bottom, spacing):
                if outer_rect.collidepoint(x, y) and not inner_rect.collidepoint(x, y):
                    waypoints.append((x, y))

        return waypoints

    def _group_waypoints(self):
        groups = []
        center_x, center_y = self.track.outer_rect.center
        sorted_waypoints = sorted(self.waypoints,
                                  key=lambda wp: -math.atan2(wp[0] - center_x, wp[1] - center_y))

        group_size = len(sorted_waypoints) // 100  # Adjust this value to change the number of groups
        for i in range(0, len(sorted_waypoints), group_size):
            groups.append(sorted_waypoints[i:i + group_size])

        return groups

    def update(self, car_pos):
        self.activated_waypoints.clear()
        self.flash_colors.clear()

        activated_group = None
        for group in self.waypoint_groups:
            for wp in group:
                if math.hypot(wp[0] - car_pos[0], wp[1] - car_pos[1]) < self.activation_radius:
                    activated_group = group
                    break
            if activated_group:
                break

        if activated_group and activated_group != self.last_activated_group:
            is_forward = self._is_forward_progress(self.last_activated_group, activated_group)
            flash_color = (0, 255, 0) if is_forward else (255, 0, 0)

            for wp in activated_group:
                self.activated_waypoints.add(wp)
                self.flash_colors[wp] = flash_color

            self.last_activated_group = activated_group

    def _is_forward_progress(self, last_group, current_group):
        if not last_group:
            return True
        last_index = self.waypoint_groups.index(last_group)
        current_index = self.waypoint_groups.index(current_group)
        return (current_index < last_index) or (last_index == 0 and current_index == len(self.waypoint_groups) - 1)

    def draw(self, screen):
        if not self.visualization_enabled:
            return

        for group in self.waypoint_groups:
            for wp in group:
                if wp in self.activated_waypoints:
                    color = self.flash_colors[wp]
                else:
                    color = (100, 100, 100)  # Gray for inactive waypoints
                pygame.draw.circle(screen, color, (int(wp[0]), int(wp[1])), 3)

    def reset(self):
        self.activated_waypoints.clear()
        self.flash_colors.clear()
        self.last_activated_group = None

    def get_progress(self, car_pos):
        for i, group in enumerate(self.waypoint_groups):
            for wp in group:
                if math.hypot(wp[0] - car_pos[0], wp[1] - car_pos[1]) < self.activation_radius:
                    return i / len(self.waypoint_groups)
        return 0
