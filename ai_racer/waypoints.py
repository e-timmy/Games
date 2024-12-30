import pygame
import math
from collections import deque


class WaypointVisualizer:
    def __init__(self, track):
        self.track = track
        self.waypoints = self._generate_waypoints()
        self.waypoint_groups = self._group_waypoints()
        self.good_groups = deque(self.waypoint_groups[:len(self.waypoint_groups) // 2])
        self.bad_groups = deque(self.waypoint_groups[len(self.waypoint_groups) // 2:])
        self.flash_waypoints = set()
        self.flash_colors = {}
        self.activation_radius = 100
        self.visualization_enabled = True
        self.last_activated_group = None

    def _generate_waypoints(self):
        waypoints = []
        spacing = 30
        corner_spacing = int(spacing * 1.4)
        track_center = self.track.track_width // 2
        inner_offset = track_center // 2
        outer_offset = self.track.track_width - inner_offset

        # Right side (vertical)
        for y in range(self.track.outer_rect.top + outer_offset,
                       self.track.outer_rect.bottom - outer_offset + 1,
                       spacing):
            col = []
            for x in range(self.track.outer_rect.right - outer_offset,
                           self.track.outer_rect.right - inner_offset + 1,
                           spacing):
                col.append((x, y))
            waypoints.extend(col)

        # Corners
        def generate_corner(start_x, start_y, dx, dy):
            corner_points = []
            for i in range(max(2, track_center // corner_spacing)):
                x = start_x + i * corner_spacing * dx
                y = start_y + i * corner_spacing * dy
                corner_points.append((x, y))
            return corner_points

        # Bottom-right corner
        waypoints.extend(generate_corner(
            self.track.outer_rect.right - outer_offset,
            self.track.outer_rect.bottom - outer_offset,
            -1, -1))

        # Bottom side (horizontal)
        for x in range(self.track.outer_rect.right - outer_offset,
                       self.track.outer_rect.left + outer_offset - 1,
                       -spacing):
            row = []
            for y in range(self.track.outer_rect.bottom - outer_offset,
                           self.track.outer_rect.bottom - inner_offset + 1,
                           spacing):
                row.append((x, y))
            waypoints.extend(row)

        # Bottom-left corner
        waypoints.extend(generate_corner(
            self.track.outer_rect.left + outer_offset,
            self.track.outer_rect.bottom - outer_offset,
            1, -1))

        # Left side (vertical)
        for y in range(self.track.outer_rect.bottom - outer_offset,
                       self.track.outer_rect.top + outer_offset - 1,
                       -spacing):
            col = []
            for x in range(self.track.outer_rect.left + inner_offset,
                           self.track.outer_rect.left + outer_offset + 1,
                           spacing):
                col.append((x, y))
            waypoints.extend(col)

        # Top-left corner
        waypoints.extend(generate_corner(
            self.track.outer_rect.left + outer_offset,
            self.track.outer_rect.top + outer_offset,
            1, 1))

        # Top side (horizontal)
        for x in range(self.track.outer_rect.left + outer_offset,
                       self.track.outer_rect.right - outer_offset + 1,
                       spacing):
            row = []
            for y in range(self.track.outer_rect.top + inner_offset,
                           self.track.outer_rect.top + outer_offset + 1,
                           spacing):
                row.append((x, y))
            waypoints.extend(row)

        # Top-right corner
        waypoints.extend(generate_corner(
            self.track.outer_rect.right - outer_offset,
            self.track.outer_rect.top + outer_offset,
            -1, 1))

        return waypoints

    def _group_waypoints(self):
        groups = []
        current_group = []
        last_wp = None

        def is_same_group(wp1, wp2):
            if not wp1:
                return True
            x1, y1 = wp1
            x2, y2 = wp2

            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 10:  # Vertical
                return current_group and abs(current_group[0][0] - x1) < 10
            if abs(dy) < 10:  # Horizontal
                return current_group and abs(current_group[0][1] - y1) < 10
            if abs(abs(dx) - abs(dy)) < 10:  # Diagonal
                return True
            return False

        for wp in self.waypoints:
            if not is_same_group(last_wp, wp) and current_group:
                groups.append(current_group)
                current_group = []
            current_group.append(wp)
            last_wp = wp

        if current_group:
            groups.append(current_group)

        return groups

    def update(self, car_pos):
        self.flash_waypoints.clear()

        activated_group = None
        for group in list(self.good_groups) + list(self.bad_groups):
            for wp in group:
                if math.hypot(wp[0] - car_pos[0], wp[1] - car_pos[1]) < self.activation_radius:
                    activated_group = group
                    break
            if activated_group:
                break

        if activated_group and activated_group != self.last_activated_group:
            is_good_group = activated_group in self.good_groups
            flash_color = (0, 255, 0) if is_good_group else (255, 0, 0)

            for wp in activated_group:
                self.flash_waypoints.add(wp)
                self.flash_colors[wp] = flash_color

            # Update queues
            if is_good_group:
                self.good_groups.remove(activated_group)
                self.bad_groups.append(activated_group)
                if self.bad_groups:
                    self.good_groups.append(self.bad_groups.popleft())
            else:
                self.bad_groups.remove(activated_group)
                self.good_groups.append(activated_group)
                if self.good_groups:
                    self.bad_groups.appendleft(self.good_groups.popleft())

            self.last_activated_group = activated_group

    def draw(self, screen):
        if not self.visualization_enabled:
            return

        # Draw good groups (blue)
        for group in self.good_groups:
            for wp in group:
                if wp in self.flash_waypoints:
                    color = self.flash_colors[wp]
                else:
                    color = (0, 0, 255)  # Blue
                pygame.draw.circle(screen, color, (int(wp[0]), int(wp[1])), 3)

        # Draw bad groups (yellow)
        for group in self.bad_groups:
            for wp in group:
                if wp in self.flash_waypoints:
                    color = self.flash_colors[wp]
                else:
                    color = (255, 255, 0)  # Yellow
                pygame.draw.circle(screen, color, (int(wp[0]), int(wp[1])), 3)

        # Debug: Draw group lines
        # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        #           (255, 0, 255), (0, 255, 255), (128, 128, 128)]
        # for i, group in enumerate(self.waypoint_groups):
        #     color = colors[i % len(colors)]
        #     pygame.draw.lines(screen, color, False, group, 1)

    def reset(self):
        self.flash_waypoints.clear()
        self.flash_colors.clear()
        self.last_activated_group = None
        self.good_groups = deque(self.waypoint_groups[:len(self.waypoint_groups) // 2])
        self.bad_groups = deque(self.waypoint_groups[len(self.waypoint_groups) // 2:])