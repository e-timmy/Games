

import pymunk
import math
from constants import *
from entities import Line, RiderPart, Rider


class GameState:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)
        self.is_playing = False
        self.lines = []
        self.current_tool = "draw"
        self.line_thickness = 2
        self.current_color = BLACK
        self.rider_parts = []
        self.rider = Rider(self.space)
        self.physics_steps_per_frame = 5

        handler = self.space.add_collision_handler(COLLISION_TYPE_RIDER, COLLISION_TYPE_LINE)
        handler.begin = self.rider_line_collision
        handler.pre_solve = self.rider_line_pre_solve

    def update(self, dt):
        if self.is_playing:
            for _ in range(self.physics_steps_per_frame):
                self.space.step(dt / self.physics_steps_per_frame)

    def add_line(self, start_pos, end_pos):
        if self.current_tool == "draw":
            line = Line(start_pos, end_pos, self.space, self.line_thickness, self.current_color)
            self.lines.append(line)
        elif self.current_tool == "erase":
            self.erase_lines(end_pos, self.line_thickness)
        elif self.current_tool == "create_rider":
            self.add_rider_part(start_pos, end_pos)

    def add_rider_part(self, start_pos, end_pos):
        part = RiderPart(start_pos, end_pos, self.current_color, self.line_thickness)
        self.rider_parts.append(part)

    def finish_rider(self):
        if self.rider_parts:
            body, shapes = self.create_rider_body()
            self.rider.create_from_parts(body, shapes, self.rider_parts)
            self.rider_parts = []
            self.current_tool = "draw"
            return self.rider.get_position()
        return None

    def create_rider_body(self):
        total_mass = 1
        moment = pymunk.moment_for_box(total_mass, (50, 50))
        body = pymunk.Body(total_mass, moment)
        body.position = self.rider_parts[0].start_pos

        shapes = []
        for part in self.rider_parts:
            shape = pymunk.Segment(body,
                                   (part.start_pos[0] - body.position.x,
                                    part.start_pos[1] - body.position.y),
                                   (part.end_pos[0] - body.position.x,
                                    part.end_pos[1] - body.position.y),
                                   part.thickness / 2)
            shape.friction = 0.7
            shape.elasticity = 0.2
            shape.collision_type = COLLISION_TYPE_RIDER
            shapes.append(shape)

        return body, shapes

    def rider_line_collision(self, arbiter, space, data):
        return True

    def rider_line_pre_solve(self, arbiter, space, data):
        for contact in arbiter.contact_point_set.points:
            rider_shape, line_shape = arbiter.shapes
            rider_body = rider_shape.body
            rider_velocity = rider_body.velocity
            normal = contact.normal
            perpendicular = (-normal.y, normal.x)

            velocity_along_surface = rider_velocity.dot(perpendicular)
            new_velocity = velocity_along_surface * pymunk.Vec2d(perpendicular)

            rider_body.velocity = new_velocity
        return True

    def erase_lines(self, pos, radius):
        for line in self.lines[:]:
            if self.point_near_line(pos, line.start_pos, line.end_pos, radius):
                self.space.remove(line.shape, line.body)
                self.lines.remove(line)

    def point_near_line(self, point, line_start, line_end, radius):
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_length == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2) <= radius

        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length * line_length)))

        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)

        distance = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
        return distance <= radius

    def set_tool(self, tool):
        self.current_tool = tool

    def set_line_thickness(self, thickness):
        self.line_thickness = thickness

    def set_color(self, color):
        self.current_color = color

    def toggle_play(self):
        self.is_playing = not self.is_playing

    def clear_canvas(self):
        for line in self.lines:
            self.space.remove(line.shape, line.body)
        self.lines.clear()
        if self.rider.body:
            self.space.remove(self.rider.body, *self.rider.shapes)
        self.rider = Rider(self.space)
