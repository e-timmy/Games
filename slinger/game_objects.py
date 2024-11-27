import math
import pygame
import pymunk
from constants import SCREEN_HEIGHT, SCREEN_WIDTH

class RopeSegment:
    def __init__(self, pos, space, radius=3):
        self.radius = radius
        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = pos

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = 0.5
        self.shape.elasticity = 0.1
        self.shape.collision_type = 2
        self.body.damping = 0.9

        space.add(self.body, self.shape)

class Rope:
    def __init__(self, space, anchor_pos, length):
        self.space = space
        self.segments = []
        self.constraints = []
        self.segment_length = 20
        self.num_segments = 25

        self.anchor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_body.position = anchor_pos

        prev_segment = None
        for i in range(self.num_segments):
            y_offset = i * self.segment_length
            pos = (anchor_pos[0], anchor_pos[1] + y_offset)

            radius = 4 if i == self.num_segments - 1 else 2.5
            segment = RopeSegment(pos, space, radius)

            if i < self.num_segments // 3:
                segment.body.mass = 2.0

            self.segments.append(segment)

            if i == 0:
                pivot = pymunk.PivotJoint(self.anchor_body, segment.body, anchor_pos)
                pivot.max_force = 50000.0
                self.space.add(pivot)
                self.constraints.append(pivot)

                pin = pymunk.PinJoint(self.anchor_body, segment.body, (0, 0), (0, 0))
                pin.max_force = 50000.0
                self.space.add(pin)
                self.constraints.append(pin)
            else:
                pivot = pymunk.PivotJoint(
                    prev_segment.body,
                    segment.body,
                    (0, self.segment_length / 2),
                    (0, -self.segment_length / 2)
                )
                pivot.max_force = 50000.0
                self.space.add(pivot)
                self.constraints.append(pivot)

                pin = pymunk.PinJoint(
                    prev_segment.body,
                    segment.body,
                    (0, self.segment_length / 2),
                    (0, -self.segment_length / 2)
                )
                pin.max_force = 50000.0
                self.space.add(pin)
                self.constraints.append(pin)

            prev_segment = segment

        self.grabbed_segment = None
        self.grab_constraint = None

    def grab(self, player_body, grab_point):
        min_distance = float('inf')
        nearest_segment = None

        for segment in self.segments:
            distance = ((segment.body.position.x - grab_point[0]) ** 2 +
                        (segment.body.position.y - grab_point[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_segment = segment

        if nearest_segment and min_distance < 30:
            self.grabbed_segment = nearest_segment
            self.grab_constraint = pymunk.PinJoint(
                player_body,
                self.grabbed_segment.body,
                (0, 0),
                (0, 0)
            )
            self.grab_constraint.max_force = 15000.0
            self.space.add(self.grab_constraint)
            print(f"Rope grabbed at segment {self.segments.index(nearest_segment)}")
        else:
            print("Failed to grab rope")

    def release(self):
        if self.grab_constraint:
            self.space.remove(self.grab_constraint)
            self.grab_constraint = None
            print("Rope released")
        self.grabbed_segment = None

    def draw(self, screen):
        if not self.segments:
            return

        points = []
        ceiling_y = 10
        points.append((int(self.anchor_body.position.x), ceiling_y))

        for segment in self.segments:
            pos = segment.body.position
            if pos.y > ceiling_y:
                points.append((int(pos.x), int(pos.y)))

        if len(points) < 2:
            return

        pygame.draw.lines(screen, (50, 50, 50), False, points, 5)
        pygame.draw.lines(screen, (120, 120, 120), False, points, 3)

        for point in points[1:]:
            pygame.draw.circle(screen, (90, 90, 90), point, 2)

class Platform:
    def __init__(self, x, y, width, height):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = x + width / 2, y + height / 2
        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.elasticity = 0.1
        self.shape.friction = 0.5