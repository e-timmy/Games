import pymunk
from constants import *


class Line:
    def __init__(self, start_pos, end_pos, space, thickness=2, color=BLACK):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.thickness = thickness
        self.color = color

        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, start_pos, end_pos, thickness / 2)
        shape.friction = 0.7
        shape.elasticity = 0.5
        shape.collision_type = COLLISION_TYPE_LINE

        space.add(body, shape)
        self.body = body
        self.shape = shape

    def draw(self, renderer):
        renderer.draw_line(self.start_pos, self.end_pos, self.color, self.thickness)


class RiderPart:
    def __init__(self, start_pos, end_pos, color, thickness):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color = color
        self.thickness = thickness

    def draw(self, renderer):
        renderer.draw_line(self.start_pos, self.end_pos, self.color, self.thickness)


class Rider:
    def __init__(self, space):
        self.space = space
        self.initial_pos = (HUD_LEFT_WIDTH + 150, WINDOW_HEIGHT // 2)
        self.body = None
        self.shapes = []
        self.parts = []

    def create_from_parts(self, body, shapes, parts):
        if self.body:
            self.space.remove(self.body, *self.shapes)

        self.body = body
        self.shapes = shapes
        self.parts = parts
        for shape in self.shapes:
            shape.friction = 0.7
            shape.elasticity = 0.2
            shape.collision_type = COLLISION_TYPE_RIDER
        self.space.add(self.body, *self.shapes)

    def draw(self, renderer):
        if self.body:
            for part in self.parts:
                start = self.body.local_to_world(part.start_pos)
                end = self.body.local_to_world(part.end_pos)
                renderer.draw_line(start, end, part.color, part.thickness)

    def reset(self):
        if self.body:
            self.body.position = self.initial_pos
            self.body.velocity = (0, 0)
            self.body.angle = 0

    def get_position(self):
        return (int(self.body.position.x), int(self.body.position.y)) if self.body else self.initial_pos

