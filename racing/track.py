import pymunk


class Track:
    def __init__(self, width, height, space):
        self.width = width
        self.height = height
        self.margin = 50
        self.track_width = 100
        self.corner_radius = 150

        # Track dimensions
        self.outer_left = self.margin
        self.outer_right = self.width - self.margin
        self.outer_top = self.margin
        self.outer_bottom = self.height - self.margin

        self.inner_left = self.margin + self.track_width
        self.inner_right = self.width - (self.margin + self.track_width)
        self.inner_top = self.margin + self.track_width
        self.inner_bottom = self.height - (self.margin + self.track_width)

        self.create_boundaries(space)

    def create_boundaries(self, space):
        static_body = space.static_body

        # Outer boundaries
        outer_segments = [
            ((self.outer_left, self.outer_top), (self.outer_right, self.outer_top)),
            ((self.outer_right, self.outer_top), (self.outer_right, self.outer_bottom)),
            ((self.outer_right, self.outer_bottom), (self.outer_left, self.outer_bottom)),
            ((self.outer_left, self.outer_bottom), (self.outer_left, self.outer_top))
        ]

        # Inner boundaries
        inner_segments = [
            ((self.inner_left, self.inner_top), (self.inner_right, self.inner_top)),
            ((self.inner_right, self.inner_top), (self.inner_right, self.inner_bottom)),
            ((self.inner_right, self.inner_bottom), (self.inner_left, self.inner_bottom)),
            ((self.inner_left, self.inner_bottom), (self.inner_left, self.inner_top))
        ]

        for segment in outer_segments + inner_segments:
            shape = pymunk.Segment(static_body, segment[0], segment[1], 1)
            shape.elasticity = 0.5
            shape.friction = 0.5
            space.add(shape)

    def get_starting_position(self):
        return (self.outer_left + self.track_width // 2, self.height // 2)