import pymunk
import math


class Player:
    def __init__(self, x, y, size, scale_factor=1.0):
        self.scale_factor = scale_factor

        # Physics body setup
        self.size = size  # Now passed as parameter
        mass = 1.0 * scale_factor  # Reduced base mass
        moment = pymunk.moment_for_box(mass, (self.size, self.size))
        self.body = pymunk.Body(mass, moment)
        self.body.position = x, y

        self.shape = pymunk.Poly.create_box(self.body, (self.size, self.size))
        self.shape.elasticity = 0.1
        self.shape.friction = 0.5

        # Adjusted movement parameters
        base_movement_speed = 200  # Reduced base speed
        base_jump_velocity = 300  # Reduced base jump

        self.movement_speed = base_movement_speed * scale_factor
        self.jump_velocity = base_jump_velocity * scale_factor
        self.swing_force = 3000 * scale_factor  # Reduced swing force
        self.max_swing_speed = 400 * scale_factor  # Reduced max swing speed

        self.grounded = False
        self.holding_rope = False
        self.rope_anchor = None
        self.grab_constraint = None
        self.current_rope = None

    def update(self, dt):
        self.grounded = abs(self.body.velocity.y) < 0.1

    def move_left(self):
        if self.holding_rope:
            self.apply_swing_force(-1)
        else:
            self.body.velocity = (-self.movement_speed, self.body.velocity.y)

    def move_right(self):
        if self.holding_rope:
            self.apply_swing_force(1)
        else:
            self.body.velocity = (self.movement_speed, self.body.velocity.y)

    def apply_swing_force(self, direction):
        if not self.rope_anchor:
            return

        dx = self.body.position.x - self.rope_anchor.x
        dy = self.body.position.y - self.rope_anchor.y
        angle = math.atan2(dy, dx)

        force_angle = angle + math.pi / 2 * (-direction)
        force_x = math.cos(force_angle) * self.swing_force
        force_y = math.sin(force_angle) * self.swing_force

        self.body.apply_force_at_local_point((force_x, force_y), (0, 0))

        current_speed = self.body.velocity.length
        if current_speed > self.max_swing_speed:
            scale = self.max_swing_speed / current_speed
            self.body.velocity *= scale

    def stop_horizontal_movement(self):
        if not self.holding_rope:
            self.body.velocity = (0, self.body.velocity.y)

    def jump(self):
        if self.grounded:
            self.body.velocity = (self.body.velocity.x, -self.jump_velocity)
            self.grounded = False

    def try_grab_rope(self, ropes):
        if self.holding_rope:
            return

        nearest_rope = None
        min_distance = float('inf')
        nearest_segment = None

        for rope in ropes:
            for segment in rope.segments:
                distance = (segment.body.position - self.body.position).length
                if distance < min_distance and distance < 30 * self.scale_factor:
                    min_distance = distance
                    nearest_rope = rope
                    nearest_segment = segment

        if nearest_rope and nearest_segment:
            self.holding_rope = True
            self.current_rope = nearest_rope
            self.rope_anchor = nearest_rope.anchor_body.position
            self.grab_constraint = pymunk.PinJoint(
                self.body,
                nearest_segment.body,
                (0, 0),
                (0, 0)
            )
            self.grab_constraint.max_force = 50000.0 * self.scale_factor
            nearest_rope.space.add(self.grab_constraint)

    def release_rope(self):
        if self.holding_rope and self.current_rope:
            self.current_rope.space.remove(self.grab_constraint)
            self.grab_constraint = None
            self.holding_rope = False
            self.rope_anchor = None
            self.current_rope = None