import pygame
import pymunk
import math

class Player:
    def __init__(self, x, y):
        self.body = pymunk.Body(3, pymunk.moment_for_box(3, (20, 20)))
        self.body.position = x, y
        self.shape = pymunk.Poly.create_box(self.body, (20, 20))
        self.shape.elasticity = 0.1
        self.shape.friction = 0.5

        self.movement_speed = 300
        self.swing_force = 5000  # Increased swing force for more impact
        self.max_swing_speed = 600
        self.grounded = False

        self.holding_rope = False
        self.rope_anchor = None
        self.grab_constraint = None

    def update(self, dt):
        self.grounded = abs(self.body.velocity.y) < 0.1
        if self.holding_rope:
            print(f"Player position: {self.body.position}, velocity: {self.body.velocity}")

    def move_left(self):
        if self.holding_rope:
            self.apply_swing_force(-1)
        else:
            self.body.velocity = (-self.movement_speed, self.body.velocity.y)
        print(f"Move left: velocity = {self.body.velocity}")

    def move_right(self):
        if self.holding_rope:
            self.apply_swing_force(1)
        else:
            self.body.velocity = (self.movement_speed, self.body.velocity.y)
        print(f"Move right: velocity = {self.body.velocity}")

    def apply_swing_force(self, direction):
        if not self.rope_anchor:
            return

        dx = self.body.position.x - self.rope_anchor.x
        dy = self.body.position.y - self.rope_anchor.y
        angle = math.atan2(dy, dx)

        # Apply force perpendicular to the rope
        # Note the change here: -direction instead of direction
        force_angle = angle + math.pi / 2 * (-direction)
        force_x = math.cos(force_angle) * self.swing_force
        force_y = math.sin(force_angle) * self.swing_force

        self.body.apply_force_at_local_point((force_x, force_y), (0, 0))

        # Limit maximum swing speed
        current_speed = self.body.velocity.length
        if current_speed > self.max_swing_speed:
            scale = self.max_swing_speed / current_speed
            self.body.velocity *= scale

        print(f"Swing force applied: direction={direction}, force=({force_x}, {force_y})")
        print(f"Current speed: {current_speed}, Velocity after swing: {self.body.velocity}")

    def stop_horizontal_movement(self):
        if not self.holding_rope:
            self.body.velocity = (0, self.body.velocity.y)
        print(f"Stop horizontal movement: velocity = {self.body.velocity}")

    def jump(self):
        if self.grounded:
            jump_velocity = -400
            self.body.velocity = (self.body.velocity.x, jump_velocity)
            self.grounded = False
        print(f"Jump: velocity = {self.body.velocity}")

    def grab_rope(self, rope):
        nearest_segment = min(rope.segments, key=lambda seg: (seg.body.position - self.body.position).length)
        distance = (nearest_segment.body.position - self.body.position).length
        if distance < 30:
            self.holding_rope = True
            self.rope_anchor = rope.anchor_body.position
            self.grab_constraint = pymunk.PivotJoint(self.body, nearest_segment.body, (0, 0), (0, 0))
            self.grab_constraint.max_force = 50000.0  # Increased for more responsive swinging
            rope.space.add(self.grab_constraint)
            print(f"Rope grabbed. Distance: {distance}, Segment: {rope.segments.index(nearest_segment)}")
        else:
            print(f"Failed to grab rope. Distance: {distance}")

    def release_rope(self, rope):
        if self.holding_rope:
            rope.space.remove(self.grab_constraint)
            self.grab_constraint = None
            self.holding_rope = False
            self.rope_anchor = None
            print("Rope released")
            # Maintain the current velocity upon release for a more natural transition
            print(f"Release velocity: {self.body.velocity}")
        else:
            print("No rope to release")