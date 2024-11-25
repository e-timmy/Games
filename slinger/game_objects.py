import pymunk
import pygame
import math

from constants import SCREEN_HEIGHT, SCREEN_WIDTH


class RopeSegment:
    def __init__(self, position, space, mass=1.0):
        self.body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, 2))
        self.body.position = position
        self.shape = pymunk.Circle(self.body, 2)
        self.shape.friction = 0.9
        self.shape.elasticity = 0.1
        self.shape.collision_type = 1
        space.add(self.body, self.shape)


class Rope:
    def __init__(self, player_body, start_pos, angle, space):
        self.space = space
        self.player_body = player_body
        self.angle = angle
        self.segments = []
        self.joints = []
        self.ceiling_anchor = None
        self.is_extending = True
        self.attached_to_ceiling = False
        self.player_joint = None
        self.extension_rate = 1000  # Increased from 400 to 1000
        self.segment_spacing = 5  # Decreased from 10 to 5 for smoother rope
        self.max_length = SCREEN_HEIGHT - 20

        # Create initial rope segment connected to player
        first_segment = RopeSegment(start_pos, space, mass=0.1)  # Reduced mass for less gravity effect
        self.segments.append(first_segment)

        # Create initial joint with player
        self.create_player_joint()

        # Apply initial velocity to first segment
        velocity = (math.cos(angle) * self.extension_rate,
                    math.sin(angle) * self.extension_rate)
        first_segment.body.velocity = velocity

    def create_player_joint(self):
        if self.player_joint and self.player_joint in self.space.constraints:
            self.space.remove(self.player_joint)

        last_segment = self.segments[-1]

        # Create slide joint for some elasticity
        joint = pymunk.SlideJoint(
            self.player_body, last_segment.body,
            (0, 0), (0, 0),
            0,  # Min distance
            10  # Max distance - small amount of stretch
        )
        joint.collide_bodies = False

        self.player_joint = joint
        self.space.add(joint)

    def update(self, dt):
        if self.is_extending:
            last_segment = self.segments[-1]

            # Check if we need to add a new segment
            if (len(self.segments) * self.segment_spacing < self.max_length and
                    last_segment.body.velocity.length > 10):  # Only extend if there's significant velocity

                # Create new segment at appropriate distance from last
                new_pos = last_segment.body.position + last_segment.body.velocity.normalized() * self.segment_spacing
                new_segment = RopeSegment(new_pos, self.space, mass=0.1)  # Reduced mass
                self.segments.append(new_segment)

                # Join to previous segment
                joint = pymunk.SlideJoint(
                    last_segment.body, new_segment.body,
                    (0, 0), (0, 0),
                    self.segment_spacing * 0.95,  # Min length
                    self.segment_spacing * 1.05  # Max length
                )
                self.joints.append(joint)
                self.space.add(joint)

                # Transfer velocity to new segment
                new_segment.body.velocity = last_segment.body.velocity * 0.99  # Slight velocity reduction

                # Update player joint to connect to new last segment
                self.create_player_joint()

            # Check for ceiling collision
            if not self.attached_to_ceiling:
                for segment in self.segments:
                    if segment.body.position.y <= 0:
                        self.attach_to_ceiling(segment)
                        break
        else:
            # Apply a small upward force to keep the rope from collapsing
            for segment in self.segments:
                segment.body.apply_force_at_local_point((0, -50), (0, 0))

    def attach_to_ceiling(self, segment):
        self.attached_to_ceiling = True
        self.is_extending = False

        # Create ceiling anchor
        self.ceiling_anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.ceiling_anchor.position = (segment.body.position.x, 0)

        # Create joint between anchor and first segment
        joint = pymunk.SlideJoint(
            self.ceiling_anchor, segment.body,
            (0, 0), (0, 0),
            0,  # Min length
            5  # Max length - small amount of give
        )
        self.joints.append(joint)
        self.space.add(joint)

        # Stop vertical velocity of segments
        for segment in self.segments:
            segment.body.velocity_func = self.limit_vertical_velocity

    def limit_vertical_velocity(self, body, gravity, damping, dt):
        # Custom velocity function to prevent excessive vertical movement
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        if body.velocity.y < 0:
            body.velocity = (body.velocity.x, body.velocity.y * 0.7)

    def remove(self):
        if self.player_joint and self.player_joint in self.space.constraints:
            self.space.remove(self.player_joint)

        for segment in self.segments:
            self.space.remove(segment.body, segment.shape)

        for joint in self.joints:
            self.space.remove(joint)

        if self.ceiling_anchor:
            self.space.remove(self.ceiling_anchor)

    def draw(self, screen):
        if len(self.segments) < 2:
            return

        # Collect all points including player position if attached
        points = [(segment.body.position.x, segment.body.position.y)
                  for segment in self.segments]

        if self.player_joint and self.player_joint in self.space.constraints:
            points.append((self.player_body.position.x, self.player_body.position.y))

        # Draw rope with outline effect
        pygame.draw.lines(screen, (0, 0, 0), False, points, 6)  # Outline
        pygame.draw.lines(screen, (40, 40, 40), False, points, 4)  # Inner line


class Player:
    def __init__(self, x, y):
        # Physics body setup
        self.body = pymunk.Body(5, pymunk.moment_for_box(5, (20, 20)))
        self.body.position = x, y
        self.shape = pymunk.Poly.create_box(self.body, (20, 20))
        self.shape.elasticity = 0.1
        self.shape.friction = 0.5

        # Movement attributes
        self.movement_speed = 300
        self.swing_force = 2000
        self.grounded = False

        # Rope handling
        self.rope = None
        self.holding_rope = False

    def shoot_rope(self, angle, space):
        if self.rope is None:
            self.rope = Rope(self.body, self.body.position, angle, space)
            self.holding_rope = True

    def update(self, dt):
        if self.rope:
            self.rope.update(dt)

        # Update grounded state
        self.grounded = abs(self.body.velocity.y) < 0.1

    def move_left(self):
        if self.holding_rope:
            self.body.apply_force_at_local_point((-self.swing_force, 0), (0, 0))
        else:
            self.body.velocity = (-self.movement_speed, self.body.velocity.y)

    def move_right(self):
        if self.holding_rope:
            self.body.apply_force_at_local_point((self.swing_force, 0), (0, 0))
        else:
            self.body.velocity = (self.movement_speed, self.body.velocity.y)

    def stop_horizontal_movement(self):
        if not self.holding_rope:
            self.body.velocity = (0, self.body.velocity.y)

    def jump(self):
        if self.grounded:
            jump_velocity = -400
            self.body.velocity = (self.body.velocity.x, jump_velocity)
            self.grounded = False

    def remove_rope(self):
        if self.rope:
            self.rope.remove()
            self.rope = None
            self.holding_rope = False


class Platform:
    def __init__(self, x, y, width, height):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = x + width / 2, y + height / 2
        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.elasticity = 0.1
        self.shape.friction = 0.5