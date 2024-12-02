import pygame
import pymunk
import math
import random

class Bullet:
    def __init__(self, start_pos, target_pos, space, scale_factor=1.0):
        self.radius = 3 * scale_factor
        self.position = pymunk.Vec2d(*start_pos)
        self.scale_factor = scale_factor

        direction = (pymunk.Vec2d(*target_pos) - self.position).normalized()
        self.velocity = direction * (500 * scale_factor)

        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = self.position
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.sensor = True
        self.shape.collision_type = 3

        space.add(self.body, self.shape)

        self.explosion_particles = []
        self.exploded = False
        self.explosion_timer = 0

    def update(self, dt, max_height):
        if not self.exploded:
            self.position += self.velocity * dt
            self.body.position = self.position
            return self.position.y > max_height
        else:
            self.explosion_timer -= dt
            for particle in self.explosion_particles[:]:
                particle['pos'] = (
                    particle['pos'][0] + particle['velocity'][0] * dt,
                    particle['pos'][1] + particle['velocity'][1] * dt
                )
                particle['timer'] -= dt
                if particle['timer'] <= 0:
                    self.explosion_particles.remove(particle)
            return self.explosion_timer <= 0

    def draw(self, screen, draw_scale):
        if not self.exploded:
            screen_pos = (int(self.position.x * draw_scale), int(self.position.y * draw_scale))
            screen_radius = int(self.radius * draw_scale)
            pygame.draw.circle(screen, (255, 0, 0), screen_pos, screen_radius)
        else:
            for particle in self.explosion_particles:
                screen_pos = (int(particle['pos'][0] * draw_scale),
                              int(particle['pos'][1] * draw_scale))
                pygame.draw.circle(screen, (255, 165, 0), screen_pos,
                                  int(2 * draw_scale))

    def explode(self):
        self.exploded = True
        self.explosion_timer = 0.5
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150) * self.scale_factor
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.explosion_particles.append({
                'pos': self.position,
                'velocity': velocity,
                'timer': 0.5
            })


class RopeSegment:
    def __init__(self, pos, space, radius, scale_factor):
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
    def __init__(self, space, anchor_pos, length, scale_factor, unfurling=False):
        self.space = space
        self.segments = []
        self.constraints = []
        self.scale_factor = scale_factor
        self.segment_length = 20 * scale_factor
        self.num_segments = int(length // self.segment_length)

        self.anchor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_body.position = anchor_pos

        self.unfurling = unfurling
        self.visual_segments = 0
        self.unfurl_speed = 0.5

        self.create_segments()

    def create_segments(self):
        prev_segment = None
        for i in range(self.num_segments):
            y_offset = i * self.segment_length
            pos = (self.anchor_body.position.x, self.anchor_body.position.y + y_offset)

            radius = 4 * self.scale_factor if i == self.num_segments - 1 else 2.5 * self.scale_factor
            segment = RopeSegment(pos, self.space, radius, self.scale_factor)

            if i < self.num_segments // 3:
                segment.body.mass = 2.0

            self.segments.append(segment)

            if i == 0:
                pivot = pymunk.PivotJoint(self.anchor_body, segment.body, self.anchor_body.position)
                pivot.max_force = 50000.0 * self.scale_factor
                self.space.add(pivot)
                self.constraints.append(pivot)

                pin = pymunk.PinJoint(self.anchor_body, segment.body, (0, 0), (0, 0))
                pin.max_force = 50000.0 * self.scale_factor
                self.space.add(pin)
                self.constraints.append(pin)
            else:
                pivot = pymunk.PivotJoint(
                    prev_segment.body,
                    segment.body,
                    (0, self.segment_length / 2),
                    (0, -self.segment_length / 2)
                )
                pivot.max_force = 50000.0 * self.scale_factor
                self.space.add(pivot)
                self.constraints.append(pivot)

                pin = pymunk.PinJoint(
                    prev_segment.body,
                    segment.body,
                    (0, self.segment_length / 2),
                    (0, -self.segment_length / 2)
                )
                pin.max_force = 50000.0 * self.scale_factor
                self.space.add(pin)
                self.constraints.append(pin)

            prev_segment = segment

        self.visual_segments = self.num_segments if not self.unfurling else 0

    def update(self, dt):
        if self.unfurling and self.visual_segments < self.num_segments:
            self.visual_segments += self.unfurl_speed
            self.visual_segments = min(self.visual_segments, self.num_segments)

    def draw(self, screen, draw_scale):
        if not self.segments:
            return

        points = []
        visible_segments = int(self.visual_segments)

        anchor_screen_pos = (int(self.anchor_body.position.x * draw_scale),
                            int(self.anchor_body.position.y * draw_scale))
        points.append(anchor_screen_pos)

        coil_radius = max(5, (self.num_segments - visible_segments)) * draw_scale

        for i, segment in enumerate(self.segments[:visible_segments]):
            screen_pos = (int(segment.body.position.x * draw_scale),
                         int(segment.body.position.y * draw_scale))
            points.append(screen_pos)

        if self.num_segments - visible_segments > 0:
            pygame.draw.circle(screen, (50, 50, 50), anchor_screen_pos, int(coil_radius))
            pygame.draw.circle(screen, (120, 120, 120), anchor_screen_pos, int(coil_radius - 2))

        if len(points) >= 2:
            pygame.draw.lines(screen, (50, 50, 50), False, points, int(5 * draw_scale))
            pygame.draw.lines(screen, (120, 120, 120), False, points, int(3 * draw_scale))

            for point in points[1:]:
                pygame.draw.circle(screen, (90, 90, 90), point, int(2 * draw_scale))


class Platform:
    def __init__(self, x, y, width, height):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = x + width / 2, y + height / 2
        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.elasticity = 0.1
        self.shape.friction = 0.5
        self.shape.collision_type = 1  # Wall collision type