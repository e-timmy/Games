import pygame
import pymunk
import math
import time


class Car:
    def __init__(self, x, y, space):
        # Physical properties
        self.width = 20  # Reduced size
        self.height = 40  # Reduced size
        mass = 1
        moment = pymunk.moment_for_box(mass, (self.width, self.height))
        self.body = pymunk.Body(mass, moment)
        self.body.position = x, y
        self.body.angle = 0  # Start facing right (we'll adjust the graphics instead)
        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))
        self.shape.elasticity = 0.1
        self.shape.friction = 0.7
        space.add(self.body, self.shape)

        # Movement properties
        self.acceleration = 2000
        self.turn_speed = 3.0
        self.max_speed = 400

        # Animation properties
        self.bounce_offset = 0
        self.bounce_speed = 0.1
        self.bounce_amplitude = 2  # Reduced bounce
        self.last_update = time.time()
        self.exhaust_particles = []
        self.particle_timer = 0

        # Create car graphics
        self.create_car_surface()

    def create_car_surface(self):
        # Create the main surface - reduced size
        surface_width = 80  # Halved from previous
        surface_height = 60  # Halved from previous
        self.car_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)

        # Calculate center point
        center_x = surface_width // 2
        center_y = surface_height // 2

        # Draw shadow (ellipse) - reduced size
        shadow_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, (0, 0, 0, 76), (center_x - 20, center_y - 20, 40, 40))
        self.car_surface.blit(shadow_surface, (0, 0))

        # Draw car body - reduced size
        car_body_points = [
            (center_x - 20, center_y - 12),  # Left top
            (center_x + 20, center_y - 12),  # Right top
            (center_x + 20, center_y + 12),  # Right bottom
            (center_x - 20, center_y + 12),  # Left bottom
        ]
        pygame.draw.polygon(self.car_surface, (52, 152, 219), car_body_points)  # Main body color
        pygame.draw.polygon(self.car_surface, (41, 128, 185), car_body_points, 2)  # Outline

        # Draw windows
        window_color = (236, 240, 241)
        pygame.draw.polygon(self.car_surface, window_color, [
            (center_x - 10, center_y - 10),
            (center_x + 10, center_y - 10),
            (center_x + 7, center_y - 2),
            (center_x - 7, center_y - 2),
        ])
        pygame.draw.polygon(self.car_surface, window_color, [
            (center_x - 7, center_y + 2),
            (center_x + 7, center_y + 2),
            (center_x + 10, center_y + 10),
            (center_x - 10, center_y + 10),
        ])

        # Draw headlights - reduced size
        self.draw_light(self.car_surface, center_x + 17, center_y - 7, (255, 255, 0), 4)  # Front lights
        self.draw_light(self.car_surface, center_x + 17, center_y + 7, (255, 255, 0), 4)

        # Draw taillights - reduced size
        self.draw_light(self.car_surface, center_x - 17, center_y - 7, (255, 0, 0), 3)  # Rear lights
        self.draw_light(self.car_surface, center_x - 17, center_y + 7, (255, 0, 0), 3)

        self.original_car_surface = self.car_surface.copy()

        # Create surfaces for light animations
        self.create_light_surfaces()

    def draw_light(self, surface, x, y, color, radius):
        # Draw the main light
        pygame.draw.circle(surface, color, (int(x), int(y)), radius)

        # Draw a smaller white center for realism
        pygame.draw.circle(surface, (255, 255, 255), (int(x), int(y)), radius // 2)

    def create_light_surfaces(self):
        # Create pulsing light overlay surfaces
        self.headlight_overlay = pygame.Surface((80, 60), pygame.SRCALPHA)
        self.taillight_overlay = pygame.Surface((80, 60), pygame.SRCALPHA)

        center_x = 40
        center_y = 30

        # Headlights glow
        for pos in [(center_x + 17, center_y - 7), (center_x + 17, center_y + 7)]:
            pygame.draw.circle(self.headlight_overlay, (255, 255, 0, 100), pos, 6)

        # Taillights glow
        for pos in [(center_x - 17, center_y - 7), (center_x - 17, center_y + 7)]:
            pygame.draw.circle(self.taillight_overlay, (255, 0, 0, 100), pos, 5)

    def update(self):
        # Update bounce animation
        current_time = time.time()
        dt = current_time - self.last_update
        self.bounce_offset = math.sin(current_time * self.bounce_speed) * self.bounce_amplitude
        self.last_update = current_time

        # Update exhaust particles
        self.particle_timer += dt
        if self.particle_timer >= 0.1:  # Create new particle every 0.1 seconds
            self.particle_timer = 0
            self.exhaust_particles.append({
                'pos': list(self.body.position),
                'life': 1.0,
                'size': 5
            })

        # Update existing particles
        for particle in self.exhaust_particles[:]:
            particle['life'] -= dt
            particle['size'] -= dt * 2
            if particle['life'] <= 0 or particle['size'] <= 0:
                self.exhaust_particles.remove(particle)

    def handle_input(self, keys):
        # Get the current velocity magnitude
        current_speed = self.body.velocity.length

        # Apply driving force
        if keys[pygame.K_UP]:
            driving_force = self.acceleration
            fx = math.cos(self.body.angle) * driving_force
            fy = math.sin(self.body.angle) * driving_force
            self.body.apply_force_at_world_point((fx, fy), self.body.position)

        if keys[pygame.K_DOWN]:
            driving_force = -self.acceleration * 0.5  # Less powerful in reverse
            fx = math.cos(self.body.angle) * driving_force
            fy = math.sin(self.body.angle) * driving_force
            self.body.apply_force_at_world_point((fx, fy), self.body.position)

        # Apply steering
        if keys[pygame.K_LEFT]:
            self.body.angle -= math.radians(self.turn_speed)
        if keys[pygame.K_RIGHT]:
            self.body.angle += math.radians(self.turn_speed)

        # Apply drag force (air resistance)
        if current_speed > 0:
            drag = 0.001 * current_speed * current_speed
            velocity_direction = self.body.velocity.normalized()
            drag_force = -velocity_direction * drag
            self.body.apply_force_at_world_point((drag_force.x, drag_force.y), self.body.position)

        # Speed limiting
        if current_speed > self.max_speed:
            scale = self.max_speed / current_speed
            self.body.velocity = (self.body.velocity.x * scale, self.body.velocity.y * scale)

    def draw(self, screen):
        # Get the car's position and angle
        pos = self.body.position
        angle_degrees = math.degrees(self.body.angle) + 90

        # Draw exhaust particles
        for particle in self.exhaust_particles:
            alpha = int(255 * particle['life'])
            color = (128, 128, 128, alpha)
            particle_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
            pygame.draw.circle(particle_surface, color, (5, 5), particle['size'])
            particle_rect = particle_surface.get_rect()
            particle_rect.center = (int(particle['pos'][0]), int(particle['pos'][1]))
            screen.blit(particle_surface, particle_rect)

        # Create a copy of the car surface for this frame
        current_surface = self.original_car_surface.copy()

        # Add light overlays with current animation state
        light_alpha = int(128 + 128 * math.sin(time.time() * 5))  # Pulsing effect
        headlight_overlay = self.headlight_overlay.copy()
        taillight_overlay = self.taillight_overlay.copy()
        headlight_overlay.set_alpha(light_alpha)
        taillight_overlay.set_alpha(light_alpha)
        current_surface.blit(headlight_overlay, (0, 0))
        current_surface.blit(taillight_overlay, (0, 0))

        # Rotate the car surface
        rotated_surface = pygame.transform.rotate(current_surface, angle_degrees)

        # Get the new rect and adjust for the bounce offset
        rect = rotated_surface.get_rect()
        rect.center = (int(pos.x), int(pos.y + self.bounce_offset))

        # Draw the car
        screen.blit(rotated_surface, rect)