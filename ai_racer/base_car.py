import pygame
import math


class BaseCar:
    def __init__(self, pos, color, number):
        self.pos = list(pos)
        self.vel = [0, 0]
        self.angle = 0
        self.color = color
        self.number = number
        self.size = (30, 15)
        self.radius = math.hypot(*self.size) / 2
        self.max_speed = 8
        self.acceleration = 0.3
        self.turn_speed = 5
        self.friction = 0.97
        self.laps = 0
        self.last_checkpoint = False
        self.race_started = False  # Add this flag
        self.has_crossed_first = False
        self.checkpoints_passed = set()
        self.next_checkpoint = 0
        self.lap_valid = False
        self.is_player = number == "1"
        self.current_checkpoint = None
        if self.is_player:
            print("Player starting - next checkpoint: 0")


    def accelerate(self):
        self.vel[0] += math.cos(math.radians(self.angle)) * self.acceleration
        self.vel[1] += math.sin(math.radians(self.angle)) * self.acceleration
        self.limit_speed()

    def decelerate(self):
        self.vel[0] -= math.cos(math.radians(self.angle)) * self.acceleration / 2
        self.vel[1] -= math.sin(math.radians(self.angle)) * self.acceleration / 2
        self.limit_speed()

    def turn(self, angle):
        speed = math.hypot(*self.vel)
        if speed > 0.5:
            self.angle += angle * (speed / self.max_speed) * 1.8

    def limit_speed(self):
        speed = math.hypot(*self.vel)
        if speed > self.max_speed:
            self.vel = [v / speed * self.max_speed for v in self.vel]

    def reset_checkpoint_progress(self):
        self.checkpoints_passed.clear()
        self.next_checkpoint = 0
        self.lap_valid = False
        if self.is_player:
            print("Progress reset - starting over from checkpoint 0")

    def update(self, track):
        # Update position
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        # Apply friction
        self.vel = [v * self.friction for v in self.vel]

        # Check for track boundaries
        reflection = track.get_reflection_vector(self.pos, self.radius)
        if reflection != (0, 0):
            self.vel = [-v * abs(r) if r != 0 else v * 0.5 for v, r in zip(self.vel, reflection)]
            self.pos[0] += reflection[0] * 5
            self.pos[1] += reflection[1] * 5

        # Check for checkpoint crossings
        checkpoint_hit = None
        for i, checkpoint in enumerate(track.checkpoints):
            if checkpoint.collidepoint(self.pos):
                checkpoint_hit = i
                break

        # Handle checkpoint logic
        if checkpoint_hit is not None:
            if checkpoint_hit != self.current_checkpoint:  # Only process if it's a new checkpoint
                self.current_checkpoint = checkpoint_hit
                if checkpoint_hit == self.next_checkpoint:
                    # Correct checkpoint hit
                    self.checkpoints_passed.add(checkpoint_hit)
                    self.next_checkpoint = (checkpoint_hit + 1) % len(track.checkpoints)
                    self.lap_valid = len(self.checkpoints_passed) == len(track.checkpoints)
                    if self.is_player:
                        print(f"Checkpoint {checkpoint_hit} crossed! Next checkpoint: {self.next_checkpoint}")
                        print(f"Checkpoints passed: {self.checkpoints_passed}")
                        print(f"Lap valid: {self.lap_valid}")
                elif checkpoint_hit not in self.checkpoints_passed:
                    # Only reset if hitting an upcoming checkpoint out of order
                    if self.is_player:
                        print(f"Wrong checkpoint! Hit {checkpoint_hit} but expected {self.next_checkpoint}")
                    self.reset_checkpoint_progress()
                # If hitting an already-passed checkpoint, do nothing
        else:
            self.current_checkpoint = None

        # Check for lap completion
        if track.start_line.collidepoint(self.pos):
            if not self.last_checkpoint:
                if self.is_player:
                    print(f"Crossing start line. Lap valid: {self.lap_valid}")
                if self.lap_valid:
                    if self.has_crossed_first:
                        self.laps += 1
                        if self.is_player:
                            print(f"LAP COMPLETED! New lap count: {self.laps}")
                    elif self.is_player:
                        print("First crossing of start line - lap not counted yet")
                    self.reset_checkpoint_progress()
                elif self.is_player:
                    print("Crossed start line but lap not valid - checkpoints missing")
                self.last_checkpoint = True
        else:
            if self.last_checkpoint:
                self.has_crossed_first = True
                if self.is_player:
                    print("Left start line area")
            self.last_checkpoint = False

    def draw(self, screen):
        rotated_surface = pygame.Surface((self.size[0] + 10, self.size[1] + 10), pygame.SRCALPHA)
        pygame.draw.rect(rotated_surface, self.color, (5, 5, *self.size))

        font = pygame.font.Font(None, 20)
        text = font.render(self.number, True, (255, 255, 255))
        text_rect = text.get_rect(center=(rotated_surface.get_width() // 2, rotated_surface.get_height() // 2))
        rotated_surface.blit(text, text_rect)

        rotated_surface = pygame.transform.rotate(rotated_surface, -self.angle)
        rect = rotated_surface.get_rect(center=self.pos)
        screen.blit(rotated_surface, rect)