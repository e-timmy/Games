import pygame
import pymunk
import pymunk.pygame_util
from car import Car
from track import Track


class Game:
    def __init__(self):
        self.width = 1024
        self.height = 768
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Retro Racer")
        self.clock = pygame.time.Clock()
        self.running = True

        # Pymunk space setup
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Initialize game objects
        self.track = Track(self.width, self.height, self.space)
        start_x, start_y = self.track.get_starting_position()
        self.car = Car(start_x, start_y, self.space)

        # Visual settings
        self.background_color = (0, 100, 0)  # Dark green
        self.track_color = (50, 50, 50)  # Dark gray

    def update(self):
        keys = pygame.key.get_pressed()
        self.car.handle_input(keys)
        self.space.step(1 / 60.0)
        self.car.update()  # Added to update car animations

    def draw(self):
        # Clear screen with background color
        self.screen.fill(self.background_color)

        # Draw track (grass and road)
        pygame.draw.rect(self.screen, self.track_color,
                         (self.track.outer_left, self.track.outer_top,
                          self.track.outer_right - self.track.outer_left,
                          self.track.outer_bottom - self.track.outer_top))

        pygame.draw.rect(self.screen, self.background_color,
                         (self.track.inner_left, self.track.inner_top,
                          self.track.inner_right - self.track.inner_left,
                          self.track.inner_bottom - self.track.inner_top))

        # Draw track boundaries
        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Segment):
                pygame.draw.line(self.screen, (255, 255, 255),
                                 shape.a, shape.b, 2)

        # Draw the car with its new visuals
        self.car.draw(self.screen)

        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            self.update()
            self.draw()
            self.clock.tick(60)