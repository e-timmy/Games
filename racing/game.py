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
        self.clock = pygame.time.Clock()
        self.running = True

        # Pymunk space setup
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.track = Track(self.width, self.height, self.space)
        start_x, start_y = self.track.get_starting_position()
        self.car = Car(start_x, start_y, self.space)

    def update(self):
        keys = pygame.key.get_pressed()
        self.car.handle_input(keys)
        self.space.step(1/60.0)

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.space.debug_draw(self.draw_options)
        self.car.draw_direction_indicator(self.screen)
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update()
            self.draw()
            self.clock.tick(60)