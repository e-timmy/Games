import pygame
import pymunk
import sys
from controllers.player_controller import PlayerController
from controllers.level_controller import LevelController
from controllers.camera_controller import CameraController
from systems.collision_manager import CollisionManager
from constants.game_constants import *


class GameEngine:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = (0, 981)

        self.camera_controller = CameraController(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.level_controller = LevelController(self.space)
        # Update player controller initialization with level_controller reference
        self.player_controller = PlayerController(self.space, self.level_controller)
        self.collision_manager = CollisionManager(self.space, self.player_controller,
                                                  self.level_controller)

    def run(self):
        running = True
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.player_controller.handle_event(event)

            # Update
            self.space.step(1 / 60.0)
            self.player_controller.update()

            # Check level completion
            if self.level_controller.check_level_complete(self.player_controller.player.body.position):
                self.camera_controller.start_transition()

            self.level_controller.update()
            self.camera_controller.update()

            # Render
            self.screen.fill(BLACK)
            self.level_controller.draw(self.screen, self.camera_controller)
            self.player_controller.draw(self.screen, self.camera_controller)

            if DEBUG_MODE:
                self.draw_debug()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    def draw_debug(self):
        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Segment):
                start = shape.a + shape.body.position
                end = shape.b + shape.body.position
                start_screen = self.camera_controller.apply(start.x, start.y)
                end_screen = self.camera_controller.apply(end.x, end.y)
                pygame.draw.line(self.screen, RED, start_screen, end_screen, 1)
            elif isinstance(shape, pymunk.Circle):
                pos = self.camera_controller.apply(shape.body.position.x, shape.body.position.y)
                pygame.draw.circle(self.screen, RED, (int(pos[0]), int(pos[1])), int(shape.radius), 1)