import pygame
import sys
from game_state import GameState
from ui_manager import UIManager
from camera import Camera
from renderer import Renderer
from constants import *


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.camera = Camera()
        self.renderer = Renderer(self.screen, self.camera)
        self.game_state = GameState()
        self.ui_manager = UIManager()

        self.drawing = False
        self.last_pos = None

    def run(self):
        while True:
            dt = 1.0 / 60.0

            for event in pygame.event.get():
                self.handle_event(event)

            self.update(dt)
            self.draw()

            pygame.display.flip()
            self.clock.tick(60)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            action = self.ui_manager.handle_click(event.pos)
            if action:
                if isinstance(action, tuple) and action[0] == "color":
                    self.game_state.set_color(action[1])
                elif action == "finish_rider":
                    rider_pos = self.game_state.finish_rider()
                    if rider_pos:
                        self.center_camera_on_rider(rider_pos)
                elif action in ["draw", "erase", "create_rider"]:
                    self.game_state.set_tool(action)
                elif action == "play":
                    self.game_state.toggle_play()
                elif action == "reset":
                    self.game_state.rider.reset()
                    self.center_camera_on_rider(self.game_state.rider.get_position())
                elif action == "clear":
                    self.game_state.clear_canvas()
            elif event.pos[0] > HUD_LEFT_WIDTH:
                self.drawing = True
                self.last_pos = self.camera.screen_to_world(event.pos)
            else:
                thickness = self.ui_manager.handle_slider(event.pos)
                if thickness is not None:
                    self.game_state.set_line_thickness(thickness)

        elif event.type == pygame.MOUSEBUTTONUP:
            self.drawing = False
            self.last_pos = None

        elif event.type == pygame.MOUSEMOTION:
            if self.drawing and event.pos[0] > HUD_LEFT_WIDTH:
                current_pos = self.camera.screen_to_world(event.pos)
                if self.last_pos:
                    self.game_state.add_line(self.last_pos, current_pos)
                self.last_pos = current_pos
            elif event.buttons[0] and event.pos[0] <= HUD_LEFT_WIDTH:
                thickness = self.ui_manager.handle_slider(event.pos)
                if thickness is not None:
                    self.game_state.set_line_thickness(thickness)

    def update(self, dt):
        self.game_state.update(dt)
        if self.game_state.rider.body:
            self.camera.follow(self.game_state.rider.get_position())

    def draw(self):
        self.screen.fill(WHITE)

        # Draw game elements
        for line in self.game_state.lines:
            line.draw(self.renderer)

        # Draw rider parts being created
        for part in self.game_state.rider_parts:
            part.draw(self.renderer)

        # Draw completed rider
        self.game_state.rider.draw(self.renderer)

        # Draw HUD elements
        self.renderer.draw_hud(self.ui_manager, self.game_state)

        # Draw eraser circle when using eraser
        if self.drawing and self.game_state.current_tool == "erase":
            mouse_pos = pygame.mouse.get_pos()
            world_pos = self.camera.screen_to_world(mouse_pos)
            screen_pos = self.camera.world_to_screen(world_pos)
            pygame.draw.circle(self.screen, BLACK, screen_pos, ERASER_RADIUS, 1)

    def center_camera_on_rider(self, rider_pos):
        self.camera.set_target(rider_pos)


if __name__ == "__main__":
    game = Game()
    game.run()