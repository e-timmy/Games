from base_car import BaseCar
import pygame


class Player(BaseCar):
    def __init__(self, pos, color, number, keys):
        super().__init__(pos, color, number)
        self.keys = self._convert_key_config(keys)
        self.waypoint_visualizer = None  # This will be set by the Game class

    def set_waypoint_visualizer(self, visualizer):
        self.waypoint_visualizer = visualizer

    def handle_input(self, game_state):
        if not game_state.race_started:
            return

        keys = pygame.key.get_pressed()
        if keys[self.keys[0]]:  # UP
            self.accelerate()
        if keys[self.keys[1]]:  # DOWN
            self.decelerate()
        if keys[self.keys[2]]:  # LEFT
            self.turn(-self.turn_speed)
        if keys[self.keys[3]]:  # RIGHT
            self.turn(self.turn_speed)

        # # Update waypoint visualizer
        # if self.waypoint_visualizer:
        #     self.waypoint_visualizer.update(self.pos)

    def draw(self, screen):
        super().draw(screen)
        if self.waypoint_visualizer:
            self.waypoint_visualizer.draw(screen)

    def _convert_key_config(self, keys):
        key_mapping = {
            # Default arrow keys
            "UP": pygame.K_UP,
            "DN": pygame.K_DOWN,
            "LT": pygame.K_LEFT,
            "RT": pygame.K_RIGHT,
            # WASD
            "W": pygame.K_w,
            "S": pygame.K_s,
            "A": pygame.K_a,
            "D": pygame.K_d,
            # IJKL
            "I": pygame.K_i,
            "J": pygame.K_j,
            "K": pygame.K_k,
            "L": pygame.K_l,
            # Numpad
            "8": pygame.K_KP8,
            "5": pygame.K_KP5,
            "4": pygame.K_KP4,
            "6": pygame.K_KP6
        }

        converted_keys = []
        for key in keys:
            converted_keys.append(key_mapping.get(key.upper(), pygame.K_UP))  # Default to UP if invalid key
        return converted_keys