import pygame


class CountdownLights:
    def __init__(self, track):
        self.lights = []
        self.setup_lights(track)
        self.lights_off_timer = 0
        self.lights_off_delay = 180  # 3 seconds at 60 FPS

    def setup_lights(self, track):
        start_line_center = track.start_line.centerx
        y_position = track.start_line.top - 20
        light_spacing = 20
        light_radius = 8

        for i in range(4):
            x = start_line_center + (i - 1.5) * light_spacing
            self.lights.append({
                'pos': (x, y_position),
                'radius': light_radius,
                'color': (255, 0, 0) if i < 3 else (0, 255, 0)
            })

    def draw(self, screen, game_state):
        if game_state.race_started:
            self.lights_off_timer += 1
            if self.lights_off_timer >= self.lights_off_delay:
                return  # Don't draw lights after delay

        for i, light in enumerate(self.lights):
            color = light['color'] if i < game_state.countdown_state else (100, 100, 100)
            pygame.draw.circle(screen, color, light['pos'], light['radius'])
            pygame.draw.circle(screen, (0, 0, 0), light['pos'], light['radius'], 2)