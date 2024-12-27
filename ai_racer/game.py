import pygame
from player import Player
from ai import AI
from ui import UI
from track import Track
from game_state import GameState
from countdown_lights import CountdownLights
from victory_screen import VictoryScreen


class Game:
    def __init__(self, screen, player_configs, target_laps=0):
        self.screen = screen
        self.track = Track(screen.get_width(), screen.get_height())
        self.game_state = GameState(screen.get_width(), screen.get_height(), self.track)
        self.countdown_lights = CountdownLights(self.track)
        self.target_laps = target_laps
        self.game_finished = False

        # Initialize cars based on configurations
        self.cars = []

        for i, config in enumerate(player_configs):
            if i >= len(self.track.start_positions):
                break

            pos = self.track.start_positions[i]

            if config.player_type == "Human":
                keys = ["UP", "DN", "LT", "RT"] if config.control_scheme == "arrows" else ["W", "S", "A", "D"]
                car = Player(pos, self.get_player_color(i), str(i + 1), keys)
            else:
                car = AI(pos, self.get_player_color(i), str(i + 1),
                         self.track, difficulty=config.difficulty)

            self.cars.append(car)

        self.ui = UI(screen)
        self.clock = pygame.time.Clock()

        self.victory_screen = VictoryScreen(screen.get_width(), screen.get_height())
        self.winner = None
        self.waiting_for_continue = False

    def update(self):
        if self.waiting_for_continue:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RETURN]:
                return True
            return False

        dt = self.clock.tick(60) / 1000

        self.update_countdown()

        car_data = [(car.color, car.number, car.laps) for car in self.cars]
        self.ui.update(car_data)

        if self.game_state.race_started:
            for car in self.cars:
                car.handle_input(self.game_state)
                car.update(self.track)

                if self.target_laps > 0 and car.laps >= self.target_laps:
                    self.winner = car
                    self.waiting_for_continue = True
                    return False

            self.check_collisions()

        return False

    def draw(self):
        self.screen.fill((0, 100, 0))  # Green background
        self.track.draw(self.screen)
        self.countdown_lights.draw(self.screen, self.game_state)

        for car in self.cars:
            car.draw(self.screen)

        self.ui.draw(self.screen)

        if self.waiting_for_continue:
            self.victory_screen.draw(self.screen, self.winner.color, self.winner.number)

    def get_player_color(self, index):
        colors = [
            (255, 0, 0),  # Red
            (0, 0, 255),  # Blue
            (0, 255, 0),  # Green
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 255, 128),  # Mint
            (255, 128, 128)  # Pink
        ]
        return colors[index % len(colors)]

    def check_collisions(self):
        # Check collisions between all pairs of cars
        for i in range(len(self.cars)):
            for j in range(i + 1, len(self.cars)):
                car1 = self.cars[i]
                car2 = self.cars[j]

                # Simple circle collision between cars
                dx = car1.pos[0] - car2.pos[0]
                dy = car1.pos[1] - car2.pos[1]
                distance = (dx ** 2 + dy ** 2) ** 0.5

                if distance < (car1.radius + car2.radius):
                    # Calculate collision normal
                    if distance == 0:  # Prevent division by zero
                        nx, ny = 1, 0
                    else:
                        nx = dx / distance
                        ny = dy / distance

                    # Relative velocity
                    rel_vel_x = car1.vel[0] - car2.vel[0]
                    rel_vel_y = car1.vel[1] - car2.vel[1]

                    # Velocity along the normal
                    vel_normal = rel_vel_x * nx + rel_vel_y * ny

                    # Do not resolve if cars are moving apart
                    if vel_normal > 0:
                        continue

                    # Restitution (bounciness)
                    restitution = 0.3  # Reduced bounce

                    # Apply impulse
                    impulse = -(1 + restitution) * vel_normal
                    car1.vel[0] += impulse * nx
                    car1.vel[1] += impulse * ny
                    car2.vel[0] -= impulse * nx
                    car2.vel[1] -= impulse * ny

                    # Separate cars to prevent sticking
                    overlap = (car1.radius + car2.radius) - distance
                    car1.pos[0] += nx * overlap * 0.5
                    car1.pos[1] += ny * overlap * 0.5
                    car2.pos[0] -= nx * overlap * 0.5
                    car2.pos[1] -= ny * overlap * 0.5

    def update_countdown(self):
        if not self.game_state.race_started:
            self.game_state.countdown_timer += 1
            if self.game_state.countdown_timer >= self.game_state.countdown_duration:
                self.game_state.countdown_state += 1
                self.game_state.countdown_timer = 0
                if self.game_state.countdown_state >= self.game_state.max_countdown_state:
                    self.game_state.race_started = True
