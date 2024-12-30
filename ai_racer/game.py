import threading
import time
import pygame
from player import Player
from ai import AI
from reinforcement_learner import ReinforcementLearner
from ui import UI
from track import Track
from game_state import GameState
from countdown_lights import CountdownLights
from victory_screen import VictoryScreen
from config import SCREEN_WIDTH, SCREEN_HEIGHT
from waypoints import WaypointVisualizer


class Game:
    def __init__(self, screen, player_configs, target_laps=0, headless=False, speed_factor=1):
        self.headless = headless
        self.screen = screen  # Can be None in headless mode
        self.track = Track(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.game_state = GameState(SCREEN_WIDTH, SCREEN_HEIGHT, self.track)
        self.countdown_lights = None if headless else CountdownLights(self.track)
        self.target_laps = target_laps
        self.waypoint_visualizer = WaypointVisualizer(self.track)

        # Initialize cars
        self.cars = []
        for i, config in enumerate(player_configs):
            if i >= len(self.track.start_positions):
                break
            pos = self.track.start_positions[i]
            if config.player_type == "Human":
                keys = ["UP", "DN", "LT", "RT"] if config.control_scheme == "arrows" else ["W", "S", "A", "D"]
                car = Player(pos, self.get_player_color(i), str(i + 1), keys)
            elif config.player_type == "RL":
                car = ReinforcementLearner(pos, self.get_player_color(i), str(i + 1), self.track)
            else:
                car = AI(pos, self.get_player_color(i), str(i + 1), self.track, difficulty=config.difficulty)
            self.cars.append(car)

        self.ui = None if headless else UI(screen)
        self.clock = pygame.time.Clock() if not headless else None
        self.victory_screen = None if headless else VictoryScreen(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.winner = None
        self.waiting_for_continue = False

        self.simulation_thread = None
        self.simulation_running = False
        self.speed_factor = speed_factor
        self.simulation_step = 0
        self.last_progress_update = 0

        # Start race immediately in headless mode
        if headless:
            self.game_state.race_started = True
            print("Race started!")

    def _update_headless(self):
        self.simulation_step += 1

        # Provide progress update every 1000 steps
        if self.simulation_step - self.last_progress_update >= 1000:
            self.last_progress_update = self.simulation_step
            self._print_progress()

        # Update multiple times based on speed factor
        for _ in range(self.speed_factor):
            for car in self.cars:
                car.handle_input(self.game_state)
                car.update(self.track)

                if self.target_laps > 0 and car.laps >= self.target_laps:
                    self.winner = car
                    return True  # Race is finished

            self.check_collisions()

        return False

    def _print_progress(self):
        print("\nRace Progress:")
        for car in self.cars:
            print(f"Car {car.number}: Lap {car.laps}")
        if self.target_laps > 0:
            leader = max(self.cars, key=lambda x: x.laps)
            progress = (leader.laps / self.target_laps) * 100
            print(f"Race completion: {progress:.1f}%")

    def update(self):

        if self.headless:
            return self._update_headless()
        else:
            return self._update_visual()

    def _update_visual(self):
        # Existing update logic for visual mode
        for car in self.cars:
            if car.is_player:  # Only update for the human player
                self.waypoint_visualizer.update(car.pos)

        if self.waiting_for_continue:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RETURN]:
                return True
            return False

        self.clock.tick(60)

        car_data = [(car.color, car.number, car.laps) for car in self.cars]
        self.ui.update(car_data)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.simulation_running = False
                return True

            if self.ui.handle_event(event):
                self.return_to_menu = True
                return True

            new_speed = self.ui.get_speed_factor()
            if new_speed != self.speed_factor:
                self.speed_factor = new_speed
                if self.speed_factor > 1:
                    if not self.simulation_thread or not self.simulation_thread.is_alive():
                        self.simulation_running = True
                        self.simulation_thread = threading.Thread(target=self.run_simulation)
                        self.simulation_thread.start()
                else:
                    self.simulation_running = False
                    if self.simulation_thread:
                        self.simulation_thread.join()

        if self.speed_factor == 1:
            for car in self.cars:
                car.handle_input(self.game_state)
                car.update(self.track)
            self.check_collisions()

        self.update_countdown()
        return False

    def draw(self):
        if self.headless:
            return

        # Existing draw logic
        self.screen.fill((0, 100, 0))
        self.track.draw(self.screen)
        self.countdown_lights.draw(self.screen, self.game_state)
        self.waypoint_visualizer.draw(self.screen)

        for car in self.cars:
            car.draw(self.screen)

        self.ui.draw(self.screen)

        if self.waiting_for_continue:
            self.victory_screen.draw(self.screen, self.winner.color, self.winner.number)

    def run_simulation(self):
        while self.simulation_running:
            for _ in range(self.speed_factor):
                for car in self.cars:
                    car.handle_input(self.game_state)
                    car.update(self.track)
                    if self.target_laps > 0 and car.laps >= self.target_laps:
                        self.winner = car
                        self.waiting_for_continue = True
                        self.simulation_running = False
                        return
                self.check_collisions()
            time.sleep(1 / 60)

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

    def __del__(self):
        # # Save reinforcement learner models when game ends
        # for car in self.cars:
        #     if isinstance(car, ReinforcementLearner):
        #         car.save_model()
        ...
