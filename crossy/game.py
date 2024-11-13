
import pygame
from player import Player
from vehicle import Vehicle
from game_state import GameState
from level_generator import generate_level
import random


class Game:
    def __init__(self, screen):
        self.screen = screen
        self.current_level = 1
        self.screen_height = 600
        self.road_height = 300
        self.level_start_time = 0
        self.level_message = ""
        self.game_over = False
        # Base values for vehicle generation
        self.vehicles_per_lane = 3  # Target number of vehicles per lane
        self.spawn_timer = 0
        self.reset(show_message=True)

    def reset(self, show_message=False):
        self.game_over = False
        prev_num_lanes = self.level_config.num_lanes if hasattr(self, 'level_config') else 0
        prev_speed = self.level_config.max_speed if hasattr(self, 'level_config') else 0

        self.level_config = generate_level(self.current_level)

        # Calculate road positions
        self.road_top = (self.screen_height - self.road_height) // 2
        self.road_bottom = self.road_top + self.road_height

        # Calculate lane positions
        lane_spacing = self.road_height / self.level_config.num_lanes
        self.lanes = [
            self.road_top + (i + 0.5) * lane_spacing
            for i in range(self.level_config.num_lanes)
        ]

        # Initialize lane directions and speeds
        self.lane_directions = [random.choice([-1, 1]) for _ in range(self.level_config.num_lanes)]

        # Initialize game objects
        self.player = Player(400, self.road_bottom + 50)
        self.vehicles = []
        self.spawn_timer = 0
        self.victory_timer = 0
        self.showing_level_complete = False

        # Generate level message
        if show_message:
            self.level_message = self.generate_level_message(prev_num_lanes, prev_speed)
            self.level_start_time = pygame.time.get_ticks()

        # Spawn initial vehicles
        self.spawn_initial_vehicles()

    def spawn_initial_vehicles(self):
        # Spawn about the same number of vehicles we want to maintain
        for lane in range(len(self.lanes)):
            # Spawn 2-3 vehicles per lane initially
            num_vehicles = random.randint(2, 3)
            spawned = 0
            attempts = 0

            while spawned < num_vehicles and attempts < 10:
                direction = self.lane_directions[lane]
                speed = random.uniform(self.level_config.min_speed,
                                       self.level_config.max_speed) * direction
                vehicle_type = random.choice(self.level_config.vehicle_types)

                # Space vehicles evenly across the screen
                segment_width = 800 // num_vehicles
                base_x = spawned * segment_width
                x = random.randint(base_x, base_x + segment_width - 100)

                # Adjust x based on direction for natural flow
                if direction < 0:
                    x = 800 - x

                # Check for minimum spacing
                if not any(abs(v.x - x) < 150 for v in self.vehicles if abs(v.y - self.lanes[lane]) < 5):
                    self.vehicles.append(Vehicle(x, self.lanes[lane], speed, vehicle_type))
                    spawned += 1
                attempts += 1

    def spawn_vehicle(self):
        # Only spawn if we're below target vehicle count
        vehicles_per_lane = {lane: 0 for lane in range(len(self.lanes))}
        for vehicle in self.vehicles:
            for lane_idx, lane_y in enumerate(self.lanes):
                if abs(vehicle.y - lane_y) < 5:
                    vehicles_per_lane[lane_idx] += 1

        # Find lanes that need more vehicles
        eligible_lanes = [lane for lane, count in vehicles_per_lane.items()
                          if count < self.vehicles_per_lane]

        if eligible_lanes:
            lane = random.choice(eligible_lanes)
            direction = self.lane_directions[lane]
            speed = random.uniform(self.level_config.min_speed,
                                   self.level_config.max_speed) * direction
            vehicle_type = random.choice(self.level_config.vehicle_types)

            x = -100 if direction > 0 else 900

            # Check if there's enough space to spawn
            vehicles_in_lane = [v for v in self.vehicles if abs(v.y - self.lanes[lane]) < 5]
            if not vehicles_in_lane or min(abs(v.x - x) for v in vehicles_in_lane) > 150:
                self.vehicles.append(Vehicle(x, self.lanes[lane], speed, vehicle_type))

    def update(self):
        if self.game_over:
            # Draw game state before overlay
            self.draw_game_state()
            # Draw semi-transparent overlay
            overlay = pygame.Surface((800, 600))
            overlay.fill((0, 0, 0))
            overlay.set_alpha(128)
            self.screen.blit(overlay, (0, 0))

            font = pygame.font.Font(None, 74)
            text = font.render("Game Over!", True, (255, 255, 255))
            text_rect = text.get_rect(center=(400, 250))
            self.screen.blit(text, text_rect)

            instruction = pygame.font.Font(None, 36)
            inst_text = instruction.render("Press SPACE to return to menu", True, (255, 255, 255))
            inst_rect = inst_text.get_rect(center=(400, 350))
            self.screen.blit(inst_text, inst_rect)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.current_level = 1  # Reset to level 1
                return GameState.MENU
            return GameState.PLAYING

        return self.update_game()

    def update_game(self):
        self.draw_game_state()

        # Update and check collisions
        if not self.game_over:
            self.player.update(self.lanes)

            # Spawn check
            self.spawn_timer += 1
            if self.spawn_timer >= self.level_config.spawn_rate:
                self.spawn_vehicle()
                self.spawn_timer = 0

            # Victory check
            if self.player.y < self.road_top:
                if not self.showing_level_complete:
                    self.victory_timer += 1
                    if self.victory_timer >= 30:
                        self.showing_level_complete = True
                        self.player.start_celebration()  # Start celebration animation

                if self.showing_level_complete:
                    self.draw_overlay(f"Level {self.current_level} Complete!")
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_SPACE]:
                        self.current_level += 1
                        self.reset(show_message=True)

        return GameState.PLAYING

    def draw_game_state(self):
        # Draw background and road
        self.screen.fill((100, 200, 100))
        pygame.draw.rect(self.screen, (80, 80, 80),
                         (0, self.road_top, 800, self.road_height))

        # Draw lane markers
        for y in range(self.road_top, self.road_bottom, self.road_height // self.level_config.num_lanes):
            pygame.draw.line(self.screen, (255, 255, 255),
                             (0, y), (800, y), 2)

        self.draw_level_message()
        self.player.draw(self.screen)

        # Update and draw vehicles
        for vehicle in self.vehicles[:]:
            vehicle.update()
            vehicle.draw(self.screen)

            if vehicle.x < -200 or vehicle.x > 1000:
                self.vehicles.remove(vehicle)
            elif vehicle.collides_with(self.player) and not self.game_over:
                self.game_over = True

    def draw_overlay(self, message):
        overlay = pygame.Surface((800, 600))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))

        font = pygame.font.Font(None, 74)
        text = font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(400, 250))
        self.screen.blit(text, text_rect)

        level_text = f"Level {self.current_level}"
        level_font = pygame.font.Font(None, 48)
        level_surface = level_font.render(level_text, True, (255, 255, 255))
        level_rect = level_surface.get_rect(center=(400, 150))
        self.screen.blit(level_surface, level_rect)

        font_small = pygame.font.Font(None, 36)
        instruction = font_small.render("Press SPACE to continue", True, (255, 255, 255))
        instruction_rect = instruction.get_rect(center=(400, 350))
        self.screen.blit(instruction, instruction_rect)

    def draw_level_message(self):
        if pygame.time.get_ticks() - self.level_start_time < 3000:  # Show for 3 seconds
            font = pygame.font.Font(None, 36)
            text = font.render(self.level_message, True, (255, 255, 255))
            text_rect = text.get_rect(center=(400, 50))

            # Draw semi-transparent background for text
            bg_rect = text_rect.copy()
            bg_rect.inflate_ip(20, 10)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
            bg_surface.fill((0, 0, 0))
            bg_surface.set_alpha(128)
            self.screen.blit(bg_surface, bg_rect)

            self.screen.blit(text, text_rect)

    def generate_level_message(self, prev_num_lanes, prev_speed):
        messages = []
        if self.current_level == 1:
            return "Level 1 - Cross the road!"

        if self.level_config.num_lanes > prev_num_lanes:
            messages.append(f"Added lane! Now {self.level_config.num_lanes} lanes")
        if self.level_config.max_speed > prev_speed:
            messages.append("Traffic speed increased!")
        if 'truck' in self.level_config.vehicle_types and prev_num_lanes > 0 and 'truck' not in generate_level(
                self.current_level - 1).vehicle_types:
            messages.append("Trucks added!")
        if 'sports_car' in self.level_config.vehicle_types and prev_num_lanes > 0 and 'sports_car' not in generate_level(
                self.current_level - 1).vehicle_types:
            messages.append("Sports cars added!")

        if not messages:
            messages.append(f"Level {self.current_level} - Traffic intensifies!")

        return " | ".join(messages)
