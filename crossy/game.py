import pygame
import random
import math
from player import Player
from game_state import GameState
from level_generator import generate_level, DifficultyConfig
from vehicle import Vehicle
from river import River, Log, Lilypad


class Game:
    def __init__(self, screen):
        self.screen = screen
        self.current_level = 1
        self.screen_width = 800
        self.screen_height = 600

        # Base size for a single lane
        self.base_lane_height = 60
        # Maximum total road height (when many lanes)
        self.max_road_height = 300

        self.river_height = 0
        self.level_start_time = 0
        self.level_message = ""
        self.game_over = False
        self.spawn_timer = 0
        self.reset(show_message=True)

    def reset(self, show_message=False):
        self.game_over = False
        prev_num_lanes = self.level_config.num_lanes if hasattr(self, 'level_config') else 0
        prev_speed = self.level_config.max_speed if hasattr(self, 'level_config') else 0

        self.level_config = generate_level(self.current_level)

        # Calculate road height based on number of lanes with scaling
        if self.level_config.num_lanes == 1:
            self.road_height = self.base_lane_height
        else:
            # Scale down lane height as number of lanes increases
            scale_factor = max(0.6, 1 - (self.level_config.num_lanes * 0.05))  # 5% reduction per lane, minimum 60% size
            adjusted_lane_height = self.base_lane_height * scale_factor
            self.road_height = min(adjusted_lane_height * self.level_config.num_lanes, self.max_road_height)

        # Calculate river size
        if self.current_level <= 5:
            self.river_height = 0
            self.river_lanes = 0
            total_height = self.road_height
        else:
            self.river_lanes = min(self.current_level - 5, 3)
            # Scale river lanes similar to road lanes
            river_scale_factor = max(0.7, 1 - (self.river_lanes * 0.05))
            self.river_height = 60 * self.river_lanes * river_scale_factor
            total_height = self.road_height + self.river_height

        # Center everything vertically
        vertical_center = self.screen_height // 2
        self.road_top = vertical_center - (total_height // 2) + (self.river_height if self.current_level > 5 else 0)
        self.road_bottom = self.road_top + self.road_height
        self.river_top = self.road_top - self.river_height
        self.river_bottom = self.road_top

        # Calculate lane positions with equal spacing
        self.lane_height = self.road_height / self.level_config.num_lanes
        self.lanes = [
            self.road_top + (i * self.lane_height) + (self.lane_height // 2)
            for i in range(self.level_config.num_lanes)
        ]

        # Initialize lane directions and speeds
        self.lane_directions = [random.choice([-1, 1]) for _ in range(self.level_config.num_lanes)]

        # Calculate target vehicles per lane based on level
        base_vehicles = 2
        level_multiplier = 1 + (self.current_level - 1) * 0.2
        self.target_vehicles_per_lane = min(base_vehicles * level_multiplier, 4)
        self.target_total_vehicles = int(self.target_vehicles_per_lane * self.level_config.num_lanes)

        # Initialize game objects
        self.player = Player(self.screen_width // 2, self.road_bottom + 50)
        self.vehicles = []
        self.spawn_timer = 0
        self.victory_timer = 0
        self.showing_level_complete = False

        # Initialize river (after level 5)
        if self.current_level > 5:
            self.river = River(self.river_top, self.river_height, self.river_lanes)
        else:
            self.river = None

        # Generate level message
        if show_message:
            self.level_message = self.generate_level_message(prev_num_lanes, prev_speed)
            self.level_start_time = pygame.time.get_ticks()

        # Spawn initial vehicles
        self.spawn_initial_vehicles()

    def get_lane_vehicle_count(self, lane_idx):
        return sum(1 for v in self.vehicles if abs(v.y - self.lanes[lane_idx]) < 5)

    def get_lane_spacing(self, lane_idx):
        # Get all vehicles in the lane
        lane_vehicles = [v for v in self.vehicles if abs(v.y - self.lanes[lane_idx]) < 5]
        if not lane_vehicles:
            return float('inf')

        # Sort vehicles by x position
        lane_vehicles.sort(key=lambda v: v.x)

        # Calculate minimum spacing between consecutive vehicles
        min_spacing = float('inf')
        for i in range(len(lane_vehicles) - 1):
            spacing = abs(lane_vehicles[i + 1].x - lane_vehicles[i].x)
            min_spacing = min(min_spacing, spacing)

        return min_spacing

    def spawn_initial_vehicles(self):
        for lane in range(len(self.lanes)):
            # Calculate number of vehicles based on level
            num_vehicles = int(self.target_vehicles_per_lane)
            spawned = 0
            attempts = 0
            min_spacing = 200  # Minimum spacing between vehicles

            while spawned < num_vehicles and attempts < 10:
                direction = self.lane_directions[lane]
                speed = random.uniform(self.level_config.min_speed,
                                       self.level_config.max_speed) * direction
                vehicle_type = random.choice(self.level_config.vehicle_types)

                # Space vehicles evenly across the screen
                segment_width = 800 // num_vehicles
                x = random.randint(spawned * segment_width, (spawned + 1) * segment_width - 100)

                # Check for minimum spacing
                if not any(abs(v.x - x) < min_spacing for v in self.vehicles if abs(v.y - self.lanes[lane]) < 5):
                    self.vehicles.append(Vehicle(x, self.lanes[lane], speed, vehicle_type))
                    spawned += 1
                attempts += 1

        # Ensure at least one vehicle per lane
        for lane in range(len(self.lanes)):
            if not any(v.y == self.lanes[lane] for v in self.vehicles):
                direction = self.lane_directions[lane]
                speed = random.uniform(self.level_config.min_speed,
                                       self.level_config.max_speed) * direction
                vehicle_type = random.choice(self.level_config.vehicle_types)
                x = random.randint(0, 700)  # Spawn within screen bounds
                self.vehicles.append(Vehicle(x, self.lanes[lane], speed, vehicle_type))

    def spawn_vehicle(self):
        # Calculate the current number of vehicles
        current_vehicles = len(self.vehicles)

        # Calculate the probability of spawning based on the difference from the target
        spawn_probability = max(0, min(1, (self.target_total_vehicles - current_vehicles) / self.target_total_vehicles))

        # Attempt to spawn based on probability
        if random.random() < spawn_probability:
            # Choose a random lane
            lane = random.randint(0, len(self.lanes) - 1)
            direction = self.lane_directions[lane]
            speed = random.uniform(self.level_config.min_speed,
                                   self.level_config.max_speed) * direction
            vehicle_type = random.choice(self.level_config.vehicle_types)

            # Spawn position based on direction
            x = -100 if direction > 0 else 900

            # Check minimum spacing from edge
            min_edge_spacing = 200  # Minimum spacing from screen edge
            vehicles_in_lane = [v for v in self.vehicles if abs(v.y - self.lanes[lane]) < 5]

            if direction > 0:
                near_edge = [v for v in vehicles_in_lane if v.x < min_edge_spacing]
            else:
                near_edge = [v for v in vehicles_in_lane if v.x > 800 - min_edge_spacing]

            if not near_edge:  # Only spawn if there's no vehicle near the edge
                self.vehicles.append(Vehicle(x, self.lanes[lane], speed, vehicle_type))

    def update(self):
        # Always update game state regardless of game over
        self.draw_game_state()

        # Update all game objects
        self.spawn_vehicle()

        # Update vehicles
        for vehicle in self.vehicles[:]:
            vehicle.update()
            if vehicle.x < -200 or vehicle.x > 1000:
                self.vehicles.remove(vehicle)

        # Update river if it exists
        if self.river:
            self.river.update()

        # If game is over, draw overlay and handle retry
        if self.game_over:
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
                self.current_level = 1
                return GameState.MENU
            return GameState.PLAYING

        return self.update_game()

    def update_game(self):
        prev_y = self.player.y

        # Update player if not game over
        if self.player.is_dead:
            if self.player.update(self.lanes, self.river_top if self.river else None,
                                  self.river_bottom if self.river else None):
                self.game_over = True
        else:
            self.player.update(self.lanes, self.river_top if self.river else None,
                               self.river_bottom if self.river else None)

        if self.river:
            # Reset all lilypads that don't have a player
            for lilypad in self.river.lilypads:
                if not lilypad.has_player and lilypad.sinking:
                    lilypad.reset_state()

            # Check if player is in river area and not dead
            if not self.player.is_dead and self.river_top <= self.player.y <= self.river_bottom:
                current_log = self.river.get_log_at(self.player.x, self.player.y)
                current_lilypad = self.river.get_lilypad_at(self.player.x, self.player.y)

                if current_log:
                    self.player.on_log = True
                    self.player.on_lilypad = False
                    self.player.move_on_log(current_log.speed)
                elif current_lilypad and not current_lilypad.is_submerged():
                    self.player.on_log = False
                    self.player.on_lilypad = True
                    self.player.move_on_lilypad(current_lilypad.speed)
                    current_lilypad.has_player = True
                    if not current_lilypad.sinking:
                        current_lilypad.start_sinking()
                elif not self.player.test_mode and not self.player.is_moving:
                    if not self.player.is_dead:
                        self.player.die('river')
                else:
                    self.player.on_log = False
                    self.player.on_lilypad = False

                    # Reset has_player for all lilypads
                    for lilypad in self.river.lilypads:
                        lilypad.has_player = False

            # Adjust player's y-position to the center of the river lane
            if not self.player.is_dead and self.river_top <= self.player.y <= self.river_bottom and not self.player.is_moving:
                river_lane = int((self.player.y - self.river_top) / self.river.lane_height)
                self.player.y = self.river.get_lane_center(river_lane)

        # Check for vehicle collisions
        for vehicle in self.vehicles:
            if not self.player.is_dead and vehicle.collides_with(self.player) and not self.player.test_mode:
                self.player.die('road')

        # Victory condition check (only if not dead)
        if not self.player.is_dead and self.player.y < (self.river_top if self.river else self.road_top):
            if not self.showing_level_complete:
                self.victory_timer += 1
                if self.victory_timer >= 30:
                    self.showing_level_complete = True
                    self.player.start_celebration()

            if self.showing_level_complete:
                self.draw_overlay(f"Level {self.current_level} Complete!")
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    self.current_level += 1
                    self.reset(show_message=True)

        return GameState.PLAYING

    def draw_game_state(self):
        # Draw background
        self.screen.fill((100, 200, 100))  # Grass color

        # Draw river (if exists)
        if self.river:
            self.river.draw(self.screen)

        # Draw road
        pygame.draw.rect(self.screen, (80, 80, 80),  # Road color
                         (0, self.road_top, 800, self.road_height))

        # Draw lane markers (including top and bottom road boundaries)
        # Top boundary
        pygame.draw.line(self.screen, (255, 255, 255),
                         (0, self.road_top), (800, self.road_top), 2)
        # Bottom boundary
        pygame.draw.line(self.screen, (255, 255, 255),
                         (0, self.road_bottom), (800, self.road_bottom), 2)
        # Lane markers
        for i in range(1, self.level_config.num_lanes):
            y = self.road_top + (i * self.lane_height)
            pygame.draw.line(self.screen, (255, 255, 255),
                             (0, y), (800, y), 2)

        self.draw_level_message()
        self.player.draw(self.screen)

        # Update and draw vehicles
        for vehicle in self.vehicles[:]:
            vehicle.update()
            vehicle.draw(self.screen)

            # Remove vehicles that are off screen
            if vehicle.x < -200 or vehicle.x > 1000:
                self.vehicles.remove(vehicle)
            # Only trigger game over if not in test mode
            elif vehicle.collides_with(self.player) and not self.game_over and not self.player.test_mode:
                if not self.player.is_dead:  # Only trigger death if not already dead
                    self.player.die('road')
                else:
                    self.game_over = True

        # Draw test mode indicator
        if self.player.test_mode:
            font = pygame.font.Font(None, 24)
            text = font.render("Test Mode", True, (255, 255, 255))
            text_rect = text.get_rect(topright=(790, 10))

            # Draw background for better visibility
            bg_rect = text_rect.copy()
            bg_rect.inflate_ip(10, 6)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
            bg_surface.fill((0, 0, 0))
            bg_surface.set_alpha(128)
            self.screen.blit(bg_surface, bg_rect)
            self.screen.blit(text, text_rect)

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
        if pygame.time.get_ticks() - self.level_start_time < 3000:
            font = pygame.font.Font(None, 36)
            text = font.render(self.level_message, True, (255, 255, 255))
            text_rect = text.get_rect(center=(400, 50))

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
        if self.current_level == 6:
            return "Level 6 - Watch out for the river and lilypads!"

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