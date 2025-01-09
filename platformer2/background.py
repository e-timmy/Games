import pygame
import random
from constants import *


class Background:
    def __init__(self):
        self.buildings = []
        self.last_building_x = 0
        self.generate_initial_buildings()

    def generate_initial_buildings(self):
        # Start generating from behind the player's starting position
        start_x = -SCREEN_WIDTH // 2  # This ensures buildings start from off-screen left
        end_x = START_X + SCREEN_WIDTH  # Generate up to and beyond player's position

        current_x = start_x
        while current_x < end_x:
            width = random.randint(MIN_BUILDING_WIDTH, MAX_BUILDING_WIDTH)
            height = random.randint(MIN_BUILDING_HEIGHT, MAX_BUILDING_HEIGHT)
            building = pygame.Rect(current_x,
                                   SCREEN_HEIGHT - height,
                                   width,
                                   height)

            # Generate window states
            window_states = {}
            for y in range(building.top + WINDOW_SPACING, building.bottom - WINDOW_SIZE, WINDOW_SPACING):
                for x in range(building.left + WINDOW_SPACING, building.right - WINDOW_SIZE, WINDOW_SPACING):
                    window_states[(x, y)] = random.random() > WINDOW_LIGHT_PROBABILITY

            self.buildings.append((building, window_states))
            current_x += width + BUILDING_GAP

        self.last_building_x = current_x

    def add_buildings_until(self, target_x):
        current_x = max(0, self.last_building_x)
        while current_x < target_x:
            width = random.randint(MIN_BUILDING_WIDTH, MAX_BUILDING_WIDTH)
            height = random.randint(MIN_BUILDING_HEIGHT, MAX_BUILDING_HEIGHT)
            building = pygame.Rect(current_x,
                                   SCREEN_HEIGHT - height,
                                   width,
                                   height)

            # Pre-generate window states
            window_states = {}
            for y in range(building.top + WINDOW_SPACING, building.bottom - WINDOW_SIZE, WINDOW_SPACING):
                for x in range(building.left + WINDOW_SPACING, building.right - WINDOW_SIZE, WINDOW_SPACING):
                    window_states[(x, y)] = random.random() > WINDOW_LIGHT_PROBABILITY

            self.buildings.append((building, window_states))
            current_x += width + BUILDING_GAP
            self.last_building_x = current_x

    def draw(self, screen, camera_offset):
        # Generate more buildings if needed
        camera_right_edge = -camera_offset[0] + SCREEN_WIDTH * 2
        if camera_right_edge > self.last_building_x - SCREEN_WIDTH:
            self.add_buildings_until(self.last_building_x + SCREEN_WIDTH)

        # Remove buildings that are completely off-screen to the left
        self.buildings = [(b, ws) for (b, ws) in self.buildings if b.right + camera_offset[0] > -BUILDING_GAP]

        for building, window_states in self.buildings:
            adjusted_rect = pygame.Rect(
                building.x + camera_offset[0],
                building.y,
                building.width,
                building.height
            )
            pygame.draw.rect(screen, BUILDING_COLOR, adjusted_rect)

            # Draw windows based on pre-generated states
            for (x, y), is_lit in window_states.items():
                window_rect = pygame.Rect(
                    x + camera_offset[0],
                    y,
                    WINDOW_SIZE,
                    WINDOW_SIZE
                )
                window_color = WINDOW_COLOR_LIT if is_lit else WINDOW_COLOR_UNLIT
                pygame.draw.rect(screen, window_color, window_rect)

        # Draw a "glow" effect at the bottom
        glow_height = 50
        glow_surface = pygame.Surface((SCREEN_WIDTH, glow_height), pygame.SRCALPHA)
        for i in range(glow_height):
            alpha = int(255 * (1 - i / glow_height))
            pygame.draw.line(glow_surface, (*GROUND_COLOR[:3], alpha), (0, i), (SCREEN_WIDTH, i))
        screen.blit(glow_surface, (0, SCREEN_HEIGHT - glow_height))