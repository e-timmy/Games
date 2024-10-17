import pygame
import math
import time
import threading
from character import Character
from pathfinder import Pathfinder
from constants import PLAYER_SPEED, PLAYER_SPRINT_SPEED, TILE_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT

class Player(Character):
    def __init__(self, x, y, environment):
        sprite_sheet = pygame.image.load('assets/town_rpg_pack/graphics/characters/hero.png').convert_alpha()
        super().__init__(x, y, sprite_sheet, 16, 16)
        self.prev_x = x
        self.prev_y = y
        self.direction = 'down'
        self.is_sprinting = False
        self.path = []
        self.pathfinder = Pathfinder(environment)
        self.target_x = None
        self.target_y = None
        self.pathfinding_timeout = 0.1  # 100 ms timeout for pathfinding
        self.pathfinding_thread = None
        self.cancel_flag = threading.Event()

    def move(self, dx, dy, environment):
        new_x = self.x + dx
        new_y = self.y + dy

        self.is_moving = dx != 0 or dy != 0

        if abs(dx) > abs(dy):
            if dx < 0:
                self.direction = 'left'
            else:
                self.direction = 'right'
        elif dy != 0:
            if dy < 0:
                self.direction = 'up'
            else:
                self.direction = 'down'

        if not environment.is_collision(new_x, new_y):
            self.prev_x = self.x
            self.prev_y = self.y
            self.x = new_x
            self.y = new_y

    def get_current_sprite(self):
        if self.direction == 'right':
            return pygame.transform.flip(self.sprites['left'][self.animation_frame], True, False)
        return self.sprites[self.direction][self.animation_frame]

    def update(self, dt, environment):
        super().update(dt)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.is_sprinting = True
        else:
            self.is_sprinting = False

        if self.path:
            target = self.path[0]
            dx = target[0] - self.x
            dy = target[1] - self.y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance > 1:
                speed = PLAYER_SPRINT_SPEED if self.is_sprinting else PLAYER_SPEED
                move_distance = speed * dt

                if distance <= move_distance:
                    self.x, self.y = target
                    self.path.pop(0)
                else:
                    ratio = move_distance / distance
                    new_x = self.x + dx * ratio
                    new_y = self.y + dy * ratio
                    if not environment.is_collision(new_x, new_y):
                        self.move(dx * ratio, dy * ratio, environment)
                    else:
                        self.path = []
                self.is_moving = True
            else:
                self.path.pop(0)
        else:
            self.is_moving = False

    def set_target(self, target_x, target_y, environment):
        # Check if the target is in a collision area
        if environment.is_collision(target_x, target_y):
            print("Target is in an obstacle area")
            return

        self.target_x = target_x
        self.target_y = target_y

        # Cancel any ongoing pathfinding
        if self.pathfinding_thread and self.pathfinding_thread.is_alive():
            self.cancel_flag.set()
            self.pathfinding_thread.join()

        # Reset the cancel flag and start a new pathfinding thread
        self.cancel_flag.clear()
        self.pathfinding_thread = threading.Thread(target=self.find_path, args=(environment,))
        self.pathfinding_thread.start()

    def find_path(self, environment):
        start_time = time.time()
        try:
            path = self.pathfinder.find_path((self.x, self.y), (self.target_x, self.target_y))
            if path and not self.cancel_flag.is_set():
                self.path = path
            elif not self.cancel_flag.is_set():
                print("No path found")
        except Exception as e:
            if not self.cancel_flag.is_set():
                print(f"Pathfinding error: {e}")
        finally:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.pathfinding_timeout and not self.cancel_flag.is_set():
                print(f"Pathfinding timed out after {elapsed_time:.2f} seconds")
                self.path = []