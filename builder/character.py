import pygame
from constants import *
from pathfinding import a_star


class Character:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.is_crouching = False
        self.height = CHAR_HEIGHT
        self.rect = pygame.Rect(x, y, CHAR_WIDTH, self.height)

    def crouch(self):
        if not self.is_crouching:
            self.is_crouching = True
            self.height = CHAR_HEIGHT // 2
            # Adjust position to account for height change
            self.y += CHAR_HEIGHT // 2
            self.rect = pygame.Rect(self.x, self.y, CHAR_WIDTH, self.height)

    def uncrouch(self, tiles):
        if self.is_crouching:
            # Check if there's space to stand
            test_rect = pygame.Rect(self.x, self.y - CHAR_HEIGHT // 2,
                                    CHAR_WIDTH, CHAR_HEIGHT)
            can_stand = True

            # Check for collision with tiles
            for y in range(len(tiles)):
                for x in range(len(tiles[0])):
                    if tiles[y][x] != EMPTY_TILE:
                        tile_rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE,
                                                TILE_SIZE, TILE_SIZE)
                        if test_rect.colliderect(tile_rect):
                            can_stand = False
                            break
                if not can_stand:
                    break

            if can_stand:
                self.is_crouching = False
                self.height = CHAR_HEIGHT
                self.y -= CHAR_HEIGHT // 2
                self.rect = pygame.Rect(self.x, self.y, CHAR_WIDTH, self.height)

    def move(self, dx, dy, tiles):
        # Try horizontal movement first
        self.x += dx
        self.rect.x = self.x
        if self.check_collision_in_direction(tiles, "horizontal"):
            self.x -= dx
            self.rect.x = self.x
            self.vel_x = 0

        # Then try vertical movement
        self.y += dy
        self.rect.y = self.y
        if self.check_collision_in_direction(tiles, "vertical"):
            self.y -= dy
            self.rect.y = self.y
            if dy > 0:
                self.on_ground = True
            self.vel_y = 0

    def check_collision_in_direction(self, tiles, direction):
        # Ground collision
        if self.y + self.height > GROUND_LEVEL:
            return True

        # Tile collision
        for y in range(len(tiles)):
            for x in range(len(tiles[0])):
                if tiles[y][x] != EMPTY_TILE:
                    tile_rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE,
                                            TILE_SIZE, TILE_SIZE)
                    if self.rect.colliderect(tile_rect):
                        return True
        return False

    def check_collision(self, tiles):
        self.on_ground = False

        # Ground collision
        if self.y + self.height > GROUND_LEVEL:
            self.y = GROUND_LEVEL - self.height
            self.vel_y = 0
            self.on_ground = True
            self.rect.y = self.y

        # Tile collision
        for y in range(len(tiles)):
            for x in range(len(tiles[0])):
                if tiles[y][x] != EMPTY_TILE:
                    tile_rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE,
                                            TILE_SIZE, TILE_SIZE)
                    if self.rect.colliderect(tile_rect):
                        # Calculate overlap
                        overlap_left = self.rect.right - tile_rect.left
                        overlap_right = tile_rect.right - self.rect.left
                        overlap_top = self.rect.bottom - tile_rect.top
                        overlap_bottom = tile_rect.bottom - self.rect.top

                        # Find smallest overlap
                        min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

                        if min_overlap == overlap_top and self.vel_y >= 0:
                            self.rect.bottom = tile_rect.top
                            self.y = self.rect.y
                            self.vel_y = 0
                            self.on_ground = True
                        elif min_overlap == overlap_bottom and self.vel_y < 0:
                            self.rect.top = tile_rect.bottom
                            self.y = self.rect.y
                            self.vel_y = 0
                        elif min_overlap == overlap_left and not self.on_ground:
                            self.rect.right = tile_rect.left
                            self.x = self.rect.x
                            self.vel_x = 0
                        elif min_overlap == overlap_right and not self.on_ground:
                            self.rect.left = tile_rect.right
                            self.x = self.rect.x
                            self.vel_x = 0

    def jump(self):
        if self.on_ground:
            self.vel_y = JUMP_SPEED

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)


class Player(Character):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.color = PLAYER_COLOR
        self.can_jump = True

    def update(self, keys, tiles):
        # Horizontal movement
        self.vel_x = 0
        if keys[pygame.K_LEFT]:
            self.vel_x = -MOVE_SPEED
        if keys[pygame.K_RIGHT]:
            self.vel_x = MOVE_SPEED

        # Jumping
        if keys[pygame.K_UP] and self.on_ground and self.can_jump:
            self.jump()
            self.can_jump = False
        elif not keys[pygame.K_UP]:
            self.can_jump = True

        # Crouching
        if keys[pygame.K_DOWN]:
            self.crouch()
        else:
            self.uncrouch(tiles)

        # Apply gravity
        self.vel_y += GRAVITY

        # Apply movement
        self.move(self.vel_x, self.vel_y, tiles)


class AI(Character):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.color = AI_COLOR
        self.path = []
        self.target_x = SCREEN_WIDTH - 100
        self.path_index = 0
        self.stuck_timer = 0
        self.last_position = (x, y)
        print(f"AI initialized at position ({x}, {y})")  # Debug print

    def pixel_to_grid(self, x, y):
        grid_x = int(x // TILE_SIZE)
        grid_y = int(y // TILE_SIZE)
        print(f"Converting pixel ({x}, {y}) to grid ({grid_x}, {grid_y})")  # Debug print
        return (grid_x, grid_y)

    def grid_to_pixel(self, grid_x, grid_y):
        pixel_x = grid_x * TILE_SIZE + TILE_SIZE // 2
        pixel_y = grid_y * TILE_SIZE + TILE_SIZE // 2
        print(f"Converting grid ({grid_x}, {grid_y}) to pixel ({pixel_x}, {pixel_y})")  # Debug print
        return (pixel_x, pixel_y)

    def can_move_through(self, tiles, grid_x, grid_y):
        if grid_y >= len(tiles) or grid_x >= len(tiles[0]) or grid_y < 0 or grid_x < 0:
            print(f"Position ({grid_x}, {grid_y}) is out of bounds")  # Debug print
            return False

        height_check = CHAR_HEIGHT // 2 if self.is_crouching else CHAR_HEIGHT
        can_move = True
        for check_y in range(grid_y, grid_y + (height_check // TILE_SIZE) + 1):
            if check_y >= len(tiles) or check_y < 0:
                can_move = False
                break
            if tiles[check_y][grid_x] != EMPTY_TILE:
                can_move = False
                break

        print(f"Can move through ({grid_x}, {grid_y}): {can_move}")  # Debug print
        return can_move

    def plan_path(self, tiles):
        print("\nPlanning new path...")  # Debug print
        start_pos = self.pixel_to_grid(self.x, self.y)
        goal_pos = self.pixel_to_grid(self.target_x, self.y)

        print(f"Planning path from {start_pos} to {goal_pos}")  # Debug print

        # Find path using A*
        grid_path = a_star(start_pos, goal_pos, tiles,
                           lambda pos: self.can_move_through(tiles, pos[0], pos[1]))

        if grid_path:
            print(f"Path found: {grid_path}")  # Debug print
            # Convert grid path to pixel coordinates
            self.path = [self.grid_to_pixel(x, y) for x, y in grid_path]
            self.path_index = 0
            self.stuck_timer = 0
        else:
            print("No path found, trying alternative path")  # Debug print
            self.try_alternative_path(tiles)

    def try_alternative_path(self, tiles):
        current_grid_y = int(self.y // TILE_SIZE)
        print(f"Trying alternative paths around y-level {current_grid_y}")  # Debug print

        for test_y in range(current_grid_y - 2, current_grid_y + 3):
            if test_y < 0 or test_y >= len(tiles):
                continue

            start_pos = self.pixel_to_grid(self.x, test_y * TILE_SIZE)
            goal_pos = self.pixel_to_grid(self.target_x, test_y * TILE_SIZE)

            print(f"Attempting path at y-level {test_y}: {start_pos} to {goal_pos}")  # Debug print

            path = a_star(start_pos, goal_pos, tiles,
                          lambda pos: self.can_move_through(tiles, pos[0], pos[1]))

            if path:
                print(f"Alternative path found at y-level {test_y}")  # Debug print
                self.path = [self.grid_to_pixel(x, y) for x, y in path]
                self.path_index = 0
                return

        print("No alternative paths found")  # Debug print
        self.path = []
        self.path_index = 0

    def update(self, tiles):
        # Print current state
        print(f"\nAI Update - Position: ({self.x}, {self.y})")
        print(f"Path exists: {bool(self.path)}, Path index: {self.path_index}")
        if self.path and self.path_index < len(self.path):
            print(f"Current target: {self.path[self.path_index]}")

        # Check if stuck
        current_pos = (self.x, self.y)
        if abs(current_pos[0] - self.last_position[0]) < 1 and \
                abs(current_pos[1] - self.last_position[1]) < 1:
            self.stuck_timer += 1
            if self.stuck_timer % 10 == 0:  # Print every 10 frames when stuck
                print(f"Potentially stuck. Timer: {self.stuck_timer}")
        else:
            self.stuck_timer = 0
        self.last_position = current_pos

        # Replan if stuck or no path
        if self.stuck_timer > 60 or not self.path or self.path_index >= len(self.path):
            print("Replanning path due to:",
                  "Stuck" if self.stuck_timer > 60 else
                  "No path" if not self.path else
                  "End of path")
            self.plan_path(tiles)
            return

        # Follow current path
        target_x, target_y = self.path[self.path_index]

        # Horizontal movement
        if abs(self.x - target_x) > MOVE_SPEED:
            self.vel_x = MOVE_SPEED if target_x > self.x else -MOVE_SPEED
            print(f"Moving {'right' if self.vel_x > 0 else 'left'} with velocity {self.vel_x}")
        else:
            self.vel_x = 0

        # Vertical movement (jumping and crouching)
        if target_y < self.y - JUMP_SPEED and self.on_ground:
            print("Attempting to jump")
            self.jump()
        elif target_y > self.y + TILE_SIZE and self.on_ground:
            print("Attempting to crouch")
            self.crouch()
        elif self.is_crouching and target_y <= self.y:
            print("Attempting to uncrouch")
            self.uncrouch(tiles)

        # Check if reached current target
        if abs(self.x - target_x) < MOVE_SPEED and \
                abs(self.y - target_y) < JUMP_SPEED:
            print(f"Reached target {self.path_index}, moving to next")
            self.path_index += 1

        # Apply gravity
        self.vel_y += GRAVITY

        # Apply movement
        self.move(self.vel_x, self.vel_y, tiles)
        self.check_collision(tiles)
