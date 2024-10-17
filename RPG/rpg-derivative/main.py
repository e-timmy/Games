import sys

import pygame
import random
from abc import ABC, abstractmethod
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TILE_SIZE = 16
PLAYER_SPEED = 2

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
# Textures
GROUND_COLOR = (34, 139, 34)  # Forest Green
OBSTACLE_COLOR = (128, 128, 128)  # Gray


from abc import ABC, abstractmethod
import pygame

class Person(ABC):
    def __init__(self, x, y, sprite_sheet, sprite_width, sprite_height):
        self.x = x
        self.y = y
        self.sprite_sheet = sprite_sheet
        self.sprite_width = sprite_width
        self.sprite_height = sprite_height
        self.direction = 'down'
        self.animation_frame = 0
        self.animation_speed = 0.15
        self.animation_time = 0
        self.is_moving = False
        self.sprites = self.load_sprites()

    def load_sprites(self):
        sprites = {}
        for i, direction in enumerate(['left', 'up', 'down']):
            sprites[direction] = [
                self.sprite_sheet.subsurface((j * self.sprite_width, i * self.sprite_height, self.sprite_width, self.sprite_height))
                for j in range(3)
            ]
        return sprites

    @abstractmethod
    def move(self, dx, dy, environment):
        pass

    def update(self, dt):
        if self.is_moving:
            self.animation_time += dt
            if self.animation_time >= self.animation_speed:
                self.animation_frame = (self.animation_frame + 1) % 3
                self.animation_time = 0
        else:
            self.animation_frame = 1  # Set to middle frame when static

    def get_current_sprite(self):
        return self.sprites[self.direction][self.animation_frame]


class Player(Person):
    def __init__(self, x, y):
        sprite_sheet = pygame.image.load('assets/town_rpg_pack/graphics/characters/hero.png').convert_alpha()
        super().__init__(x, y, sprite_sheet, 16, 16)
        self.prev_x = x
        self.prev_y = y
        self.last_direction = 'down'

    def move(self, dx, dy, environment):
        new_x = self.x + dx
        new_y = self.y + dy

        self.is_moving = dx != 0 or dy != 0

        if dx < 0:
            self.direction = 'left'
            self.last_direction = 'left'
        elif dx > 0:
            self.direction = 'left'  # We'll flip this horizontally when rendering
            self.last_direction = 'right'
        elif dy < 0:
            self.direction = 'up'
            self.last_direction = 'up'
        elif dy > 0:
            self.direction = 'down'
            self.last_direction = 'down'

        if not environment.is_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y

    def get_current_sprite(self):
        sprite = self.sprites[self.direction][self.animation_frame]
        if self.last_direction == 'right':
            return pygame.transform.flip(sprite, True, False)
        return sprite

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.map = [[0 for _ in range(width)] for _ in range(height)]
        self.generate_obstacles()

    def generate_obstacles(self):
        # Generate borders
        for x in range(self.width):
            self.map[0][x] = 1
            self.map[self.height - 1][x] = 1
        for y in range(self.height):
            self.map[y][0] = 1
            self.map[y][self.width - 1] = 1

        # Generate random obstacles
        for _ in range(50):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            self.map[y][x] = 1

    def is_collision(self, x, y):
        tile_x = int(x // TILE_SIZE)
        tile_y = int(y // TILE_SIZE)
        return self.map[tile_y][tile_x] == 1

class Display:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)
        self.grass_tile = pygame.image.load('assets/town_rpg_pack/graphics/elements/grass-tile.png').convert_alpha()

    def draw_environment(self, environment, player):
        self.screen.fill(GROUND_COLOR)

        visible_tiles_x = SCREEN_WIDTH // TILE_SIZE
        visible_tiles_y = SCREEN_HEIGHT // TILE_SIZE

        start_x = max(0, int(player.x // TILE_SIZE - visible_tiles_x // 2))
        start_y = max(0, int(player.y // TILE_SIZE - visible_tiles_y // 2))
        end_x = min(environment.width, start_x + visible_tiles_x)
        end_y = min(environment.height, start_y + visible_tiles_y)

        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                screen_x = (x - start_x) * TILE_SIZE
                screen_y = (y - start_y) * TILE_SIZE
                self.screen.blit(self.grass_tile, (screen_x, screen_y))
                if environment.map[y][x] == 1:
                    pygame.draw.rect(self.screen, OBSTACLE_COLOR, (screen_x, screen_y, TILE_SIZE, TILE_SIZE))

        player_sprite = player.get_current_sprite()

        player_screen_x = (player.x // TILE_SIZE - start_x) * TILE_SIZE
        player_screen_y = (player.y // TILE_SIZE - start_y) * TILE_SIZE
        self.screen.blit(player_sprite, (player_screen_x, player_screen_y))
    def draw_menu(self):
        self.screen.fill(BLACK)
        title = self.font.render("The Dispossessed", True, WHITE)
        play_button = self.font.render("Play", True, WHITE)

        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, SCREEN_HEIGHT // 3))
        self.screen.blit(play_button, (SCREEN_WIDTH // 2 - play_button.get_width() // 2, SCREEN_HEIGHT // 2))


def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("The Dispossessed")
    clock = pygame.time.Clock()

    environment = Environment(100, 100)
    player = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    display = Display(screen)

    game_state = "menu"

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Get the time passed since last frame in seconds

        # Store previous position
        player.prev_x = player.x
        player.prev_y = player.y

        # Handle input and move player
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * PLAYER_SPEED
        dy = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * PLAYER_SPEED
        player.move(dx * dt, dy * dt, environment) # Get the time passed since last frame in seconds

        player.update(dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and game_state == "menu":
                mouse_pos = pygame.mouse.get_pos()
                play_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 - 20, 100, 40)
                if play_button_rect.collidepoint(mouse_pos):
                    game_state = "playing"

        if game_state == "menu":
            display.draw_menu()
        elif game_state == "playing":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player.move(-PLAYER_SPEED, 0, environment)
            if keys[pygame.K_RIGHT]:
                player.move(PLAYER_SPEED, 0, environment)
            if keys[pygame.K_UP]:
                player.move(0, -PLAYER_SPEED, environment)
            if keys[pygame.K_DOWN]:
                player.move(0, PLAYER_SPEED, environment)

            display.draw_environment(environment, player)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

################## 

class TileSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Tile Selector")
        self.selected_tiles = {}
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Canvas and scrollbars for tiles
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(canvas_frame)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid layout
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Tile frame inside canvas
        self.tile_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.tile_frame, anchor=tk.NW)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        open_button = ttk.Button(button_frame, text="Open Image", command=self.open_image)
        open_button.pack(side=tk.LEFT, padx=5)

        save_button = ttk.Button(button_frame, text="Save Selected Tiles", command=self.save_tiles)
        save_button.pack(side=tk.LEFT, padx=5)

    def open_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")])

        if self.image_path:
            self.load_tiles()

    def load_tiles(self):
        # Clear existing tiles
        for widget in self.tile_frame.winfo_children():
            widget.destroy()

        img = Image.open(self.image_path)
        width, height = img.size
        tiles_x = width // 16
        tiles_y = height // 16

        for y in range(tiles_y):
            for x in range(tiles_x):
                left = x * 16
                top = y * 16
                right = left + 16
                bottom = top + 16
                tile = img.crop((left, top, right, bottom))

                # Resize for better visibility
                tile = tile.resize((32, 32), Image.NEAREST)
                photo = ImageTk.PhotoImage(tile)

                label = tk.Label(self.tile_frame, image=photo)
                label.image = photo  # Keep a reference
                label.grid(row=y, column=x, padx=2, pady=2)
                label.bind("<Button-1>", lambda e, x=x, y=y: self.tile_clicked(x, y))

        # Update canvas scroll region
        self.tile_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def tile_clicked(self, x, y):
        tile_name = tk.simpledialog.askstring("Name Tile", f"Enter name for tile ({x}, {y}):")
        if tile_name:
            self.selected_tiles[(x, y)] = tile_name
            print(f"Tile ({x}, {y}) named: {tile_name}")

    def save_tiles(self):
        if not self.selected_tiles:
            messagebox.showinfo("No Tiles", "No tiles have been selected.")
            return

        if not self.image_path:
            messagebox.showerror("Error", "No image loaded.")
            return

        # Create subdirectory for saving tiles
        base_dir = os.path.dirname(self.image_path)
        file_name = os.path.splitext(os.path.basename(self.image_path))[0]
        save_dir = os.path.join(base_dir, file_name + "_tiles")
        os.makedirs(save_dir, exist_ok=True)

        # Open the original image
        img = Image.open(self.image_path)

        # Save selected tiles
        for (x, y), name in self.selected_tiles.items():
            left = x * 16
            top = y * 16
            right = left + 16
            bottom = top + 16
            tile = img.crop((left, top, right, bottom))

            tile_path = os.path.join(save_dir, f"{name}.png")
            tile.save(tile_path)
            print(f"Tile saved as {tile_path}")

        messagebox.showinfo("Success", "All selected tiles have been saved.")


def main2():
    root = tk.Tk()
    app = TileSelector(root)
    root.mainloop()


if __name__ == "__main__":
    main2()
    # main()