import pygame
from map_generator import MapGenerator
from player import Player
from camera import Camera
from renderer import Renderer


class Game:
    """Main game class that controls the game loop and state."""

    def __init__(self):
        """Initialize the game state and components."""
        # Screen setup
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("2D RPG")

        # Game settings
        self.tile_size = 32
        self.map_width = 100
        self.map_height = 100
        self.running = True
        self.clock = pygame.time.Clock()

        # Generate map
        self.map_generator = MapGenerator(self.map_width, self.map_height)
        self.map_data = self.map_generator.generate()
        self.map_generator.print_map()

        # Create player at center of map
        center_x, center_y = self.map_width // 2, self.map_height // 2
        self.player = Player(center_x, center_y)

        # Create camera
        self.camera = Camera(
            self.map_width,
            self.map_height,
            self.screen_width,
            self.screen_height,
            self.tile_size
        )

        # Create renderer
        self.renderer = Renderer(self.screen, self.tile_size)

    def handle_input(self):
        """Handle player input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Handle key press events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player.toggle_sprint(True)

            # Handle key release events
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    self.player.toggle_sprint(False)

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0

        if keys[pygame.K_UP]:
            dy = -1
        if keys[pygame.K_DOWN]:
            dy = 1
        if keys[pygame.K_LEFT]:
            dx = -1
        if keys[pygame.K_RIGHT]:
            dx = 1

        if dx != 0 or dy != 0:
            self.player.move(dx, dy, self.map_data)

    def update(self):
        """Update the game state."""
        # Center camera on player
        self.camera.center_on(self.player.x, self.player.y)

    def render(self):
        """Render the current game state."""
        self.renderer.render(self.map_data, self.player, self.camera)

    def run(self):
        """Run the main game loop."""
        while self.running:
            self.handle_input()
            self.update()
            self.render()
            self.clock.tick(60)  # 60 FPS