# constants.py
# Window settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CAPTION = "Garden Minesweeper"

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
LIGHT_GREEN = (144, 238, 144)
GRASS_GREEN = (86, 176, 86)
DIRT_BROWN = (139, 69, 19)
GRAY = (128, 128, 128)
RED = (255, 50, 50)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
DARK_GRAY = (64, 64, 64)
SAFE_GREEN = (50, 255, 50)
GRID_LINE_COLOR = (0, 0, 0, 30)

# Game settings
MAX_CELL_SIZE = 30
MIN_CELL_SIZE = 15
GRID_PADDING = 40

# Metal detector settings
METAL_DETECTOR_SIZE = 50
FLASH_DURATION = 500  # milliseconds

# Menu settings
DIFFICULTY_BUTTON_RADIUS = 30
DIFFICULTY_SPACING = 100
TOOLTIP_PADDING = 10

# Number colors
NUMBER_COLORS = {
    1: (0, 0, 255),     # Blue
    2: (0, 128, 0),     # Green
    3: (255, 0, 0),     # Red
    4: (0, 0, 128),     # Dark Blue
    5: (128, 0, 0),     # Dark Red
    6: (0, 128, 128),   # Teal
    7: (128, 0, 128),   # Purple
    8: (128, 128, 128)  # Gray
}