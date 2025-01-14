SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TILE_SIZE = 32
CHAR_WIDTH = 20
CHAR_HEIGHT = 40
GROUND_LEVEL = (SCREEN_HEIGHT // TILE_SIZE - 3) * TILE_SIZE  # Align with tile grid

# Colors
BACKGROUND_COLOR = (100, 100, 255)
GROUND_COLOR = (100, 70, 40)
PLAYER_COLOR = (255, 200, 0)
AI_COLOR = (200, 0, 255)
STONE_COLOR = (128, 128, 128)

# Tile types
EMPTY_TILE = 0
STONE_TILE = 1

# Physics
GRAVITY = 0.5
JUMP_SPEED = -8  # Reduced to only jump one tile height
MOVE_SPEED = 5

# Gameplay constraints
MAX_JUMP_HEIGHT = TILE_SIZE  # Maximum jump height of one tile

