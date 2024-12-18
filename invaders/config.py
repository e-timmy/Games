# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Player settings
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 30
PLAYER_SPEED = 5
PLAYER_LIVES = 3

# Bullet settings
BULLET_SPEED = 7
BULLET_SIZE = 4

# Shield settings
SHIELD_PIECE_SIZE = 4
SHIELD_WIDTH = 88  # Must be divisible by SHIELD_PIECE_SIZE
SHIELD_HEIGHT = 64  # Must be divisible by SHIELD_PIECE_SIZE
NUM_SHIELDS = 4

# Scoring
ENEMY_SCORES = {
    'basic': 10,
    'medium': 20,
    'advanced': 30
}

# Enemy settings
ENEMY_WIDTH = 25
ENEMY_HEIGHT = 20
ENEMY_MOVE_DOWN = 30
ENEMY_START_X = 50
ENEMY_START_Y = 50

# Enemy properties
ENEMY_ROWS = {
    'advanced': {'row': 0, 'shoot_chance': 0.004, 'size_factor': 0.8},
    'medium': {'row': 1, 'shoot_chance': 0.003, 'size_factor': 0.9},
    'basic': {'row': 2, 'shoot_chance': 0.002, 'size_factor': 1.0}
}

ENEMY_ROW_HEIGHT = 40

# Enemy formation settings
ENEMIES_PER_ROW = 14
ENEMY_SPACING = 10
ENEMY_FORMATION_WIDTH = ENEMY_WIDTH * ENEMIES_PER_ROW + ENEMY_SPACING * (ENEMIES_PER_ROW - 1)
ENEMY_FORMATION_START_X = (SCREEN_WIDTH - ENEMY_FORMATION_WIDTH) // 2

# Player respawn settings
RESPAWN_BLINK_TIME = 150  # milliseconds per blink
RESPAWN_DURATION = 3000   # total invulnerability time in milliseconds
# Update only this section in config.py
PLAYER_START_Y = SCREEN_HEIGHT - 15  # Adjusted to be lower on the screen

RESPAWN_POSITIONS = [
    (SCREEN_WIDTH // 4, PLAYER_START_Y),
    (SCREEN_WIDTH // 2, PLAYER_START_Y),
    (3 * SCREEN_WIDTH // 4, PLAYER_START_Y)
]

# New constant for enemy speed increase
ENEMY_SPEED_INCREASE_FACTOR = 1.1  # 10% speed increase each time they move down

# Shield shape (1 represents a piece, 0 represents empty space)
SHIELD_SHAPE = [
    [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
]

LEVEL_SETTINGS = {
    1: {
        'speed_multiplier': 1.0,
        'enemies_per_row': 8,
        'shoot_frequency_multiplier': 1.0,
        'points_multiplier': 1.0
    },
    2: {
        'speed_multiplier': 1.2,
        'enemies_per_row': 9,
        'shoot_frequency_multiplier': 1.2,
        'points_multiplier': 1.5
    },
    3: {
        'speed_multiplier': 1.4,
        'enemies_per_row': 10,
        'shoot_frequency_multiplier': 1.4,
        'points_multiplier': 2.0
    },
    # Add more levels as needed
}

"""
The issues : 
- player isn't flashing when they're momentarily invulnerable after losing a life
- formation messes up when enemies move down/hit edge - notable the furthest bottom right enemy moves out of formation - whilst the remaining seem pretty fine. 
- formation doesn't react properly to edge. Currently they only shift down each hits the edge - but they should all move down together when the furthest column hits the edge - this prevents them from overlapping
- the formation moves too fast.

I also want animations for each enemy when just moving
HUD is weird - with player representations appearing on top of one another
No loss state for agents getting to bottom (top of shield)

Animation of enemies coming into position from top of screen.
"""