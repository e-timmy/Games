import math

# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Player settings
PLAYER_START_SIZE = 20
PLAYER_ACCELERATION = 0.2
PLAYER_MAX_SPEED = 5
PLAYER_FRICTION = 0.02
PLAYER_ROTATION_SPEED = 5

# Bubble settings
INITIAL_BUBBLE_COUNT = 10
MAX_TOTAL_BUBBLES = 15
MIN_BUBBLE_SIZE = 5
MAX_BUBBLE_SIZE = 40
MIN_BUBBLE_SPEED = 0.5
MAX_BUBBLE_SPEED = 2
WOBBLE_AMOUNT = 0.5
MIN_BUBBLE_DISTANCE = 50

# Bullet settings
BULLET_SIZE = 5
BULLET_SPEED = 4
SHOOT_COOLDOWN = 500  # Milliseconds between shots
BULLET_WOBBLE_AMOUNT = 0.2  # Less wobble than regular bubbles

# Absorption mechanics
ABSORPTION_RATE = 0.05  # How quickly bubbles are absorbed (size units per frame)
STICKY_FACTOR = 0.15  # How strongly bubbles influence each other's velocity during absorption

# Energy/Mass conservation settings
SHOT_MASS_COST_RATIO = 0.01  # 10% of bullet size comes from player
MIN_PLAYER_SIZE = 10
PROPULSION_BUBBLE_COUNT = 3  # Number of bubbles created per thrust
MIN_PROPULSION_SIZE = 2
MAX_PROPULSION_SIZE = 4
PROPULSION_SPEED = 3
PROPULSION_MASS_RATIO = 0.002  # 2% of player mass converted to propulsion bubbles
MIN_BULLET_SIZE = 5
MAX_BULLET_SIZE = 5
BULLET_SIZE_VARIANCE = 0.3  # ±30% variance in bullet size