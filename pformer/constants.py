import pygame

# Game constants
WIDTH, HEIGHT = 400, 600
FPS = 60
PLAYER_SIZE = 30
PLATFORM_HEIGHT = 10
PLATFORM_MIN_WIDTH = 70
PLATFORM_MAX_WIDTH = 150
GRAVITY = 0.3
JUMP_POWER = -15
HORIZONTAL_SPEED = 5
SCROLL_THRESHOLD = 200
PLATFORM_GAP = 100
GROUND_HEIGHT = 20
HIGH_SCORES_FILE = "jumper_scores.txt"
MOVING_PLATFORM_SPEED = 2
CAMERA_SMOOTHNESS = 0.05

# Game states
TITLE_SCREEN = 0
PLAYING = 1
GAME_OVER = 2
ENTER_NAME = 3

# Platform types
NORMAL_PLATFORM = 0
MOVING_PLATFORM = 1
BREAKABLE_PLATFORM = 2

# Colors
BACKGROUND_COLOR = (0, 0, 40)
PLAYER_COLOR = (255, 0, 128)
PLATFORM_COLORS = [
    (0, 255, 255),  # Cyan - Normal
    (255, 255, 0),  # Yellow - Moving
    (255, 100, 100)  # Red-ish - Breakable
]
GROUND_COLOR = (50, 50, 200)
TEXT_COLOR = (255, 255, 255)
TITLE_COLOR = (0, 255, 255)

# Initialize pygame fonts
pygame.font.init()
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 20)
big_font = pygame.font.Font(None, 72)
title_font = pygame.font.Font(None, 100)

# Debug settings
MAX_DEBUG_LOGS = 10
debug_log = []

def log_debug(message):
    print(f"DEBUG: {message}")
    debug_log.append(message)
    if len(debug_log) > MAX_DEBUG_LOGS:
        debug_log.pop(0)

def lerp(a, b, t):
    return a + (b - a) * t