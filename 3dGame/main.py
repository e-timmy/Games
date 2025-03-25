import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Text Adventure Prototype")

# Room dimensions
ROOM_SIZE = 10.0

# Door dimensions
DOOR_WIDTH = 2.0
DOOR_HEIGHT = 3.0

# Colors (RGBA)
RED = (0.8, 0.2, 0.2, 1.0)
GREEN = (0.2, 0.8, 0.2, 1.0)
BLUE = (0.2, 0.2, 0.8, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
GRAY = (0.5, 0.5, 0.5, 1.0)
LIGHT_GRAY = (0.8, 0.8, 0.8, 1.0)

# Player settings
player_pos = [0.0, 0.0, 5.0]  # Start in the first room
player_height = 1.8  # Player's eye height
yaw = 0.0  # horizontal rotation (in degrees)
pitch = 0.0  # vertical rotation (in degrees)
pitch_limit = 80.0  # limit for looking up/down
mouse_sensitivity = 0.2
move_speed = 0.1
collision_radius = 0.5  # Player collision detection radius

# Game state
rooms = [
    {"pos": [0, 0, 0], "size": ROOM_SIZE,
     "doors": [{"pos": [0, 0, -ROOM_SIZE / 2], "target_room": 1, "target_pos": [0, 0, ROOM_SIZE / 2 - 1]}]},
    {"pos": [0, 0, -ROOM_SIZE - 5], "size": ROOM_SIZE,
     "doors": [{"pos": [0, 0, ROOM_SIZE / 2], "target_room": 0, "target_pos": [0, 0, -ROOM_SIZE / 2 + 1]}]}
]

# Textures
textures = {}

# Light position (above the player)
light_position = [0.0, ROOM_SIZE / 2 - 1.0, 0.0, 1.0]

# Mouse handling
pygame.mouse.set_visible(False)  # Hide mouse cursor
pygame.event.set_grab(True)  # Capture mouse input


def create_procedural_textures():
    """Create procedural textures for walls, floors, and ceilings"""
    global textures

    # Create wall texture (brick-like)
    size = 256
    wall_texture = []

    for i in range(size):
        for j in range(size):
            # Create brick pattern
            brick_height = 32
            brick_width = 64
            mortar_thickness = 4

            # Determine if we're on a brick or mortar
            brick_row = (i // brick_height) % 2
            in_horizontal_mortar = (i % brick_height) < mortar_thickness

            if brick_row == 0:
                in_vertical_mortar = (j % brick_width) < mortar_thickness
            else:
                # Offset bricks in alternating rows
                in_vertical_mortar = ((j + brick_width // 2) % brick_width) < mortar_thickness

            if in_horizontal_mortar or in_vertical_mortar:
                # Mortar color with some variation
                var = np.random.randint(-10, 10)
                wall_texture.extend([100 + var, 100 + var, 100 + var, 255])
            else:
                # Brick color with some variation
                r_var = np.random.randint(-20, 20)
                g_var = np.random.randint(-10, 10)
                b_var = np.random.randint(-10, 10)
                wall_texture.extend([180 + r_var, 90 + g_var, 80 + b_var, 255])

    # Create floor texture (wooden planks)
    floor_texture = []

    for i in range(size):
        for j in range(size):
            # Create wooden plank pattern
            plank_width = 32
            gap_thickness = 2

            in_gap = (j % plank_width) < gap_thickness

            if in_gap:
                # Gap color
                var = np.random.randint(-5, 5)
                floor_texture.extend([60 + var, 40 + var, 30 + var, 255])
            else:
                # Wood color with grain
                grain = (i + j // 3) % 32
                if grain < 2 or grain > 29:
                    r_var = np.random.randint(-30, 0)
                    g_var = np.random.randint(-20, 0)
                    b_var = np.random.randint(-10, 0)
                else:
                    r_var = np.random.randint(-10, 10)
                    g_var = np.random.randint(-10, 10)
                    b_var = np.random.randint(-10, 10)

                floor_texture.extend([160 + r_var, 120 + g_var, 80 + b_var, 255])

    # Create ceiling texture (stucco-like)
    ceiling_texture = []

    for i in range(size):
        for j in range(size):
            # Create a noisy stucco pattern
            noise = np.random.randint(-15, 15)
            ceiling_texture.extend([220 + noise, 220 + noise, 220 + noise, 255])

    # Generate texture IDs and upload texture data
    textures['wall'] = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textures['wall'])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes(wall_texture))
    glGenerateMipmap(GL_TEXTURE_2D)

    textures['floor'] = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textures['floor'])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes(floor_texture))
    glGenerateMipmap(GL_TEXTURE_2D)

    textures['ceiling'] = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textures['ceiling'])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes(ceiling_texture))
    glGenerateMipmap(GL_TEXTURE_2D)

    # Create door texture
    door_size = 128
    door_texture = []

    for i in range(door_size):
        for j in range(door_size):
            # Create wooden door pattern with panels
            panel_border = 16
            panel_size = door_size - 2 * panel_border

            in_panel = (panel_border <= i < panel_border + panel_size and
                        panel_border <= j < panel_border + panel_size)

            if in_panel:
                # Panel color
                r_var = np.random.randint(-15, 15)
                g_var = np.random.randint(-15, 15)
                b_var = np.random.randint(-15, 15)
                door_texture.extend([120 + r_var, 70 + g_var, 50 + b_var, 255])
            else:
                # Frame color (wooden)
                r_var = np.random.randint(-10, 10)
                g_var = np.random.randint(-10, 10)
                b_var = np.random.randint(-10, 10)
                door_texture.extend([160 + r_var, 100 + g_var, 70 + b_var, 255])

    # Generate texture ID and upload texture data
    textures['door'] = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textures['door'])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, door_size, door_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes(door_texture))
    glGenerateMipmap(GL_TEXTURE_2D)


def setup_opengl():
    """Setup the OpenGL state"""
    glClearColor(0.1, 0.1, 0.1, 1.0)

    # Setup depth testing
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Setup face culling (draw only front faces)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)

    # Setup blending for transparency
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Setup texturing
    glEnable(GL_TEXTURE_2D)

    # Setup lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    # Set light properties
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

    # Set material properties
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
    glMateriali(GL_FRONT, GL_SHININESS, 32)

    # Enable color material (use glColor for ambient and diffuse)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

    # Setup perspective
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(70, SCREEN_WIDTH / SCREEN_HEIGHT, 0.1, 100.0)


def update_view():
    """Update the view based on player position and orientation"""
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Apply pitch rotation (looking up/down)
    glRotatef(pitch, 1.0, 0.0, 0.0)
    # Apply yaw rotation (looking left/right)
    glRotatef(yaw, 0.0, 1.0, 0.0)

    # Move the world relative to the player position (accounting for player height)
    glTranslatef(-player_pos[0], -(player_pos[1] + player_height), -player_pos[2])

    # Update light position relative to the player
    glLightfv(GL_LIGHT0, GL_POSITION, [player_pos[0], player_pos[1] + player_height + 5.0, player_pos[2], 1.0])


def find_current_room():
    """Determine which room the player is in based on position"""
    for i, room in enumerate(rooms):
        room_pos = room["pos"]
        room_size = room["size"]

        # Check if player is within room boundaries (with some tolerance)
        if (abs(player_pos[0] - room_pos[0]) < room_size / 2 + 1 and
                abs(player_pos[1] - room_pos[1]) < room_size / 2 + 1 and
                abs(player_pos[2] - room_pos[2]) < room_size / 2 + 1):
            return i
    return 0  # Default to first room if not found


def draw_rooms():
    """Draw all rooms"""
    for room_index, room in enumerate(rooms):
        draw_room(room, room_index)


def draw_room(room, room_index):
    """Draw a single room with its doors"""
    room_pos = room["pos"]
    room_size = room["size"]
    half_size = room_size / 2

    # Draw room at its position
    glPushMatrix()
    glTranslatef(room_pos[0], room_pos[1], room_pos[2])

    # Floor
    glBindTexture(GL_TEXTURE_2D, textures['floor'])
    glBegin(GL_QUADS)
    glNormal3f(0.0, 1.0, 0.0)  # Normal pointing up
    glColor4f(*WHITE)
    glTexCoord2f(0, 0);
    glVertex3f(-half_size, -half_size, half_size)
    glTexCoord2f(4, 0);
    glVertex3f(half_size, -half_size, half_size)
    glTexCoord2f(4, 4);
    glVertex3f(half_size, -half_size, -half_size)
    glTexCoord2f(0, 4);
    glVertex3f(-half_size, -half_size, -half_size)
    glEnd()

    # Ceiling
    glBindTexture(GL_TEXTURE_2D, textures['ceiling'])
    glBegin(GL_QUADS)
    glNormal3f(0.0, -1.0, 0.0)  # Normal pointing down
    glColor4f(*WHITE)
    glTexCoord2f(0, 0);
    glVertex3f(-half_size, half_size, -half_size)
    glTexCoord2f(4, 0);
    glVertex3f(half_size, half_size, -half_size)
    glTexCoord2f(4, 4);
    glVertex3f(half_size, half_size, half_size)
    glTexCoord2f(0, 4);
    glVertex3f(-half_size, half_size, half_size)
    glEnd()

    # Walls (with doorways)
    draw_walls_with_doorways(room, half_size)

    glPopMatrix()


def draw_walls_with_doorways(room, half_size):
    """Draw the walls of a room with doorways"""
    doors = room.get("doors", [])

    # Calculate wall segments for each wall, accounting for doorways
    walls = {
        "front": {"vertices": [(-half_size, -half_size, -half_size), (half_size, -half_size, -half_size),
                               (half_size, half_size, -half_size), (-half_size, half_size, -half_size)],
                  "normal": (0, 0, 1), "segments": []},
        "back": {"vertices": [(half_size, -half_size, half_size), (-half_size, -half_size, half_size),
                              (-half_size, half_size, half_size), (half_size, half_size, half_size)],
                 "normal": (0, 0, -1), "segments": []},
        "left": {"vertices": [(-half_size, -half_size, half_size), (-half_size, -half_size, -half_size),
                              (-half_size, half_size, -half_size), (-half_size, half_size, half_size)],
                 "normal": (1, 0, 0), "segments": []},
        "right": {"vertices": [(half_size, -half_size, -half_size), (half_size, -half_size, half_size),
                               (half_size, half_size, half_size), (half_size, half_size, -half_size)],
                  "normal": (-1, 0, 0), "segments": []}
    }

    # Process doors and create wall segments
    for door in doors:
        door_pos = door["pos"]
        door_width_half = DOOR_WIDTH / 2
        door_height = DOOR_HEIGHT

        # Determine which wall the door is on
        wall_key = None
        local_door_pos = door_pos.copy()  # Door position relative to room center

        if abs(abs(local_door_pos[2]) - half_size) < 0.1:
            # Door is on front or back wall
            wall_key = "front" if local_door_pos[2] < 0 else "back"
            # Calculate door segments
            x_start = max(-half_size, local_door_pos[0] - door_width_half)
            x_end = min(half_size, local_door_pos[0] + door_width_half)

            if x_start > -half_size:
                # Left wall segment
                walls[wall_key]["segments"].append({
                    "type": "wall",
                    "vertices": [
                        (-half_size, -half_size, local_door_pos[2]),
                        (x_start, -half_size, local_door_pos[2]),
                        (x_start, -half_size + door_height, local_door_pos[2]),
                        (-half_size, -half_size + door_height, local_door_pos[2])
                    ]
                })

            if x_end < half_size:
                # Right wall segment
                walls[wall_key]["segments"].append({
                    "type": "wall",
                    "vertices": [
                        (x_end, -half_size, local_door_pos[2]),
                        (half_size, -half_size, local_door_pos[2]),
                        (half_size, -half_size + door_height, local_door_pos[2]),
                        (x_end, -half_size + door_height, local_door_pos[2])
                    ]
                })

            # Top wall segment
            walls[wall_key]["segments"].append({
                "type": "wall",
                "vertices": [
                    (x_start, -half_size + door_height, local_door_pos[2]),
                    (x_end, -half_size + door_height, local_door_pos[2]),
                    (x_end, half_size, local_door_pos[2]),
                    (x_start, half_size, local_door_pos[2])
                ]
            })

            # Add doorway (no actual door, just for reference)
            walls[wall_key]["segments"].append({
                "type": "doorway",
                "vertices": [
                    (x_start, -half_size, local_door_pos[2]),
                    (x_end, -half_size, local_door_pos[2]),
                    (x_end, -half_size + door_height, local_door_pos[2]),
                    (x_start, -half_size + door_height, local_door_pos[2])
                ],
                "door": door
            })

        elif abs(abs(local_door_pos[0]) - half_size) < 0.1:
            # Door is on left or right wall
            wall_key = "left" if local_door_pos[0] < 0 else "right"
            # Calculate door segments
            z_start = max(-half_size, local_door_pos[2] - door_width_half)
            z_end = min(half_size, local_door_pos[2] + door_width_half)

            if z_start > -half_size:
                # Front wall segment
                walls[wall_key]["segments"].append({
                    "type": "wall",
                    "vertices": [
                        (local_door_pos[0], -half_size, -half_size),
                        (local_door_pos[0], -half_size, z_start),
                        (local_door_pos[0], -half_size + door_height, z_start),
                        (local_door_pos[0], -half_size + door_height, -half_size)
                    ]
                })

            if z_end < half_size:
                # Back wall segment
                walls[wall_key]["segments"].append({
                    "type": "wall",
                    "vertices": [
                        (local_door_pos[0], -half_size, z_end),
                        (local_door_pos[0], -half_size, half_size),
                        (local_door_pos[0], -half_size + door_height, half_size),
                        (local_door_pos[0], -half_size + door_height, z_end)
                    ]
                })

            # Top wall segment
            walls[wall_key]["segments"].append({
                "type": "wall",
                "vertices": [
                    (local_door_pos[0], -half_size + door_height, z_start),
                    (local_door_pos[0], -half_size + door_height, z_end),
                    (local_door_pos[0], half_size, z_end),
                    (local_door_pos[0], half_size, z_start)
                ]
            })

            # Add doorway (no actual door, just for reference)
            walls[wall_key]["segments"].append({
                "type": "doorway",
                "vertices": [
                    (local_door_pos[0], -half_size, z_start),
                    (local_door_pos[0], -half_size, z_end),
                    (local_door_pos[0], -half_size + door_height, z_end),
                    (local_door_pos[0], -half_size + door_height, z_start)
                ],
                "door": door
            })

    # Draw walls
    glBindTexture(GL_TEXTURE_2D, textures['wall'])
    for wall_key, wall in walls.items():
        # Check if wall has segments (doors)
        if wall["segments"]:
            # Draw wall segments
            for segment in wall["segments"]:
                if segment["type"] == "wall":
                    glBegin(GL_QUADS)
                    glNormal3f(*wall["normal"])
                    glColor4f(*WHITE)
                    for i, vertex in enumerate(segment["vertices"]):
                        # Calculate texture coordinates
                        u = (i == 1 or i == 2) ? 1: 0
                        v = (i == 2 or i == 3) ? 1: 0
                        glTexCoord2f(u, v)
                        glVertex3f(*vertex)
                    glEnd()
                # Doorways are empty spaces, don't draw them
        else:
            # Draw full wall
            glBegin(GL_QUADS)
            glNormal3f(*wall["normal"])
            glColor4f(*WHITE)
            for i, vertex in enumerate(wall["vertices"]):
                u = (i == 1 or i == 2) ? 2: 0
                v = (i == 2 or i == 3) ? 2: 0
                glTexCoord2f(u, v)
                glVertex3f(*vertex)
            glEnd()


def check_door_collision():
    """Check if player is crossing through a doorway"""
    current_room_index = find_current_room()
    current_room = rooms[current_room_index]

    for door in current_room.get("doors", []):
        door_pos = [
            current_room["pos"][0] + door["pos"][0],
            current_room["pos"][1] + door["pos"][1],
            current_room["pos"][2] + door["pos"][2]
        ]

        # Check distance from player to door
        door_dist = math.sqrt(
            (player_pos[0] - door_pos[0]) ** 2 +
            (player_pos[1] - door_pos[1]) ** 2 +
            (player_pos[2] - door_pos[2]) ** 2
        )

        # If player is close to door, check if they're crossing the threshold
        if door_dist < 2.0:
            # Get target room
            target_room_index = door["target_room"]
            if target_room_index != current_room_index:
                # Check which side of the door the player is on
                target_room = rooms[target_room_index]

                # Calculate normalized door direction vector
                if abs(door["pos"][0]) > 0:  # Door on left/right wall
                    door_dir = [1, 0, 0] if door["pos"][0] > 0 else [-1, 0, 0]
                else:  # Door on front/back wall
                    door_dir = [0, 0, 1] if door["pos"][2] > 0 else [0, 0, -1]

                # Project player position onto door plane
                proj = (player_pos[0] - door_pos[0]) * door_dir[0] + (player_pos[2] - door_pos[2]) * door_dir[2]

                # Check if player is crossing the threshold
                if abs(proj) < 0.2:  # Within threshold
                    return None  # No transition needed, player is at the doorway
                elif (proj > 0 and current_room_index != target_room_index) or (
                        proj < 0 and current_room_index == target_room_index):
                    # Player is crossing from current room to target room
                    return None  # No need to teleport, player can walk through


def move_player(dx, dz):
    """Move player and handle collisions"""
    global player_pos

    # Calculate movement vector in world space
    yaw_rad = math.radians(yaw)

    # Calculate forward and right vectors
    forward_x = -math.sin(yaw_rad)
    forward_z = -math.cos(yaw_rad)
    right_x = -math.cos(yaw_rad)
    right_z = math.sin(yaw_rad)

    # Calculate new position
    new_x = player_pos[0] + (forward_x * dz + right_x * dx) * move_speed
    new_z = player_pos[2] + (forward_z * dz + right_z * dx) * move_speed

    # Check for wall collisions
    current_room_index = find_current_room()
    current_room = rooms[current_room_index]
    room_pos = current_room["pos"]
    room_size = current_room["size"]
    half_size = room_size / 2

    # Check collision with walls
    can_move_x = True
    can_move_z = True

    # Room boundaries with offset for player size
    min_x = room_pos[0] - half_size + collision_radius
    max_x = room_pos[0] + half_size - collision_radius
    min_z = room_pos[2] - half_size + collision_radius
    max_z = room_pos[2] + half_size - collision_radius

    # Check doorways (to allow movement through them)
    for door in current_room.get("doors", []):
        door_pos = door["pos"]
        door_width_half = DOOR_WIDTH / 2

        # Adjust collision boundaries for doorways
        door_world_pos = [room_pos[0] + door_pos[0], room_pos[1] + door_pos[1], room_pos[2] + door_pos[2]]

        # Check if door is on X or Z axis
        if abs(door_pos[0]) > 0.1:  # Door on left or right wall (X axis)
            # Allow passing through if player is at the doorway
            if (abs(door_world_pos[0] - new_x) < 1.0 and
                    abs(door_world_pos[2] - player_pos[2]) < door_width_half):
                if door_pos[0] < 0:  # Left wall
                    min_x = min(min_x, room_pos[0] - half_size - collision_radius)
                else:  # Right wall
                    max_x = max(max_x, room_pos[0] + half_size + collision_radius)

        elif abs(door_pos[2]) > 0.1:  # Door on front or back wall (Z axis)
            # Allow passing through if player is at the doorway
            if (abs(door_world_pos[2] - new_z) < 1.0 and
                    abs(door_world_pos[0] - player_pos[0]) < door_width_half):
                if door_pos[2] < 0:  # Front wall
                    min_z = min(min_z, room_pos[2] - half_size - collision_radius)
                else:  # Back wall
                    max_z = max(max_z, room_pos[2] + half_size + collision_radius)

    # Apply boundary constraints
    new_x = max(min_x, min(max_x, new_x))
    new_z = max(min_z, min(max_z, new_z))

    # Update player position
    player_pos[0] = new_x
    player_pos[2] = new_z

    # Check if player walked through a doorway
    check_door_collision()


def draw_crosshair():
    """Draw a simple crosshair in the center of the screen"""
    # Disable lighting and depth testing for 2D rendering
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)

    # Switch to 2D orthographic projection for the crosshair
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Draw crosshair
    glDisable(GL_TEXTURE_2D)
    glBegin(GL_LINES)
    glColor4f(1.0, 1.0, 1.0, 0.7)
    # Horizontal line
    glVertex2f(SCREEN_WIDTH / 2 - 10, SCREEN_HEIGHT / 2)
    glVertex2f(SCREEN_WIDTH / 2 + 10, SCREEN_HEIGHT / 2)
    # Vertical line
    glVertex2f(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 10)
    glVertex2f(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 10)
    glEnd()

    # Restore 3D projection
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    # Re-enable lighting and depth testing
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)


def draw_ui():
    """Draw UI elements like room information and controls"""
    # Disable lighting and depth testing for 2D rendering
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Render text using pygame
    font = pygame.font.SysFont(None, 24)
    current_room_index = find_current_room()
    room_text = font.render(f"Room {current_room_index + 1}", True, (255, 255, 255))
    controls_text = font.render("Arrow Keys: Move | Mouse: Look | ESC: Exit", True, (255, 255, 255))

    # Convert surfaces to OpenGL textures
    draw_text(room_text, 10, 10)
    draw_text(controls_text, SCREEN_WIDTH - controls_text.get_width() - 10, 10)

    # Restore 3D projection
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    # Re-enable lighting and depth testing
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)


def draw_text(text_surface, x, y):
    """Draw a text surface at the given position"""
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    width, height = text_surface.get_width(), text_surface.get_height()

    # Create texture
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Draw textured quad
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glBegin(GL_QUADS)
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glTexCoord2f(0, 0);
    glVertex2f(x, y)
    glTexCoord2f(1, 0);
    glVertex2f(x + width, y)
    glTexCoord2f(1, 1);
    glVertex2f(x + width, y + height)
    glTexCoord2f(0, 1);
    glVertex2f(x, y + height)
    glEnd()

    # Delete texture
    glDeleteTextures(1, [texture])


def main():
    global yaw, pitch, player_pos

    setup_opengl()
    create_procedural_textures()

    # Set the center position for mouse
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    pygame.mouse.set_pos(center_x, center_y)

    # Main game loop
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Handle keyboard input for movement
        keys = pygame.key.get_pressed()
        dx, dz = 0, 0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dz = 1  # Move forward
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dz = -1  # Move backward
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx = -1  # Strafe left
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx = 1  # Strafe right

        # Move the player
        if dx != 0 or dz != 0:
            move_player(dx, dz)

        # Get mouse movement for looking around
        mouse_dx, mouse_dy = pygame.mouse.get_rel()

        # Update yaw (left/right)
        yaw += mouse_dx * mouse_sensitivity
        yaw %= 360  # Keep in the range [0, 360)

        # Update pitch (up/down) - no inversion
        pitch -= mouse_dy * mouse_sensitivity
        pitch = max(-pitch_limit, min(pitch, pitch_limit))  # Apply limits

        # Reset mouse position to center if it gets too close to the edge
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if abs(mouse_x - center_x) > 100 or abs(mouse_y - center_y) > 100:
            pygame.mouse.set_pos(center_x, center_y)

        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update view
        update_view()

        # Draw the rooms
        draw_rooms()

        # Draw crosshair
        draw_crosshair()

        # Draw UI
        draw_ui()

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()