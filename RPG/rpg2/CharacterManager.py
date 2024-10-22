import math

from non_player_character import NPC1, NPC2, NPC3, Alien, Slime, Bat, Ghost, Spider
import random
from constants import TILE_SIZE, CHARACTER_SCALE_FACTOR


class NPCManager:
    def __init__(self, environment):
        self.environment = environment
        self.npcs = []
        self.spawn_npcs()

    def spawn_npcs(self):
        mid_x = self.environment.width // 2
        mid_y = self.environment.height // 2

        # Spawn NPCs in specific quadrants
        npc_configs = [
            (NPC1, mid_x + mid_x // 2, mid_y // 2),  # North-east
            (NPC2, mid_x + mid_x // 2, mid_y + mid_y // 2),  # South-east
            (NPC3, mid_x // 2, mid_y + mid_y // 2),  # South-west
            (Alien, mid_x // 2, mid_y // 2),  # North-west
        ]

        for npc_class, x, y in npc_configs:
            npc = npc_class()
            npc.x = x * TILE_SIZE
            npc.y = y * TILE_SIZE
            self.npcs.append(npc)

    def update(self):
        for npc in self.npcs:
            npc.update()
            npc.patrol_with_stops(self.environment)


class MonsterManager:
    def __init__(self, environment):
        self.environment = environment
        self.monsters = []
        self.bosses = []
        self.spawn_monsters()
        self.spawn_bosses()

    def spawn_monsters(self):
        mid_x = self.environment.width // 2
        mid_y = self.environment.height // 2

        # Define quadrant boundaries
        quadrant_configs = [
            # (Monster Class, x_min, x_max, y_min, y_max)
            (Slime, mid_x, self.environment.width - 1, 0, mid_y - 1),  # NE
            (Bat, mid_x, self.environment.width - 1, mid_y, self.environment.height - 1),  # SE
            (Ghost, 0, mid_x - 1, mid_y, self.environment.height - 1),  # SW
            (Spider, 0, mid_x - 1, 0, mid_y - 1)  # NW
        ]

        # Spawn monsters in each quadrant
        for monster_class, x_min, x_max, y_min, y_max in quadrant_configs:
            for _ in range(3):
                monster = monster_class()
                # Add buffer from edges
                buffer = 2  # tiles from edge
                monster.x = random.randint(x_min + buffer, x_max - buffer) * TILE_SIZE
                monster.y = random.randint(y_min + buffer, y_max - buffer) * TILE_SIZE
                monster.area_center_x = monster.x
                monster.area_center_y = monster.y
                self.monsters.append(monster)

    def spawn_bosses(self):
        # Map quadrant positions to monster types
        quadrant_boss_types = {
            (0, 0): Spider,  # NW
            (1, 0): Slime,  # NE
            (0, 1): Ghost,  # SW
            (1, 1): Bat  # SE
        }

        quadrant_width = self.environment.width // 2
        quadrant_height = self.environment.height // 2

        for i, quadrant in enumerate(self.environment.quadrants):
            if quadrant.boss_area is None:
                continue

            # Calculate quadrant indices
            qx = i % 2
            qy = i // 2

            # Get the correct boss type for this quadrant
            boss_class = quadrant_boss_types.get((qx, qy))
            if not boss_class:
                continue

            boss = boss_class()  # Slower base speed for bosses

            # Get boss area center coordinates
            boss_x, boss_y = quadrant.boss_area

            # Convert local quadrant coordinates to global coordinates
            global_x = quadrant.x + boss_x
            global_y = quadrant.y + boss_y

            # Set boss properties
            boss.x = global_x * TILE_SIZE
            boss.y = global_y * TILE_SIZE
            boss.area_center_x = boss.x
            boss.area_center_y = boss.y
            boss.set_as_boss()

            self.bosses.append(boss)

    def update(self, player):
        for monster in self.monsters + self.bosses:
            distance_to_player = math.sqrt((monster.x - player.x) ** 2 + (monster.y - player.y) ** 2)

            if distance_to_player <= monster.perception_range:
                self.chase_player(monster, player)
            else:
                monster.random_movement(self.environment)

            monster.update()

    def chase_player(self, monster, player):
        dx = player.x - monster.x
        dy = player.y - monster.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance != 0:
            dx = dx / distance
            dy = dy / distance

        new_x = monster.x + dx * monster.speed
        new_y = monster.y + dy * monster.speed

        if not self.environment.is_collision(new_x, new_y):
            monster.x = new_x
            monster.y = new_y

        # Update direction based on movement
        if abs(dx) > abs(dy):
            monster.direction = 1 if dx < 0 else 2  # Left or Right
        else:
            monster.direction = 3 if dy < 0 else 0  # Up or Down

        monster.moving = True
