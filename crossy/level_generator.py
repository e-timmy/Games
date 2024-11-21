class DifficultyConfig:
    def __init__(self):
        # Base values
        self.base_lanes = 1
        self.base_speed_min = 1
        self.base_speed_max = 1.5
        self.base_spawn_rate = 120

        # Scaling factors (per level)
        self.lane_increase_rate = 0.5 # How quickly lanes are added
        self.speed_increase_rate = 0.1  # How quickly speed increases
        self.spawn_rate_decrease = 5  # How quickly spawn rate decreases

        # Caps
        self.max_lanes = 7
        self.max_speed = 4.5
        self.min_spawn_rate = 90

        # River specific
        self.river_speed_base = 1
        self.river_speed_increase = 0.15
        self.max_river_speed = 2.5
        self.river_spawn_rate_base = 180
        self.river_spawn_rate_decrease = 10
        self.min_river_spawn_rate = 120


class LevelConfig:
    def __init__(self):
        # Road settings
        self.num_lanes = 3
        self.spawn_rate = 120
        self.min_speed = 2
        self.max_speed = 3
        self.vehicle_types = ['car']

        # River settings
        self.has_river = False
        self.river_lanes = 0
        self.river_min_speed = 1
        self.river_max_speed = 2
        self.logs_per_lane = 0
        self.log_sizes = ['medium']
        self.river_spawn_rate = 180
        self.include_lilypads = False


def generate_level(level_number, difficulty=None):
    if difficulty is None:
        difficulty = DifficultyConfig()

    config = LevelConfig()

    # Calculate level-based values using difficulty config
    level_factor = level_number - 1

    # Road configuration
    config.num_lanes = min(
        difficulty.base_lanes + int(level_factor * difficulty.lane_increase_rate),
        difficulty.max_lanes
    )

    config.min_speed = min(
        difficulty.base_speed_min + (level_factor * difficulty.speed_increase_rate),
        difficulty.max_speed - 0.5
    )

    config.max_speed = min(
        difficulty.base_speed_max + (level_factor * difficulty.speed_increase_rate),
        difficulty.max_speed
    )

    config.spawn_rate = max(
        difficulty.base_spawn_rate - (level_factor * difficulty.spawn_rate_decrease),
        difficulty.min_spawn_rate
    )

    # Vehicle types progression
    if level_number >= 4:
        config.vehicle_types.append('truck')
    if level_number >= 7:
        config.vehicle_types.append('sports_car')

    # River configuration (starts at level 6)
    if level_number >= 6:
        config.has_river = True
        config.river_lanes = min(1 + (level_number - 6) // 2, 3)

        river_level_factor = level_number - 6
        config.river_min_speed = min(
            difficulty.river_speed_base + (river_level_factor * difficulty.river_speed_increase),
            difficulty.max_river_speed - 0.5
        )
        config.river_max_speed = min(
            difficulty.river_speed_base + 0.5 + (river_level_factor * difficulty.river_speed_increase),
            difficulty.max_river_speed
        )

        # Logs per lane (starts high and gradually decreases)
        base_logs = 4
        level_reduction = river_level_factor * 0.25  # Reduced from 0.5
        config.logs_per_lane = max(2, int(base_logs - level_reduction))

        if level_number >= 8:
            config.include_lilypads = True

        if level_number >= 10:
            config.log_sizes.append('large')
        if level_number >= 12:
            config.log_sizes.append('small')

        config.river_spawn_rate = max(
            difficulty.river_spawn_rate_base - (river_level_factor * difficulty.river_spawn_rate_decrease),
            difficulty.min_river_spawn_rate
        )

    return config