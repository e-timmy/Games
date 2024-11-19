class LevelConfig:
    def __init__(self):
        # Road settings
        self.num_lanes = 3
        self.spawn_rate = 120  # frames between spawns
        self.min_speed = 2
        self.max_speed = 3
        self.vehicle_types = ['car']

        # River settings (active after level 5)
        self.has_river = False
        self.river_lanes = 0
        self.river_min_speed = 1
        self.river_max_speed = 2
        self.logs_per_lane = 0
        self.log_sizes = ['medium']
        self.river_spawn_rate = 180  # frames between log spawns


def generate_level(level_number):
    config = LevelConfig()

    # Road configuration
    config.num_lanes = min(3 + (level_number - 1) // 2, 7)
    config.min_speed = min(2 + (level_number - 1) * 0.5, 5)
    config.max_speed = min(3 + (level_number - 1) * 0.5, 6)
    config.spawn_rate = max(120 - (level_number - 1) * 10, 60)

    if level_number >= 3:
        config.vehicle_types.append('truck')
    if level_number >= 5:
        config.vehicle_types.append('sports_car')

    # River configuration (starts at level 6)
    if level_number >= 6:
        config.has_river = True
        config.river_lanes = min(1 + (level_number - 6) // 2, 3)  # Add a new lane every 2 levels
        config.river_min_speed = min(1 + (level_number - 6) * 0.3, 3)
        config.river_max_speed = min(2 + (level_number - 6) * 0.3, 4)

        # Logs per lane decreases as levels progress (making it harder)
        base_logs = 4
        level_reduction = (level_number - 6) * 0.5
        config.logs_per_lane = max(2, int(base_logs - level_reduction))

        # Add different log sizes as the game progresses
        if level_number >= 8:
            config.log_sizes.append('large')
        if level_number >= 10:
            config.log_sizes.append('small')

        # Spawn rate increases (lower number) with level
        config.river_spawn_rate = max(180 - (level_number - 6) * 15, 90)

    return config