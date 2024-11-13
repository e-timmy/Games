class LevelConfig:
    def __init__(self):
        self.num_lanes = 3
        self.spawn_rate = 120  # frames between spawns
        self.min_speed = 2
        self.max_speed = 3
        self.vehicle_types = ['car']


def generate_level(level_number):
    config = LevelConfig()

    # Increase number of lanes (max 7)
    config.num_lanes = min(3 + (level_number - 1) // 2, 7)

    # Increase speed range
    config.min_speed = min(2 + (level_number - 1) * 0.5, 5)
    config.max_speed = min(3 + (level_number - 1) * 0.5, 6)

    # Decrease spawn rate (make more frequent)
    config.spawn_rate = max(120 - (level_number - 1) * 10, 60)

    # Add vehicle types
    if level_number >= 3:
        config.vehicle_types.append('truck')
    if level_number >= 5:
        config.vehicle_types.append('sports_car')

    return config