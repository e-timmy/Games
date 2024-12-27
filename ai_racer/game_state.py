class GameState:
    def __init__(self, screen_width, screen_height, track):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.track = track
        self.race_started = False
        self.countdown_state = 0
        self.countdown_timer = 0
        self.countdown_duration = 60  # frames per light
        self.max_countdown_state = 4  # 3 reds + 1 green