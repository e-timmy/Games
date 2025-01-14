class GameState:
    def __init__(self):
        self.is_playing = False

    def toggle_play(self):
        self.is_playing = not self.is_playing