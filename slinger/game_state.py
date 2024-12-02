class GameState:
    def __init__(self):
        self.game_over = False
        self.current_level = 1
        self.max_levels = 3
        self.level_complete = False
        self.bullets_per_level = 1

    def set_game_over(self):
        self.game_over = True

    def is_game_over(self):
        return self.game_over

    def next_level(self):
        if self.current_level < self.max_levels:
            self.current_level += 1
            self.level_complete = False
            self.bullets_per_level += 1
        else:
            self.set_game_over()

    def set_level_complete(self):
        self.level_complete = True

    def reset(self):
        self.game_over = False
        self.current_level = 1
        self.level_complete = False
        self.bullets_per_level = 1