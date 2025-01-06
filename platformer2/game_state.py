class GameState:
    def __init__(self):
        self.score = 0
        self.max_distance = 0

    def update_score(self, player_x):
        distance = int(player_x)
        if distance > self.max_distance:
            self.score += distance - self.max_distance
            self.max_distance = distance

    def get_score(self):
        return self.score