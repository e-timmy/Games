class GameState:
    def __init__(self, left_paddle, right_paddle, ball):
        self.left_paddle = left_paddle
        self.right_paddle = right_paddle
        self.ball = ball
        self.left_score = 0
        self.right_score = 0
        self.games_played = 0

    def get_state(self):
        return [
            self.left_paddle.rect.centery / 600,
            self.ball.rect.centerx / 800,
            self.ball.rect.centery / 600,
            self.ball.dx / 10,
            self.ball.dy / 10
        ]

    def update(self):
        self.ball.move()
        ball_hit = self.ball.check_collision(self.left_paddle, self.right_paddle) == 'left'

        game_over = False
        won = False

        if self.ball.rect.right >= 800:
            self.left_score += 1
            self.games_played += 1
            game_over = True
            won = True
        elif self.ball.rect.left <= 0:
            self.right_score += 1
            self.games_played += 1
            game_over = True
            won = False

        return ball_hit, game_over, won

    def reset(self):
        self.ball.reset()