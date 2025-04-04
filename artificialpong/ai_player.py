import random
from collections import deque


class AIPlayer:
    def __init__(self, paddle, difficulty):
        self.paddle = paddle
        self.difficulty = difficulty
        self.set_difficulty(difficulty)
        self.move_history = deque(maxlen=5)
        self.move_threshold = 0.6

    def set_difficulty(self, difficulty):
        if difficulty == "Easy":
            self.prediction_noise = 40
            self.paddle.set_speed(3)
        elif difficulty == "Medium":
            self.prediction_noise = 20
            self.paddle.set_speed(5)
        else:  # Hard
            self.prediction_noise = 5
            self.paddle.set_speed(7)

    def move(self, ball):
        noisy_ball_y = ball.rect.centery + random.uniform(-self.prediction_noise, self.prediction_noise)

        time_to_reach = (self.paddle.rect.centerx - ball.rect.centerx) / ball.dx if ball.dx != 0 else 0
        predicted_y = noisy_ball_y + ball.dy * time_to_reach

        if self.paddle.rect.centery < predicted_y:
            raw_move = 1
        elif self.paddle.rect.centery > predicted_y:
            raw_move = -1
        else:
            raw_move = 0

        self.move_history.append(raw_move)

        avg_move = sum(self.move_history) / len(self.move_history)

        if avg_move > self.move_threshold:
            return False, True
        elif avg_move < -self.move_threshold:
            return True, False
        else:
            return False, False

    def reset_position(self):
        target_y = 300
        if self.paddle.rect.centery < target_y - 10:
            return False, True
        elif self.paddle.rect.centery > target_y + 10:
            return True, False
        else:
            return False, False