from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, paddle):
        self.paddle = paddle

    @abstractmethod
    def act(self, game_state):
        pass

    @abstractmethod
    def update(self, game_state, action, next_state, reward, done):
        pass