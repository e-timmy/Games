# difficulty.py
from dataclasses import dataclass

@dataclass
class DifficultyLevel:
    name: str
    grid_size: int
    mines: int
    detector_uses: int
    color: tuple
    best_time: float = float('inf')

class Difficulties:
    EASY = DifficultyLevel("Easy", 10, 10, 5, (50, 205, 50))      # Light green
    MEDIUM = DifficultyLevel("Medium", 16, 40, 3, (255, 215, 0))  # Gold
    HARD = DifficultyLevel("Hard", 24, 99, 1, (220, 20, 60))      # Crimson

    @staticmethod
    def get_all():
        return [Difficulties.EASY, Difficulties.MEDIUM, Difficulties.HARD]