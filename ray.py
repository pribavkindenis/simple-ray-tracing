import numpy as np


class Ray:

    def __init__(self, n: int, position: np.ndarray, direction: np.ndarray):
        self.n = n
        self.position = position
        self.direction = direction / np.linalg.norm(direction)
        self._validate()

    def point(self, t: float) -> np.ndarray:
        return self.position + self.direction * t

    def _validate(self):
        if len(self.position.shape) != 1 or len(self.position) != self.n:
            raise ValueError("Position vector must have shape (1, n)")
        if len(self.direction.shape) != 1 or len(self.direction) != self.n:
            raise ValueError("Direction vector must have shape (1, n)")
