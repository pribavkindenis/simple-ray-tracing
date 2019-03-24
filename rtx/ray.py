import numpy as np


class Ray:

    def __init__(self,
                 dim: int,
                 position: np.ndarray,
                 direction: np.ndarray):
        self.dim = dim
        self.position = position
        self.direction = direction / np.linalg.norm(direction)
        self._validate()

    def point(self, t: float) -> np.ndarray:
        return self.position + self.direction * t

    def _validate(self):
        if len(self.position.shape) != 1 or len(self.position) != self.dim:
            raise ValueError("Position vector must have shape (1, n)")
        if len(self.direction.shape) != 1 or len(self.direction) != self.dim:
            raise ValueError("Direction vector must have shape (1, n)")
