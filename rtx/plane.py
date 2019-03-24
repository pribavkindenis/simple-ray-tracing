import numpy as np
import copy
from rtx.ray import Ray
from rtx.surface import Surface
from typing import Optional


class Plane(Surface):

    def __init__(self,
                 dim: int,
                 n1: float,
                 n2: float,
                 position: np.ndarray,
                 n: np.ndarray):
        self.dim = dim
        self.position = position
        self.n = n / np.linalg.norm(n)
        super().__init__(n1, n2)

    def normal(self, position: np.ndarray) -> np.ndarray:
        return copy.deepcopy(self.n)

    def intersection_position(self, ray: Ray) -> Optional[np.ndarray]:
        t = np.dot(self.n, self.position - ray.position) / \
            np.dot(self.n, ray.direction)
        return ray.point(t) if super().valid_t(t) else None

    def _validate(self):
        if self.dim < 2:
            raise ValueError("Dimension must be at least 2")
        if len(self.position.shape) != 1 or len(self.position) != self.dim:
            raise ValueError("Position vector must have shape (1, n)")
        if len(self.n.shape) != 1 or len(self.n) != self.dim:
            raise ValueError("Normal vector must have shape (1, n)")
