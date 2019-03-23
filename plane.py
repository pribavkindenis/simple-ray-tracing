import numpy as np
import copy
from ray import Ray
from surface import Surface


class Plane(Surface):

    def __init__(self, n: int, n1: float, n2: float, position: np.ndarray, normal: np.ndarray):
        self.n = n
        self.position = position
        self.normal = normal / np.linalg.norm(normal)
        super().__init__(n1, n2)

    def _normal(self, ray: Ray):
        return copy.deepcopy(self.normal)

    def _intersection_position(self, ray: Ray) -> np.ndarray:
        t = np.dot(self.normal, self.position - ray.position) / \
            np.dot(self.normal, ray.direction)
        if t < 1 ** -30 or isinstance(t, complex):
            raise ValueError("No intersection")
        return ray.point(t)

    def _validate(self):
        if self.n < 2:
            raise ValueError("Dimension must be at least 2")
        if len(self.position.shape) != 1 or len(self.position) != self.n:
            raise ValueError("Position vector must have shape (1, n)")
        if len(self.normal.shape) != 1 or len(self.normal) != self.n:
            raise ValueError("Normal vector must have shape (1, n)")
