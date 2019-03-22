import numpy as np
import copy
from ray import Ray
from surface import Surface


class Plane(Surface):

    def __init__(self, n: int, n1: float, n2: float, pos_vector: np.ndarray, normal_vector: np.ndarray):
        self.n = n
        self.pos_vector = pos_vector
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        super().__init__(n1, n2)

    def _normal(self):
        return copy.deepcopy(self.normal_vector)

    def _intersection_position(self, ray: Ray) -> np.ndarray:
        t = np.dot(self.normal_vector, self.pos_vector - ray.position) / \
            np.dot(self.normal_vector, ray.direction)
        if t < 1 ** -20:
            raise ValueError("No intersection")
        return ray.point(t)

    def _validate(self):
        if len(self.pos_vector.shape) != 1 or len(self.pos_vector) != self.n:
            raise ValueError("Direction vector must have shape (1, n)")
        if len(self.normal_vector.shape) != 1 or len(self.normal_vector) != self.n:
            raise ValueError("Normal vector must have shape (1, n)")
