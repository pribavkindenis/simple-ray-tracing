import numpy as np
from ray import Ray
from surface import Surface


class Sphere(Surface):
    def __init__(self, n: int, n1: float, n2: float, center: np.ndarray, radius: float):
        self.n = n
        self.center = center
        self.radius = radius
        super().__init__(n1, n2)

    def _intersection_position(self, ray: Ray) -> np.ndarray:
        pc = ray.position - self.center
        a = 1
        b = 2*np.dot(pc, ray.direction)
        c = np.dot(pc, pc) - self.radius**2
        t1, t2 = np.roots([a, b, c])
        min_t = min(t1, t2)
        t = min_t if min_t >= 0 else max(t1, t2)
        if t < 1 ** -30 or isinstance(t, complex):
            raise ValueError("No intersection")
        return ray.point(t)

    def _normal(self, ray: Ray) -> np.ndarray:
        temp = self._intersection_position(ray) - self.center
        return temp / np.linalg.norm(temp)

    def _validate(self):
        if self.n < 2:
            raise ValueError("Dimension must be at least 2")
        if len(self.center.shape) != 1 or len(self.center) != self.n:
            raise ValueError("Center vector must have shape (1, n)")
        if self.radius <= 0:
            raise ValueError("Radius must be positive")
