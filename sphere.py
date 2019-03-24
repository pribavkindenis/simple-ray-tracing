import numpy as np
from ray import Ray
from surface import Surface


class Sphere(Surface):
    def __init__(self,
                 dim: int,
                 n1: float,
                 n2: float,
                 center: np.ndarray,
                 radius: float):
        self.dim = dim
        self.center = center
        self.radius = radius
        super().__init__(n1, n2)

    def intersection_position(self, ray: Ray) -> np.ndarray or None:
        pc = ray.position - self.center
        a = 1
        b = 2*np.dot(pc, ray.direction)
        c = np.dot(pc, pc) - self.radius**2
        t1, t2 = np.roots([a, b, c])
        min_t = min(t1, t2)
        t = min_t if min_t >= 10**-10 else max(t1, t2)
        return ray.point(t) if super().valid_t(t) else None

    def normal(self, position: np.ndarray) -> np.ndarray:
        normal = position - self.center
        return normal / np.linalg.norm(normal)

    def _validate(self):
        if self.dim < 2:
            raise ValueError("Dimension must be at least 2")
        if len(self.center.shape) != 1 or len(self.center) != self.dim:
            raise ValueError("Center vector must have shape (1, n)")
        if self.radius <= 0:
            raise ValueError("Radius must be positive")
