import numpy as np
from ray import Ray
from surface import Surface


class Ellipse(Surface):

    def __init__(self,
                 dim: int,
                 n1: float,
                 n2: float,
                 center: np.ndarray,
                 semi_axes: np.ndarray):
        self.dim = dim
        self.center = center
        self.semi_axes = semi_axes
        super().__init__(n1, n2)
        self.m = self._transform_matrix()

    def _transform_matrix(self) -> np.ndarray:
        m = np.array([[0] * self.dim] * self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    m[i, j] = self.semi_axes.prod() / self.semi_axes[i]
        return m

    def intersection_position(self, ray: Ray) -> np.ndarray or None:
        md = self.m.dot(ray.direction)
        mpc = self.m.dot(ray.position - self.center)
        a = md.dot(md)
        b = 2*md.dot(mpc)
        c = mpc.dot(mpc) - self.semi_axes.prod()**2
        t1, t2 = np.roots([a, b, c])
        min_t = min(t1, t2)
        t = min_t if min_t >= 10**-10 else max(t1, t2)
        return ray.point(t) if super().valid_t(t) else None

    def normal(self, position: np.ndarray) -> np.ndarray:
        normal = 2 * (position - self.center) / np.power(self.semi_axes, 2)
        return normal / np.linalg.norm(normal)

    def _validate(self):
        if self.dim < 2:
            raise ValueError("Dimension must be at least 2")
        if len(self.center.shape) != 1 or len(self.center) != self.dim:
            raise ValueError("Center vector must have shape (1, n)")
        if len(self.semi_axes.shape) != 1 or len(self.semi_axes) != self.dim:
            raise ValueError("Semi-axes must have shape (1, n)")
        for i in range(self.dim):
            if self.semi_axes[i] <= 0:
                raise ValueError("Semi-axis must be positive")
