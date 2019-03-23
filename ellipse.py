import numpy as np
from ray import Ray
from surface import Surface


class Ellipse(Surface):

    def __init__(self, n: int, n1: float, n2: float, center: np.ndarray, semi_axes: np.ndarray):
        self.n = n
        self.center = center
        self.semi_axes = semi_axes
        super().__init__(n1, n2)
        self.m = self._transform_matrix()

    def _transform_matrix(self) -> np.ndarray:
        m = np.array([[0] * self.n] * self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    m[i, j] = self.semi_axes.prod() / self.semi_axes[i]
        return m

    def _intersection_position(self, ray: Ray) -> np.ndarray:
        md = self.m.dot(ray.direction)
        mpc = self.m.dot(ray.position - self.center)
        a = md.dot(md)
        b = 2*md.dot(mpc)
        c = mpc.dot(mpc) - self.semi_axes.prod()**2
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
        if len(self.semi_axes.shape) != 1 or len(self.semi_axes) != self.n:
            raise ValueError("Semi-axes must have shape (1, n)")
        for i in range(self.n):
            if self.semi_axes[i] <= 0:
                raise ValueError("Semi-axis must be positive")
