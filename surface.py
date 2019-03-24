import numpy as np
import math
from typing import Optional
from ray import Ray


class Surface:

    @staticmethod
    def valid_t(t):
        return t > 1 ** -40 and not isinstance(t, complex)

    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self._validate()

    def reflected(self, ray: Ray) -> Optional[Ray]:
        position = self.intersection_position(ray)
        if position is None:
            return None
        direction = self._reflected_direction(ray, position)
        return Ray(ray.dim, position, direction)

    def refracted(self, ray: Ray) -> Optional[Ray]:
        position = self.intersection_position(ray)
        if position is None:
            return None
        direction = self._refracted_direction(ray, position)
        if direction is None:
            return None
        return Ray(ray.dim, position, direction)

    def _reflected_direction(self, ray: Ray, position: np.ndarray) -> np.ndarray:
        normal = self.normal(position)
        return ray.direction - 2 * np.dot(ray.direction, normal) * normal

    def _refracted_direction(self, ray: Ray, position: np.ndarray) -> Optional[np.ndarray]:
        normal = self.normal(position)
        n1 = self.n1
        n2 = self.n2
        scalar_prod = np.dot(ray.direction, normal)
        if scalar_prod < 0:
            n1, n2 = n2, n1
            normal = -normal
            scalar_prod = np.dot(ray.direction, normal)
        radicand = 1 - (n1 / n2) ** 2 * (1 - scalar_prod ** 2)
        if radicand <= 0:
            return None
        return (n1 * ray.direction - normal * (n1*abs(scalar_prod) - n2*math.sqrt(radicand))) / n2

    def intersection_position(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError("Surface can't calculate intersection position")

    def normal(self, position: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Surface doesn't have a normal vector")

    def _validate(self):
        raise NotImplementedError("Surface doesn't do validation")
