import numpy as np
import math
from typing import Tuple
from ray import Ray


class Surface:

    @staticmethod
    def valid_t(t):
        return t > 1 ** -40 and not isinstance(t, complex)

    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self._validate()

    def reflected_and_refracted(self, ray: Ray) -> Tuple[Ray, Ray] or None:
        position = self.intersection_position(ray)
        if position is None:
            return None
        reflected = Ray(ray.dim, position, self._reflected_direction(ray, position))
        refracted = Ray(ray.dim, position, self._refracted_direction(ray, position))
        return reflected, refracted

    def reflected(self, ray: Ray) -> Ray or None:
        position = self.intersection_position(ray)
        if position is None:
            return None
        direction = self._reflected_direction(ray, position)
        return Ray(ray.dim, position, direction)

    def refracted(self, ray: Ray) -> Ray or None:
        position = self.intersection_position(ray)
        if position is None:
            return None
        direction = self._refracted_direction(ray, position)
        return Ray(ray.dim, position, direction)

    def _reflected_direction(self, ray: Ray, position: np.ndarray) -> np.ndarray:
        normal = self.normal(position)
        return ray.direction - 2 * np.dot(ray.direction, normal) * normal

    def _refracted_direction(self, ray: Ray, position: np.ndarray) -> np.ndarray:
        normal = self.normal(position)
        n1 = self.n1
        n2 = self.n2
        scalar_prod = np.dot(ray.direction, normal)
        if scalar_prod < 0:
            n1, n2 = n2, n1
            normal = -normal
            scalar_prod = -scalar_prod
        radicand = ((n2 ** 2 - n1 ** 2) / (scalar_prod ** 2 * n1 ** 2)) + 1
        return n1 * ray.direction - scalar_prod * normal * n1 * (1 - math.sqrt(radicand))

    def intersection_position(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError("Surface can't calculate intersection position")

    def normal(self, position: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Surface doesn't have a normal vector")

    def _validate(self):
        raise NotImplementedError("Surface doesn't do validation")
