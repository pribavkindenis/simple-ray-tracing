import numpy as np
import math
from typing import Callable
from ray import Ray


class Surface:

    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self._validate()

    def reflected(self, ray: Ray) -> Ray:
        return self._create_ray(ray, self._reflected_direction)

    def refracted(self, ray: Ray) -> Ray:
        return self._create_ray(ray, self._refracted_direction)

    def _create_ray(self, ray: Ray, direction: Callable[[Ray], np.ndarray]) -> Ray:
        position = self._intersection_position(ray)
        return Ray(ray.n, position, direction(ray))

    def _reflected_direction(self, ray: Ray) -> np.ndarray:
        normal = self._normal(ray)
        return ray.direction - 2 * np.dot(ray.direction, normal) * normal

    def _refracted_direction(self, ray: Ray) -> np.ndarray:
        normal = self._normal(ray)
        n1 = self.n1
        n2 = self.n2
        scalar_prod = np.dot(ray.direction, normal)
        if scalar_prod < 0:
            n1, n2 = n2, n1
            normal = -normal
            scalar_prod = -scalar_prod
        radicand = ((n2 ** 2 - n1 ** 2) / (scalar_prod ** 2 * n1 ** 2)) + 1
        return n1 * ray.direction - scalar_prod * normal * n1 * (1 - math.sqrt(radicand))

    def _intersection_position(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError("Surface can't calculate intersection position")

    def _normal(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError("Surface doesn't have a normal vector")

    def _validate(self):
        raise NotImplementedError("Surface doesn't do validation")
