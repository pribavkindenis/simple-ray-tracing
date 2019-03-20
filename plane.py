import numpy as np
from ray import Ray


class Plane:

    def __init__(self,
                 n: int,
                 pos_vector: np.ndarray,
                 normal_vector: np.ndarray):
        self.n = n
        self.pos_vector = pos_vector
        self.normal_vector = normal_vector
        self.__validate_init_data()
        self.__normalize_normal_vector()

    def get_reflected_ray(self, ray: Ray) -> Ray:
        pos_vector = self.intersect_with(ray)
        dir_vector = self.get_reflected_ray_direction(ray)
        return Ray(ray.n, pos_vector, dir_vector)

    def intersect_with(self, ray: Ray) -> np.ndarray:
        t = np.dot(self.normal_vector, self.pos_vector - ray.pos_vector) / \
               np.dot(self.normal_vector, ray.dir_vector)
        if t < 1**-20:
            raise ValueError("No intersection")
        return ray.get_point(t)

    def get_reflected_ray_direction(self, ray: Ray):
        return ray.dir_vector - 2 * np.dot(ray.dir_vector, self.normal_vector) * self.normal_vector

    def __normalize_normal_vector(self):
        self.normal_vector = self.normal_vector / np.linalg.norm(self.normal_vector)

    def __validate_init_data(self):
        if len(self.pos_vector.shape) != 1 or len(self.pos_vector) != self.n:
            raise ValueError("Direction vector must have shape (1, n)")
        if len(self.normal_vector.shape) != 1 or len(self.normal_vector) != self.n:
            raise ValueError("Normal vector must have shape (1, n)")
