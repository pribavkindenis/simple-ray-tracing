import numpy as np
from ray import Ray
from surface import Surface


class Plane(Surface):
    def __init__(self,
                 n: int,
                 pos_vector: np.ndarray,
                 normal_vector: np.ndarray,
                 first_refractive_index: float,
                 second_refractive_index: float):
        self.n = n
        self.pos_vector = pos_vector
        self.normal_vector = normal_vector
        self.first_refractive_index = first_refractive_index
        self.second_refractive_index = second_refractive_index
        self.__validate_init_data()
        self.__normalize_normal_vector()

    def get_normal_vector(self):
        return self.normal_vector

    def get_first_refractive_index(self):
        return self.first_refractive_index

    def get_second_refractive_index(self):
        return self.second_refractive_index

    def intersection_position(self, ray: Ray) -> np.ndarray:
        t = np.dot(self.normal_vector, self.pos_vector - ray.pos_vector) / \
               np.dot(self.normal_vector, ray.dir_vector)
        if t < 1**-20:
            raise ValueError("No intersection")
        return ray.get_point(t)

    def __validate_init_data(self):
        if len(self.pos_vector.shape) != 1 or len(self.pos_vector) != self.n:
            raise ValueError("Direction vector must have shape (1, n)")
        if len(self.normal_vector.shape) != 1 or len(self.normal_vector) != self.n:
            raise ValueError("Normal vector must have shape (1, n)")

    def __normalize_normal_vector(self):
        self.normal_vector = self.normal_vector / np.linalg.norm(self.normal_vector)
