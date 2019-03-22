import numpy as np
import copy
import math
from ray import Ray


class Surface:
    def reflected_ray(self, ray: Ray) -> Ray:
        pos_vector = self.intersection_position(ray)
        dir_vector = self.__reflected_ray_direction(ray)
        return Ray(ray.n, pos_vector, dir_vector)

    def refracted_ray(self, ray: Ray) -> Ray:
        pos_vector = self.intersection_position(ray)
        dir_vector = self.__refracted_ray_direction(ray)
        return Ray(ray.n, pos_vector, dir_vector)

    def __reflected_ray_direction(self, ray: Ray):
        normal_vector = self.get_normal_vector()
        return ray.dir_vector - 2 * np.dot(ray.dir_vector, normal_vector) * normal_vector

    def __refracted_ray_direction(self, ray: Ray):
        normal_vector = copy.deepcopy(self.get_normal_vector())
        n1 = self.get_first_refractive_index()
        n2 = self.get_second_refractive_index()
        scalar_prod = np.dot(ray.dir_vector, normal_vector)
        if scalar_prod < 0:
            n1, n2 = n2, n1
            normal_vector = -normal_vector
            scalar_prod = np.dot(ray.dir_vector, normal_vector)
        return n1 * ray.dir_vector - scalar_prod * normal_vector * n1 * \
               (1 - math.sqrt(((n2 ** 2 - n1 ** 2) / (scalar_prod ** 2 * n1 ** 2)) + 1))

    def intersection_position(self, ray: Ray):
        raise NotImplementedError("Surface can't calculate intersection position")

    def get_normal_vector(self):
        raise NotImplementedError("Surface doesn't have a normal vector")

    def get_first_refractive_index(self):
        raise NotImplementedError("Surface doesn't have a first refractive index")

    def get_second_refractive_index(self):
        raise NotImplementedError("Surface doesn't have a second refractive index")
