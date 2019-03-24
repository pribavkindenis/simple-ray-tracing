import numpy as np
from plane import Plane
from ray import Ray
from sphere import Sphere
from ellipse import Ellipse
from two_dim_render import plane_render, sphere_render, ellipse_render


if __name__ == "__main__":
    ray = Ray(2, np.array([3, 6]), np.array([-1, -7]))
    plane = Plane(2, 1.666, 1, np.array([-6, -1]), np.array([-2, -2]))
    plane_render(ray, plane)
    r1 = Ray(2, np.array([-2, 2]), np.array([1, 1]))
    r2 = Ray(2, np.array([7, 5]), np.array([1, 2]))

    sphere = Sphere(2, 1, 1.8, np.array([5, 6]), 4)
    sphere_render(r1, sphere, 5)
    sphere_render(r2, sphere, 5)

    ellipse = Ellipse(2, 1.2, 1, np.array([5, 6]), np.array([6, 3]))
    ellipse_render(r1, ellipse)
    ellipse_render(r2, ellipse)
