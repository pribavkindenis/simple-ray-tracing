import numpy as np
from rtx.plane import Plane
from rtx.ray import Ray
from rtx.sphere import Sphere
from rtx.ellipse import Ellipse
from rtx.render.two_dim_render import plane_render, sphere_render, ellipse_render


if __name__ == "__main__":
    ray = Ray(2, np.array([3, 6]), np.array([-1, -7]))
    plane = Plane(2, 1.4, 1, np.array([-6, -1]), np.array([-2, -2]))
    plane_render(ray, plane)
    r1 = Ray(2, np.array([-2, 2]), np.array([1, 1]))
    r2 = Ray(2, np.array([7, 5]), np.array([1, 2]))

    sphere = Sphere(2, 1, 1.8, np.array([5, 6]), 4)
    sphere_render(r1, sphere, 100)
    sphere_render(r2, sphere, 100)

    ellipse = Ellipse(2, 1.2, 1, np.array([5, 6]), np.array([60, 30]))
    ellipse_render(r1, ellipse, 300)
    ellipse_render(r2, ellipse, 300)
