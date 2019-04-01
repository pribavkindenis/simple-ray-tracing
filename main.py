import numpy as np
from rtx.plane import Plane
from rtx.ray import Ray
from rtx.sphere import Sphere
from rtx.ellipse import Ellipse
from rtx.util.util import transform3d_z
import rtx.render.two_dim_render as two_dim_render
import rtx.render.three_dim_render as three_dim_render


def show_two_dim_render_demo():
    ray = Ray(2, np.array([3, 6]), np.array([-1, -7]))
    plane = Plane(2, 1.4, 1, np.array([-6, -1]), np.array([-2, -2]))
    two_dim_render.plane_render(ray, plane)
    r1 = Ray(2, np.array([-2, 2]), np.array([1, 1]))
    r2 = Ray(2, np.array([7, 5]), np.array([1, 2]))

    sphere = Sphere(2, 1, 1.8, np.array([5, 6]), 4)
    two_dim_render.sphere_render(r1, sphere, 100)
    two_dim_render.sphere_render(r2, sphere, 100)

    ellipse = Ellipse(2, 1.2, 1, np.array([5, 6]), np.array([60, 30]))
    two_dim_render.ellipse_render(r1, ellipse, 300)
    two_dim_render.ellipse_render(r2, ellipse, 300)


def show_three_dim_render_demo():
    ray = Ray(3, np.array([4, 1, -10]), np.array([0, 1, 1]))
    plane = Plane(3, 1, 2, np.array([5, 5, 10]), np.array([0, 0, 1]))
    three_dim_render.plane_render(ray, plane)

    ray_point = np.array([5, 5, 100])
    ray_direction = np.array([30, 30, -100])

    rays = [Ray(3, ray_point, transform3d_z(i).dot(ray_direction)) for i in np.linspace(0, 2 * np.pi, 100)]
    three_dim_render.multiple_rays_plane_render(rays, plane)


if __name__ == "__main__":
    # show_two_dim_render_demo()
    show_three_dim_render_demo()
