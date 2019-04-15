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
    sphere1 = Sphere(2, 1, 1.8, np.array([5, 6]), 4.3)
    two_dim_render.sphere_render(r1, sphere1, 100)

    sphere2 = Sphere(2, 1, 1.8, np.array([5, 6]), 4)
    two_dim_render.sphere_render(r2, sphere2, 100)

    ellipse1 = Ellipse(2, 1.2, 1, np.array([5, 6]), np.array([60, 30]))
    two_dim_render.ellipse_render(r1, ellipse1, 300)

    ellipse2 = Ellipse(2, 1.2, 1, np.array([15, 25]), np.array([6, 10]))
    two_dim_render.ellipse_render(r2, ellipse2, 200)


def three_d_plane():
    ray = Ray(3, np.array([4, 1, -10]), np.array([0, 1, 1]))
    plane = Plane(3, 1, 2, np.array([5, 5, 10]), np.array([0, 0, 1]))
    three_dim_render.plane_render(ray,
                                  plane,
                                  figsize=(7, 7),
                                  plane_length=30,
                                  ray_length=30,
                                  normal_length=30,
                                  xlim=(-15, 35),
                                  ylim=(-15, 35),
                                  zlim=(-10, 60))


def three_d_multiple_plane():
    ray_point = np.array([0, 0, 70])
    plane = Plane(3, 1.230, 1, np.array([0, 0, 10]), np.array([0, 0, 1]))
    ray_direction = np.array([40, 40, -100])
    rays = [Ray(3, ray_point, transform3d_z(i).dot(ray_direction)) for i in np.linspace(0, 2 * np.pi, 150)]
    three_dim_render.multiple_rays_plane_render(rays,
                                                plane,
                                                figsize=(7, 7),
                                                plane_length=70,
                                                ray_length=40,
                                                xlim=(-60, 60),
                                                ylim=(-60, 60),
                                                zlim=(-20, 100))


def three_d_sphere():
    ray = Ray(3, np.array([0, 5, 0]), np.array([-1.3, 1, 0.5]))
    sphere = Sphere(3, 1.23, 1, np.array([0, 0, 0]), 20)
    three_dim_render.sphere_render(ray,
                                   sphere,
                                   100,
                                   sphere_color="#470180",
                                   ray_color="#0245d4",
                                   reflected_color="#ffc400",
                                   refracted_color="#ff0000",
                                   ray_length=20,
                                   figsize=(7, 7),
                                   xlim=(-30, 30),
                                   ylim=(-30, 30),
                                   zlim=(-30, 30))


def three_d_ellipse():
    ray = Ray(3, np.array([0, 5, 0]), np.array([-1.3, 1, 0.5]))
    ellipse = Ellipse(3, 1.23, 1, np.array([0, 0, 0]), np.array([12, 30, 12]))
    three_dim_render.ellipse_render(ray,
                                    ellipse,
                                    100,
                                    sphere_color="#470180",
                                    ray_color="#0245d4",
                                    reflected_color="#ffc400",
                                    refracted_color="#ff0000",
                                    ray_length=20,
                                    figsize=(7, 7),
                                    xlim=(-30, 30),
                                    ylim=(-30, 30),
                                    zlim=(-30, 30))


def show_three_dim_render_demo():
    three_d_plane()
    three_d_multiple_plane()
    three_d_sphere()
    three_d_ellipse()


if __name__ == "__main__":
    show_two_dim_render_demo()
    show_three_dim_render_demo()
