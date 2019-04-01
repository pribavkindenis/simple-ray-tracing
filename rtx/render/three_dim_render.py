import numpy as np
from rtx.plane import Plane
from rtx.ray import Ray
from typing import *
from rtx.sphere import Sphere
from rtx.ellipse import Ellipse
from rtx.surface import Surface
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
import copy


fig_size = (15, 10)

ray_length = 50
plane_length = 30
normal_length = 20

plane_color = "indigo"
reflected_color = "orange"
refracted_color = "red"
ray_color = "blue"


def plane_render(ray: Ray, plane: Plane):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    reflected = plane.reflected(ray)
    refracted = plane.refracted(ray)
    length = 5
    plane_position = plane.position

    if reflected is not None:
        length: float = np.linalg.norm(ray.position - reflected.position)
        plane_position = reflected.position
        draw_ray(ax, reflected, color=reflected_color, label="Reflected")

    if refracted is not None:
        draw_ray(ax, refracted, color=refracted_color, label="Refracted")

    pos_n, neg_n = get_normal_rays(plane.n, plane_position)
    draw_ray(ax, ray, length, color=ray_color, label="Ray")
    draw_ray(ax, pos_n, normal_length, color="black", line_style="--")
    draw_ray(ax, neg_n, normal_length, color="black", line_style="--")
    draw_plane(ax, plane, plane_position, color=plane_color, alpha=0.3, draw_borders=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.legend()
    fig.show()


def multiple_rays_plane_render(rays: List[Ray], plane: Plane, length: int = 100):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    for ray in rays:
        reflected = plane.reflected(ray)
        refracted = plane.refracted(ray)
        length = 5

        if reflected is not None:
            length: float = np.linalg.norm(ray.position - reflected.position)
            draw_ray(ax, reflected, color=reflected_color)

        if refracted is not None:
            draw_ray(ax, refracted, color=refracted_color)

        draw_ray(ax, ray, length, color=ray_color)
    draw_plane(ax, plane, plane.position, length=length, color=plane_color, alpha=0.3, draw_borders=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    legend_lines = [
        Line2D([0], [0], color=ray_color),
        Line2D([0], [0], color=reflected_color),
        Line2D([0], [0], color=refracted_color)]
    fig.legend(legend_lines, ['Ray', 'Reflected', 'Refracted'])
    fig.show()


def get_normal_rays(normal: np.ndarray, position: np.ndarray) -> Tuple[Ray, Ray]:
    return (
        Ray(3, position, normal),
        Ray(3, position, -normal),
    )


def draw_ray(ax: Axes3D,
             ray: Ray,
             length: float = ray_length,
             color: str = "blue",
             line_style: str = "-",
             label: str = None):
    end_point = ray.point(length)
    x = np.array([ray.position[0], end_point[0]])
    y = np.array([ray.position[1], end_point[1]])
    z = np.array([ray.position[2], end_point[2]])
    ax.plot(x, y, z, color=color, linestyle=line_style, label=label)


def draw_line(ax: Axes3D,
              a: np.ndarray,
              b: np.ndarray,
              color: str = "blue",
              line_style: str = "-",
              label: str = None):
    x = np.array([a[0], b[0]])
    y = np.array([a[1], b[1]])
    z = np.array([a[2], b[2]])
    ax.plot(x, y, z, color=color, linestyle=line_style, label=label)


def draw_plane(ax: Axes3D,
               plane: Plane,
               point: np.array,
               length: int = plane_length,
               elem_num: int = 100,
               color: str = "blue",
               alpha: float = 0.5,
               draw_diagonals: bool = False,
               draw_borders: bool = False):
    rays = get_orthogonal_rays(point, plane.n)
    corners = [rays[i].point(length) for i in range(4)]
    if draw_diagonals:
        for i in range(4):
            draw_ray(ax, rays[i], length, color=color, line_style="--")
    if draw_borders:
        draw_line(ax, corners[0], corners[1], color=color, line_style="-")
        draw_line(ax, corners[1], corners[2], color=color, line_style="-")
        draw_line(ax, corners[2], corners[3], color=color, line_style="-")
        draw_line(ax, corners[3], corners[0], color=color, line_style="-")
    minimum = np.minimum.reduce(corners)
    maximum = np.maximum.reduce(corners)
    x = np.linspace(minimum[0], maximum[0], elem_num)
    y = np.linspace(minimum[1], maximum[1], elem_num)
    xx, yy = np.meshgrid(x, y)
    d = -plane.position.dot(plane.n)
    zz = (-plane.n[0] * xx - plane.n[1] * yy - d) / plane.n[2]
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha)


def get_orthogonal_rays(point: np.array,
                        normal: np.array) -> Tuple[Ray, Ray, Ray, Ray]:
    first, second, third, fourth = get_orthogonal_directions(normal)
    return (
        Ray(3, point, first),
        Ray(3, point, second),
        Ray(3, point, third),
        Ray(3, point, fourth)
    )


def get_orthogonal_directions(normal: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
    first = get_orthogonal(normal, 1, 1)
    second = get_orthogonal(normal, 1, -1)
    third = get_orthogonal(normal, -1, -1)
    fourth = get_orthogonal(normal, -1, 1)
    return first, second, third, fourth


def get_orthogonal(normal: np.array,
                   px: float,
                   py: float) -> np.ndarray:
    pz = - (px * normal[0] + py * normal[1]) / normal[2]
    return np.array([px, py, pz])
