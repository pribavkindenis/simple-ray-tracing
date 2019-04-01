import numpy as np
from rtx.plane import Plane
from rtx.ray import Ray
from typing import Tuple
from rtx.sphere import Sphere
from rtx.ellipse import Ellipse
from rtx.surface import Surface
from matplotlib import pyplot as plt
from matplotlib import patches


surface_color = "black"
reflected_color = "orange"
refracted_color = "red"
ray_color = "blue"


def plane_render(ray: Ray, plane: Plane):
    fig, ax = plt.subplots()
    reflected = plane.reflected(ray)
    refracted = plane.refracted(ray)
    distance = 5
    plane_position = plane.position

    if reflected is not None:
        distance: float = np.linalg.norm(ray.position - reflected.position)
        plane_position = reflected.position
        ax.plot(*_ray_coords(reflected), reflected_color)

    if refracted is not None:
        ax.plot(*_ray_coords(refracted), refracted_color)

    ax.plot(*_ray_coords(ray, distance), ray_color)
    ax.plot(*_plane_coords(plane, plane_position), surface_color)
    ax.plot(*_normal_coords(plane, plane_position), surface_color, linestyle="--")
    ax.axis('equal')
    ax.grid()
    fig.show()


def sphere_render(ray: Ray, sphere: Sphere, max_level: int = 10):
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle(sphere.center, sphere.radius, fill=False, color="black"))
    _recursive_render(ax, ray, sphere, "blue", 0, max_level)
    ax.autoscale()
    plt.axis('equal')
    plt.grid()
    fig.show()


def ellipse_render(ray: Ray, ellipse: Ellipse, max_level: int = 10):
    fig, ax = plt.subplots()
    width = ellipse.semi_axes[0] * 2
    height = ellipse.semi_axes[1] * 2
    ax.add_patch(patches.Ellipse(ellipse.center, width, height, fill=False, color="black"))
    _recursive_render(ax, ray, ellipse, "blue", 0, max_level)
    ax.autoscale()
    plt.axis('equal')
    plt.grid()
    fig.show()


def _recursive_render(ax: plt.Axes, ray: Ray, surface: Surface, color: str, level: int, max_level: int):
    reflected = surface.reflected(ray)
    refracted = surface.refracted(ray)
    distance = 5

    if reflected is not None:
        distance: float = np.linalg.norm(ray.position - reflected.position)
        if level < max_level:
            _recursive_render(ax, reflected, surface, reflected_color, level + 1, max_level)

    if refracted is not None and level < max_level:
        _recursive_render(ax, refracted, surface, refracted_color, level + 1, max_level)

    ax.plot(*_ray_coords(ray, distance), color)


def _plane_coords(plane: Plane, position: np.ndarray, length: int = 20) -> Tuple[list, list]:
    e = _rotate(plane.n)
    a = position + e * length
    b = position - e * length
    return [a[0], b[0]], [a[1], b[1]]


def _normal_coords(plane: Plane, position: np.ndarray, length: int = 20) -> Tuple[list, list]:
    e = plane.n
    a = position + e * length
    b = position - e * length
    return [a[0], b[0]], [a[1], b[1]]


def _rotate(v: np.array) -> np.ndarray:
    temp = [v[1], -v[0]]
    return np.array(temp)


def _ray_coords(ray: Ray, length: float = 5) -> Tuple[list, list]:
    end_point = ray.point(length)
    return [ray.position[0], end_point[0]], [ray.position[1], end_point[1]]
