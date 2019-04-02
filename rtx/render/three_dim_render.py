import numpy as np
from rtx.plane import Plane
from rtx.ray import Ray
from typing import *
from rtx.sphere import Sphere
from rtx.ellipse import Ellipse
from rtx.surface import Surface
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


def plane_render(ray: Ray,
                 plane: Plane,
                 figsize: Tuple = (5, 5),
                 xlim: Tuple[float, float] = None,
                 ylim: Tuple[float, float] = None,
                 zlim: Tuple[float, float] = None,
                 ray_length: float = 50,
                 plane_length: float = 30,
                 normal_length: float = 20,
                 ray_color: str = "blue",
                 plane_color: str = "indigo",
                 reflected_color: str = "orange",
                 refracted_color: str = "red"):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    length = ray_length
    plane_position = plane.position

    reflected = plane.reflected(ray)
    refracted = plane.refracted(ray)

    if reflected is not None:
        length: float = np.linalg.norm(ray.position - reflected.position)
        plane_position = reflected.position
        _draw_ray(ax, reflected, ray_length, color=reflected_color)

    if refracted is not None:
        _draw_ray(ax, refracted, ray_length, color=refracted_color)

    pos_n, neg_n = _get_normal_rays(plane.n, plane_position)

    _draw_ray(ax, ray, length, color=ray_color)
    _draw_ray(ax, pos_n, normal_length, color="black", line_style="--")
    _draw_ray(ax, neg_n, normal_length, color="black", line_style="--")
    _draw_plane(ax, plane, plane_position, plane_length, color=plane_color, alpha=0.3, draw_borders=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    legend_lines = [
        Line2D([0], [0], color=ray_color),
        Line2D([0], [0], color=reflected_color),
        Line2D([0], [0], color=refracted_color)]
    fig.legend(legend_lines, ['Ray', 'Reflected', 'Refracted'])
    fig.show()


def multiple_rays_plane_render(rays: List[Ray],
                               plane: Plane,
                               figsize: Tuple = (5, 5),
                               xlim: Tuple[float, float] = None,
                               ylim: Tuple[float, float] = None,
                               zlim: Tuple[float, float] = None,
                               ray_length: float = 50,
                               plane_length: float = 100,
                               ray_color: str = "blue",
                               plane_color: str = "indigo",
                               reflected_color: str = "orange",
                               refracted_color: str = "red"):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for ray in rays:
        reflected = plane.reflected(ray)
        refracted = plane.refracted(ray)
        length = ray_length

        if reflected is not None:
            length: float = np.linalg.norm(ray.position - reflected.position)
            _draw_ray(ax, reflected, ray_length, color=reflected_color)

        if refracted is not None:
            _draw_ray(ax, refracted, ray_length, color=refracted_color)

        _draw_ray(ax, ray, length, color=ray_color)
    _draw_plane(ax, plane, plane.position, length=plane_length, color=plane_color, alpha=0.3, draw_borders=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    legend_lines = [
        Line2D([0], [0], color=ray_color),
        Line2D([0], [0], color=reflected_color),
        Line2D([0], [0], color=refracted_color)]
    fig.legend(legend_lines, ['Ray', 'Reflected', 'Refracted'])
    fig.show()


def sphere_render(ray: Ray,
                  sphere: Sphere,
                  max_level: int = 10,
                  figsize: Tuple = (5, 5),
                  xlim: Tuple[float, float] = None,
                  ylim: Tuple[float, float] = None,
                  zlim: Tuple[float, float] = None,
                  ray_length: float = 50,
                  ray_color: str = "blue",
                  sphere_color: str = "indigo",
                  reflected_color: str = "orange",
                  refracted_color: str = "red",
                  elem_num: int = 40,
                  alpha: float = 0.4):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    _draw_sphere(ax, sphere, sphere_color, elem_num, alpha)
    _recursive_render(ax, ray, sphere, ray_length, ray_color, reflected_color, refracted_color, 0, max_level)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    legend_lines = [
        Line2D([0], [0], color=ray_color),
        Line2D([0], [0], color=reflected_color),
        Line2D([0], [0], color=refracted_color)]
    fig.legend(legend_lines, ['Ray', 'Reflected', 'Refracted'])
    fig.show()


def ellipse_render(ray: Ray,
                   ellipse: Ellipse,
                   max_level: int = 10,
                   figsize: Tuple = (5, 5),
                   xlim: Tuple[float, float] = None,
                   ylim: Tuple[float, float] = None,
                   zlim: Tuple[float, float] = None,
                   ray_length: float = 50,
                   ray_color: str = "blue",
                   sphere_color: str = "indigo",
                   reflected_color: str = "orange",
                   refracted_color: str = "red",
                   elem_num: int = 40,
                   alpha: float = 0.4):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    _draw_ellipse(ax, ellipse, sphere_color, elem_num, alpha)
    _recursive_render(ax, ray, ellipse, ray_length, ray_color, reflected_color, refracted_color, 0, max_level)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    legend_lines = [
        Line2D([0], [0], color=ray_color),
        Line2D([0], [0], color=reflected_color),
        Line2D([0], [0], color=refracted_color)]
    fig.legend(legend_lines, ['Ray', 'Reflected', 'Refracted'])
    fig.show()


def _recursive_render(ax: Axes3D,
                      ray: Ray,
                      surface: Surface,
                      ray_length: float,
                      ray_color: str,
                      reflected_color: str,
                      refracted_color: str,
                      level: int,
                      max_level: int):
    reflected = surface.reflected(ray)
    refracted = surface.refracted(ray)
    length = ray_length

    if reflected is not None:
        length: float = np.linalg.norm(ray.position - reflected.position)
        if level < max_level:
            _recursive_render(ax,
                              reflected,
                              surface,
                              ray_length,
                              reflected_color,
                              reflected_color,
                              refracted_color,
                              level + 1,
                              max_level)

    if refracted is not None and level < max_level:
        _recursive_render(ax,
                          refracted,
                          surface,
                          ray_length,
                          refracted_color,
                          reflected_color,
                          refracted_color,
                          level + 1,
                          max_level)

    _draw_ray(ax, ray, length, ray_color)


def _draw_ray(ax: Axes3D,
              ray: Ray,
              length: float,
              color: str = "blue",
              line_style: str = "-",
              label: str = None):
    end_point = ray.point(length)
    x = np.array([ray.position[0], end_point[0]])
    y = np.array([ray.position[1], end_point[1]])
    z = np.array([ray.position[2], end_point[2]])
    ax.plot(x, y, z, color=color, linestyle=line_style, label=label)


def _draw_line(ax: Axes3D,
               a: np.ndarray,
               b: np.ndarray,
               color: str = "blue",
               line_style: str = "-",
               label: str = None):
    x = np.array([a[0], b[0]])
    y = np.array([a[1], b[1]])
    z = np.array([a[2], b[2]])
    ax.plot(x, y, z, color=color, linestyle=line_style, label=label)


def _draw_plane(ax: Axes3D,
                plane: Plane,
                point: np.array,
                length: float,
                elem_num: int = 100,
                color: str = "blue",
                alpha: float = 0.5,
                draw_diagonals: bool = False,
                draw_borders: bool = False):
    rays = _get_orthogonal_rays(point, plane.n)
    corners = [rays[i].point(length) for i in range(4)]
    if draw_diagonals:
        for i in range(4):
            _draw_ray(ax, rays[i], length, color=color, line_style="--")
    if draw_borders:
        _draw_line(ax, corners[0], corners[1], color=color, line_style="-")
        _draw_line(ax, corners[1], corners[2], color=color, line_style="-")
        _draw_line(ax, corners[2], corners[3], color=color, line_style="-")
        _draw_line(ax, corners[3], corners[0], color=color, line_style="-")
    minimum = np.minimum.reduce(corners)
    maximum = np.maximum.reduce(corners)
    x = np.linspace(minimum[0], maximum[0], elem_num)
    y = np.linspace(minimum[1], maximum[1], elem_num)
    xx, yy = np.meshgrid(x, y)
    d = -plane.position.dot(plane.n)
    zz = (-plane.n[0] * xx - plane.n[1] * yy - d) / plane.n[2]
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha)


def _draw_sphere(ax: Axes3D,
                 sphere: Sphere,
                 color: str,
                 elem_num: int,
                 alpha: float):
    u = np.linspace(-np.pi/2, np.pi/2, elem_num)
    v = np.linspace(0, 2 * np.pi, elem_num)
    xi, phy = np.meshgrid(v, u)
    x = sphere.center[0] + sphere.radius * np.sin(xi) * np.cos(phy)
    y = sphere.center[1] + sphere.radius * np.sin(xi) * np.sin(phy)
    z = sphere.center[2] + sphere.radius * np.cos(xi)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha)


def _draw_ellipse(ax: Axes3D,
                  ellipse: Ellipse,
                  color: str,
                  elem_num: int,
                  alpha: float):
    rx, ry, rz = ellipse.semi_axes

    u = np.linspace(0, np.pi, elem_num)
    v = np.linspace(0, 2 * np.pi, elem_num)
    xi, phy = np.meshgrid(u, v)
    x = ellipse.center[0] + rx * np.sin(xi) * np.cos(phy)
    y = ellipse.center[1] + ry * np.sin(xi) * np.sin(phy)
    z = ellipse.center[2] + rz * np.cos(xi)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha)


def _get_orthogonal_rays(point: np.array,
                         normal: np.array) -> Tuple[Ray, Ray, Ray, Ray]:
    first, second, third, fourth = _get_orthogonal_directions(normal)
    return (
        Ray(3, point, first),
        Ray(3, point, second),
        Ray(3, point, third),
        Ray(3, point, fourth)
    )


def _get_orthogonal_directions(normal: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
    first = _get_orthogonal(normal, 1, 1)
    second = _get_orthogonal(normal, 1, -1)
    third = _get_orthogonal(normal, -1, -1)
    fourth = _get_orthogonal(normal, -1, 1)
    return first, second, third, fourth


def _get_orthogonal(normal: np.array,
                    px: float,
                    py: float) -> np.ndarray:
    pz = - (px * normal[0] + py * normal[1]) / normal[2]
    return np.array([px, py, pz])


def _get_normal_rays(normal: np.ndarray, position: np.ndarray) -> Tuple[Ray, Ray]:
    return (
        Ray(3, position, normal),
        Ray(3, position, -normal),
    )