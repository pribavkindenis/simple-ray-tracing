import numpy as np
from plane import Plane
from ray import Ray
from sphere import Sphere
from ellipse import Ellipse

if __name__ == "__main__":
    ray = Ray(2, np.array([3, 6]), np.array([-1, -7]))
    plane = Plane(2, 1, 1.5, np.array([-6, -1]), np.array([-2, -2]))

#    print(plane.reflected(ray).direction)
#    print(plane.refracted(ray).direction)

    r1 = Ray(2, np.array([-5, 9]), np.array([3, -1]))
    r2 = Ray(2, np.array([7, 5]), np.array([0, 1]))
    sphere = Sphere(2, 1, 1.5, np.array([5, 6]), 4)

    print(sphere.reflected(r1).position)
    print(sphere.reflected(r2).position)

    ellipse = Ellipse(2, 1, 1.5, np.array([5, 6]), np.array([4, 4]))
    print(ellipse.reflected(r1).position)
    print(ellipse.reflected(r2).position)
