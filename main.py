import numpy as np
from plane import Plane
from ray import Ray

if __name__ == "__main__":
    ray = Ray(2, np.array([3, 6]), np.array([-1, -7]))
    plane = Plane(2, 1, 1.5, np.array([-6, -1]), np.array([-2, -2]))

    print(plane.reflected(ray).direction)
    print(plane.refracted(ray).direction)


