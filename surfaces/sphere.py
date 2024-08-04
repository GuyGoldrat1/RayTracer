import numpy as np


class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index -1

    def intersect(self, ray_origin, ray_direction):
        L = ray_origin - self.position
        b = 2.0 * np.dot(ray_direction, L)
        c = np.dot(L, L) - self.radius ** 2
        delta = b ** 2 - 4 * c
        t = 0
        if delta > 0:
            sqrt_delta = np.sqrt(delta)
            t1 = (-b + sqrt_delta) / 2
            t2 = (-b - sqrt_delta) / 2
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
            elif t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                return None, None, None  # Both t1 and t2 are negative

            intersection_point = ray_origin + t * np.array(ray_direction)
            normal_vector = (intersection_point - self.position) / self.radius
            return t, intersection_point, normal_vector
        return None, None, None
