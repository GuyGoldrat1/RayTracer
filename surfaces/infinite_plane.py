import numpy as np

class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index -1

    def intersect(self, ray_origin, ray_direction):
        denom = np.dot(ray_direction, self.normal)
        if np.abs(denom) > 1e-6:  # Check if the ray is not parallel to the plane
            t = (self.offset - np.dot(ray_origin, self.normal)) / denom
            if t >= 0:  # Check if the intersection is in front of the ray origin
                intersection_point = ray_origin + t * np.array(ray_direction)
                normal_vector = np.array(self.normal)
                return t, intersection_point, normal_vector
        return None, None, None
