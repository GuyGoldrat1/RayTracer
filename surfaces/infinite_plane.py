import numpy as np

class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index -1

    def intersect(self, ray):
        denom = np.dot(ray.direction, self.normal)
        if np.abs(denom) > 1e-6:  
            t = (self.offset - np.dot(ray.position, self.normal)) / denom
            if t >= 0:  
                intersection_point = ray.position + t * np.array(ray.direction)
                normal_vector = np.array(self.normal)
                return t, intersection_point, normal_vector
        return None, None, None
