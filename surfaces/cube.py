import numpy as np

class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index-1


    def intersect(self ,ray):
        half_size = self.scale / 2

        t_min = ((self.position - half_size)- ray.position) / ray.direction
        t_max = ((self.position - half_size)- ray.position) / ray.direction
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_enter = np.max(t1)
        t_exit = np.min(t2)

        if t_enter < t_exit and t_exit > 0:
            Intersection_Point= ray.position + t_enter * ray.direction
            normal_vector = np.zeros(3)
            if t_enter == t1[0]:
                normal_vector= np.array([-1, 0, 0]) if ray.direction[0] < 0 else np.array([1, 0, 0])
            elif t_enter == t1[1]:
                normal_vector= np.array([0, -1, 0]) if ray.direction[1] < 0 else np.array([0, 1, 0])
            elif t_enter == t1[2]:
                normal_vector= np.array([0, 0, -1]) if ray.direction[2] < 0 else np.array([0, 0, 1])
            return t_enter, Intersection_Point, normal_vector
        return None, None, None
