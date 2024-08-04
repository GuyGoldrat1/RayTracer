import numpy as np

class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index-1

    def intersect(self ,ray_origin, ray_direction):
        # Calculate the half-size of the box
        half_size = self.scale / 2

        # Initialize t_min and t_max to represent the interval of intersection
        t_min = ((self.position - half_size)- ray_origin) / ray_direction
        t_max = ((self.position - half_size)- ray_origin) / ray_direction

        # Ensure t_min is the entry and t_max is the exit for each slab
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        # Find the largest t1 and smallest t2
        t_enter = np.max(t1)
        t_exit = np.min(t2)

        # If the intervals overlap, there is an intersection
        # if t_enter <= t_exit:
        #     # Calculate the intersection point

        if t_enter < t_exit and t_exit > 0:
            Intersection_Point= ray_origin + t_enter * ray_direction
            normal_vector = np.zeros(3)
            # Determine the normal by checking which t_min or t_max was used
            if t_enter == t1[0]:
                normal_vector= np.array([-1, 0, 0]) if ray_direction[0] < 0 else np.array([1, 0, 0])
            elif t_enter == t1[1]:
                normal_vector= np.array([0, -1, 0]) if ray_direction[1] < 0 else np.array([0, 1, 0])
            elif t_enter == t1[2]:
                normal_vector= np.array([0, 0, -1]) if ray_direction[2] < 0 else np.array([0, 0, 1])
            return t_enter, Intersection_Point, normal_vector
        return None, None, None
