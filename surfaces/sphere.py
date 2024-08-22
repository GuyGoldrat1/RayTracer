import numpy as np


class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index -1

    def intersect(self, incoming_ray):
        vector_to_center = incoming_ray.position - self.position
        b_coeff = 2.0 * np.dot(incoming_ray.direction, vector_to_center)
        c_coeff = np.dot(vector_to_center, vector_to_center) - self.radius ** 2
        discriminant = b_coeff ** 2 - 4 * c_coeff
        intersection_distance = 0
        if discriminant > 0:
            sqrt_discriminant = np.sqrt(discriminant)
            distance1 = (-b_coeff + sqrt_discriminant) / 2
            distance2 = (-b_coeff - sqrt_discriminant) / 2
            if distance1 > 0 and distance2 > 0:
                intersection_distance = min(distance1, distance2)
            elif distance1 > 0:
                intersection_distance = distance1
            elif distance2 > 0:
                intersection_distance = distance2
            else:
                return None, None, None  

            intersection_position = incoming_ray.position + intersection_distance * np.array(incoming_ray.direction)
            surface_normal = (intersection_position - self.position) / self.radius
            return intersection_distance, intersection_position, surface_normal
        return None, None, None
            
