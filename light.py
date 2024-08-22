import numpy as np

class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        self.position = position
        self.color = color
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius
        self.position = np.array(position)
        self.color = np.array(color)

    def Normalize(self, vector):
        return vector / np.linalg.norm(vector)

    def get_direction(self,shifted_point):
        return self.Normalize(self.position- shifted_point)

    def get_euclidean_distance(self, intersection): 
        return np.linalg.norm(self.position - intersection)



