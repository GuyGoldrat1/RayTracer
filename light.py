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

    def get_direction(self,shifted_point): #computes the direction vector from the intersection point to the light source and normalizes it.
        return self.Normalize(self.position- shifted_point)

    def get_euclidean_distance(self, intersection): # computes the Euclidean distance between the intersection point and the light source.
        return np.linalg.norm(self.position - intersection)





    # def get_irradiance(self, dist_light, NdotL):
    #     return self.color * NdotL / (dist_light**2.) * 100

    #