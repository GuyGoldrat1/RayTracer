import numpy as np


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = position
        self.look_at = look_at
        self.up_vector = up_vector
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.position = np.array(self.position)
        self.look_at = np.array(self.look_at)
        self.up_vector = np.array(self.up_vector)
        self.direction= np.subtract(position, look_at)


        self.Vz = self.Normalize(self.look_at - self.position) 
        self.Vx = self.Normalize(np.cross(self.Vz, self.up_vector)) 
        self.Vy = np.cross(self.Vx, self.Vz)  

        self.screen_center = self.position + self.Vz * self.screen_distance
        self.screen_height = 0
        self.pixel_width = 0
        self.pixel_height = 0

    def Normalize(self, vector):
        return vector / np.linalg.norm(vector)

    def update_screen_ratio(self, width, height):
        ratio = float(width) / float(height)
        self.screen_height = self.screen_width / ratio

        self.pixel_width = self.screen_width / width
        self.pixel_height = self.screen_height / height

    def screen_location(self, i, j, half_width, half_height):
        return self.screen_center + ((i - half_width) * self.pixel_width * self.Vx) + (
                    (j - half_height) * self.pixel_height * self.Vy)

    def ray_direction(self, pixel_position):
        return self.Normalize(pixel_position - self.position)


