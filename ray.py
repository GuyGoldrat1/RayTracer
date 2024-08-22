import numpy as np


class Ray:
    def __init__(self, position, direction):
        self.position = position
        self.direction = self.normalize(direction)

    def normalize(self, direction):
        return direction / np.linalg.norm(direction)


