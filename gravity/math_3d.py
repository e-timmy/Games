import numpy as np
from math import tan, pi


class Matrix4x4:
    @staticmethod
    def perspective(fov, aspect, near, far):
        f = 1.0 / tan(fov / 2.0)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ])

    @staticmethod
    def translate(x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_np(self):
        return np.array([self.x, self.y, self.z, 1.0])

    @staticmethod
    def from_np(arr):
        return Vector3(arr[0], arr[1], arr[2])


def project_point(point, projection_matrix):
    v = point.to_np()
    transformed = np.dot(projection_matrix, v)
    if transformed[3] != 0:
        transformed = transformed / transformed[3]
    return Vector3(transformed[0], transformed[1], transformed[2])