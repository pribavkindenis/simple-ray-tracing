import numpy as np


class Ray:

    def __init__(self,
                 n: int,
                 pos_vector: np.ndarray,
                 dir_vector: np.ndarray):
        self.n = n
        self.pos_vector = pos_vector
        self.dir_vector = dir_vector
        self.__validate_init_data()
        self.__normalize_dir_vector()

    def get_point(self, t: float) -> np.ndarray:
        return self.pos_vector + self.dir_vector * t

    def __normalize_dir_vector(self):
        self.dir_vector = self.dir_vector / np.linalg.norm(self.dir_vector)

    def __validate_init_data(self):
        if len(self.pos_vector.shape) != 1 or len(self.pos_vector) != self.n:
            raise ValueError("Position vector must have shape (1, n)")
        if len(self.dir_vector.shape) != 1 or len(self.dir_vector) != self.n:
            raise ValueError("Direction vector must have shape (1, n)")
