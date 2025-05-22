from .utils import distance, majority_vote, distance_numpy, distance_numba
from line_profiler import profile
import numpy as np

class kNN():
    
    def __init__(self, k: int, backend: str = 'plain'):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if not isinstance(backend, str):
            raise ValueError("backend must be a string.")
        if backend not in ['plain', 'numpy', 'numba']:
            raise ValueError("backend must be either 'plain' or 'numpy'.")

        self.k = k
        self.backend = backend

        if backend == 'plain':
            self.distance = distance
        elif backend == 'numpy':
            self.distance = distance_numpy
        elif backend == 'numba':
            self.distance = distance_numba

    @profile
    def _get_k_nearest_neighbors(
            self,
            X: list[list[float]],
            y: list[int],
            x: list[float]
        ) -> list[int]:
        dist = [distance(x, point) for point in X]
        sorted_neighbors = sorted(enumerate(dist), key=lambda pair: pair[1])
        k_neighbors = [y[i] for i, _ in sorted_neighbors[:self.k]]
        return k_neighbors

    @profile
    def __call__(
            self,
            data: tuple[list[list[float]], list[int]],
            new_points: list[list[float]]
        ):
        X, y = data

        if self.backend == 'numpy':
            X = np.array(X)
            new_points = np.array(new_points)

        return [
            majority_vote(
                self._get_k_nearest_neighbors(X, y, point)
            )
            for point in new_points
        ]
