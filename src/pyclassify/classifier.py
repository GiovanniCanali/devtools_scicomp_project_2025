from .utils import distance, majority_vote

class kNN():
    
    def __init__(self, k: int):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k

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

    def __call__(
            self,
            data: tuple[list[list[float]], list[int]],
            new_points: list[list[float]]
        ):
        X, y = data
        return [
            majority_vote(
                self._get_k_nearest_neighbors(X, y, point)
            )
            for point in new_points
        ]
