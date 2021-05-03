from collections import Counter

import numpy as np

from .base import BaseAlgorithm


class KNN(BaseAlgorithm):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]

        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

    @staticmethod
    def _euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


# Testing
if __name__ == "__main__":
    pass
