import numpy as np

from .base import BaseAlgorithm


class KNN(BaseAlgorithm):
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


# Testing
if __name__ == "__main__":
    pass
