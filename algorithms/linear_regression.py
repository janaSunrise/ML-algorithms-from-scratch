import numpy as np

from .base import BaseRegression


class LinearRegression(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, w) + b


# Testing
if __name__ == "__main__":
    pass
