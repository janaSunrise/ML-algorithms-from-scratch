import abc

import numpy as np


class BaseAlgorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplementedError


class BaseRegression(BaseAlgorithm):
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
        # Assign the variables
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Weights and bias
        self.weights, self.bias = None, None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights, self.bias = np.zeros(n_features), 0

        # Minimizing loss, and finding the correct Weights and biases using Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = self._approximation(X, self.weights, self.bias)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict(self, X, w, b):
        raise NotImplementedError

    def _approximation(self, X, w, b):
        raise NotImplementedError
