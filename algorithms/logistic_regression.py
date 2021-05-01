import numpy as np

from .base import BaseRegression



class LogisticRegression(BaseRegression):
    @staticmethod
    def _get_linear_model(X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        linear_model = self._get_linear_model(X, w, b)

        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]

        return np.array(y_predicted_cls)

    def _approximation(self, X, w, b):
        linear_model = self._get_linear_model(X, w, b)

        return self._sigmoid(linear_model)

    def _sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)


# Testing
if __name__ == "__main__":
    pass
