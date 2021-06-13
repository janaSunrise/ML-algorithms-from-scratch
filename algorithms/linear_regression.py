import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from .base import BaseAlgorithm


class LinearRegression(BaseAlgorithm):
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
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


# Testing
if __name__ == "__main__":
    # Prepare the data
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and predict with the regressor
    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    # Print the MSE
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean squared error: {mse}")

    # Print accuracy
    print(f"Accuracy: {r2_score(y_test, predictions)}")
