import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from .base import BaseRegression


class LinearRegression(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, w) + b


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    return corr**2


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
