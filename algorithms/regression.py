import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .base import BaseRegression


class LinearRegression(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, w) + b


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

    @staticmethod
    def _sigmoid(x):
        return 1 / (np.exp(-x) + 1)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


# Testing
if __name__ == "__main__":
    # -- Linear regression -- #
    print("Linear regression")

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

    print()

    # -- Logistic Regression -- #
    print("Logistic Regression")

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = LogisticRegression(learning_rate=0.0002, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print(f"Accuracy: {accuracy_score(predictions, y_test)}")
    print(f"Report: {classification_report(y_test, predictions)}")
