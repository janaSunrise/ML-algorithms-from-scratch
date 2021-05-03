import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .base import BaseAlgorithm


class NaiveBayes(BaseAlgorithm):
    def __init__(self):
        self._classes = None
        self._mean = None
        self._var = None
        self._priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])

            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior

            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator


# Testing
if __name__ == "__main__":
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(predictions, y_test)}")
    print(f"Report: {classification_report(y_test, predictions)}")
