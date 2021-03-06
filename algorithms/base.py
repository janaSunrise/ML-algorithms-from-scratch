import abc


class BaseAlgorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplementedError
