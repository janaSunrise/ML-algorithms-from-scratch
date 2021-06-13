import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components, self.mean = None, None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Get the covariance
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Eigen vector sort
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]

        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0: self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)


# Testing
if __name__ == "__main__":
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    # Display the shapes
    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    # Get the projections
    x1, x2 = X_projected[:, 0], X_projected[:, 1]

    # Create the graph
    plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
