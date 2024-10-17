# custom Naive Bayes implementation using numpy
import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None  # Store unique class labels
        self.mean = None  # Mean of each feature per class
        self.var = None  # Variance of each feature per class
        self.priors = None  # Prior probabilities of each class

    def fit(self, X, y):
        """Fit the model to the training data."""
        self.classes = np.unique(y)  # Identify unique class labels
        n_features = X.shape[1]
        n_classes = len(self.classes)

        # Initialize mean, variance, and priors for each class
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]  # Select data of class `c`
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]

    def _gaussian_density(self, class_idx, x):
        """Compute Gaussian density function."""
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-0.5 * ((x - mean) ** 2) / (var + 1e-6))
        denominator = np.sqrt(2 * np.pi * var + 1e-6)
        return numerator / denominator

    def predict(self, X):
        """Predict class labels for given data."""
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """Predict the class for a single data point."""
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])  # Use log for numerical stability
            class_conditional = np.sum(np.log(self._gaussian_density(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
