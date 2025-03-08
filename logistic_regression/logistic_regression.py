import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.001, n_iterations = 10000, threshold = 0.53):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.threshold = threshold

    def gradient_descent_fit(self, X, y):
        """
        X: a numpy array with dimension of (n_samples, n_features) or sample training data
        y: a numpy array with dimension of (n_samples, 1) or result training data
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (2/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2/n_samples) * np.sum(y_predicted - y)

            self.weights -= dw * self.learning_rate
            self.bias -= db * self.learning_rate


    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_classification = [1 if x > self.threshold else 0 for x in y_predicted]
        return y_predicted_classification, self.weights, self.bias

    def sigmoid(self, S):
        """
            S: an numpy array
            return sigmoid function of each element of S
        """
        return 1/(1 + np.exp(-S))