import numpy as np
class LinearRegression:
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None # a vector where number of features taken = number of element example: [w1, w2, w3] for y = w1a + w2b+ w3c + bias
        self.bias = None # a scalar
    
    def gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias # an array/vector of y predicted for each samples
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) * 2 # an array/vector of derivative for each features
            db = (1 / n_samples) * np.sum(y_predicted - y) * 2 # a scalar of bias

            self.weights -= dw * self.learning_rate
            self.bias -= db * self.learning_rate

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias # an array/vector of y predicted for each samples
    
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)