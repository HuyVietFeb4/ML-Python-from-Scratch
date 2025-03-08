import numpy as np
class LinearRegression:
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None # a vector where number of features taken = number of element example: [w1, w2, w3] for y = w1a + w2b+ w3c + bias
        self.bias = None # a scalar
    
    def gradient_descent_fit(self, X, y):
        """
        X: a numpy array with dimension of (n_samples, n_features) or sample training data
        y: a numpy array with dimension of (n_samples, 1) or result training data
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias # an array/vector of y predicted for each samples. Dimension: (n_samples, 1)
            # when making calculations with scalar and vector, boardcasting happens
            # .dot when facing 2d array is vector multiplication
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y)) # an array/vector of derivative for each features. Dimension: (n_features, 1)
            db = (2 / n_samples) * np.sum(y_predicted - y) # a scalar of bias

            self.weights -= (dw * self.learning_rate)
            self.bias -= (db * self.learning_rate)

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias # an array/vector of y predicted for each samples
        return y_approximated, self.weights, self.bias
    

    