import numpy as np

class Perceptron:
    def __init__(self, learning_rate = 0.001, n_iterations = 10000):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.bias = None
        self.weights = None
        self.activation_function = self._ReLU

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_train = np.array([1 if i > 0 else 0 for i in y])
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # perceptron update rule
                update = self.learning_rate * (y_train[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    def _ReLU(self, x):
        return np.where(x >= 0, 1, 0)
    def _SoftPlus(self, x):
        return np.log(1 + np.exp(x))
    def _SigMoid(self, x):
        return 1 / (1 + np.exp(-x))