import numpy as np
from collections import Counter
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X 
        self.y_train = y

    def predict(self, X):
        predict_labels = [self._predict(x) for x in X]
        return np.array(predict_labels)

    def _predict(self, x):
        # Calculate euclidean distances of all points
        eucl_distances = [self.euclidean_distances(x, x_train) for x_train in self.X_train]
        # Take k shortest distances
        # 1. Take k index of k shortest
        # 2. Take k shortest form y_train
        k_indices = np.argsort(eucl_distances)[:self.k] # have an array with indexes from 1 to 150
        k_nearest_labels = [self.y_train[i] for i in k_indices] # have an array with lables for example [0,1,2,2,2,2,1,0]
        # Return the label (0 or 1 or 2)
        most_common = Counter(k_nearest_labels).most_common(1) # if no 1 or any number return a full list of tuple (element, number of existance)
        return most_common[0][0]

    def euclidean_distances(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))