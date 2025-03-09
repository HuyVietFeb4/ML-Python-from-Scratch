import numpy as np
# Gaussian Naive Bayes
class NaviveBayes:
    def __init__(self):
        self._classes = None
        self.n_classes = 0 # >= 2
        self._means = None
        self._vars = None
        self._priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._means = np.zeros((n_classes, n_features), dtype=np.float64)
        self._vars = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_class = X[y == c] # Boolean indexing. 
            # 1. y == c: Return an np array with the size y of True and False elements. For each element in the same index in y if == c it is True else False
            # 2. X[y == c]: Return an np array of X where the index is only True based on previous array. An np array that being return is size (c_appearances, n_features)
            # Purpose: to groups the feature data (X) by class (c) so that you can compute class-specific statistics like mean, variance, and priors
            self._means[idx, :] = X_class.mean(axis=0)
            # In documentation: axis=0 = "along the index" (process columns by moving down rows). axis=1 = "along the columns" (process rows by moving across columns).
            self._vars[idx, :] = X_class.var(axis=0)
            self._priors[idx] = X_class.shape[0] / float(n_samples) # c_appearances / n_samples

    def predict(self, X):
        y_predict = [self._predict(x) for x in X]
        return y_predict
    
    def _predict(self, x):
        """
        x(1, n_features): np array
        """
        posteriors = []
        for idx, _ in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x))) # log(p(feature1 | class)) + log(p(feature2 | class)) + log(p(feature3 | class))
            posteriors.append(prior + posterior)

        return self._classes[np.argmax(posteriors)]
        
    def _pdf(self, class_idx, x): # probability density function
        mean = self._means[class_idx]
        var = self._vars[class_idx]
        numerator = np.exp( -( (x - mean) ** 2 ) / ( 2 * var ))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator