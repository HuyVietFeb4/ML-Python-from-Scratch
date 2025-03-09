import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import naive_bayes

# iris = pd.read_csv('./Iris.csv')
# X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
# y = iris['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).values
breast_cancer = pd.read_csv('./breast_cancer.csv')
X = breast_cancer[["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]].values
y = breast_cancer['diagnosis'].map({'B': 1, 'M': 0}).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

def accuracy(y_predict, y_true):
    """
    y_predict: np array of predicted values
    y_true: np array of true values
    """
    accuracy = np.sum(y_predict == y_true)/len(y_true)
    return accuracy

from naive_bayes import NaviveBayes

nb_cls = NaviveBayes()
nb_cls.fit(X_train, y_train)
y_predict = nb_cls.predict(X_test)

for i in range(len(y_predict)):
    print(f"Predicted: {y_predict[i]}. True values: {y_test[i]}")
print(f"From scratch predictions accuracy: {accuracy(y_predict, y_test)}")

skl_nb = naive_bayes.GaussianNB()
skl_nb.fit(X_train, y_train)
y_skl_predict = skl_nb.predict(X_test)
print(f"Scikit-learn predictions accuracy: {accuracy(y_skl_predict, y_test)}")
