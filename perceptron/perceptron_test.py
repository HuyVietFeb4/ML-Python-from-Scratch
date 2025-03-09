import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# iris = pd.read_csv('./Iris.csv')
# X = iris.loc[:99,['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
# y = iris.loc[:99,'Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1}).values
breast_cancer = pd.read_csv('./breast_cancer.csv')
X = breast_cancer[["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]].values
y = breast_cancer['diagnosis'].map({'B': 1, 'M': 0}).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

def scratch_accuracy(y_predict, y_true):
    """
    y_predict: np array of predicted values
    y_true: np array of true values
    """
    accuracy = np.sum(y_predict == y_true)/len(y_true)
    return accuracy

from perceptron import Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_predict = perceptron.predict(X_test)

perceptron_skl = linear_model.Perceptron()
perceptron_skl.fit(X_train, y_train)
#Making prediction on test data
y_skl_pred = perceptron_skl.predict(X_test)
#Finding accuracy
accuracy_skl = metrics.accuracy_score(y_test, y_skl_pred)


for i in range(len(y_predict)):
    print(f"Predicted: {y_predict[i]}. True values: {y_test[i]}")
print(f"From scratch predictions accuracy: {scratch_accuracy(y_predict, y_test)}")
print(f"Scikit learn predictions accuracy: {accuracy_skl}")


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
# x0_1 = np.amin(X_train[:, 0])
# x0_2 = np.amax(X_train[:, 0])
# x1_1 = (-perceptron.weights[0] * x0_1 - perceptron.bias) / perceptron.weights[1]
# x1_2 = (-perceptron.weights[0] * x0_2 - perceptron.bias) / perceptron.weights[1]
# ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
# ymin = np.amin(X_train[:, 1])
# ymax = np.amax(X_train[:, 1])
# ax.set_ylim([ymin - 3, ymax + 3])
# plt.show()
