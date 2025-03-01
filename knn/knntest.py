import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.preprocessing import LabelEncoder
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# iris = pd.read_csv('./Iris.csv')

# X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
# y = iris['Species'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)

# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# plt.figure()
# plt.scatter(X[:, 1], X[:, 2], c = y_encoded, cmap = cmap, edgecolors = 'k', s = 20)
# plt.show()

penguins = pd.read_csv('./penguins.csv')
penguins.dropna()
X = penguins[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']].values
y = penguins['species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
plt.figure()
plt.scatter(X[:, 1], X[:, 3], c = y_encoded, cmap = cmap, edgecolors = 'k', s = 20)
plt.show()

# from knn import KNN
# def percentage(part, whole):
#     return 100 * float(part)/float(whole)
# for k in range(100):
#     clf = KNN(k+1)
#     clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)

#     accuracy = percentage(np.sum(y_test == predictions),len(y_test))
#     print(f"For k = {k+1}, accuracy = {accuracy}%")
