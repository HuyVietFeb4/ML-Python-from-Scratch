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
X = penguins[['flipper_length_mm']].values
y = penguins['body_mass_g'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


from linear_regression import LinearRegression

regressor = LinearRegression()
regressor.gradient_descent(X_train, y_train)
y_predictions = regressor.predict(X)

# plt.scatter(X[:, 0], y, color='b', marker='o' , s = 30)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
m3 = plt.scatter(X[:, 0], y, color='b', marker='o' , s = 30)
plt.plot(X, y_predictions, color="black", linewidth=2, label="Prediction")
plt.show()