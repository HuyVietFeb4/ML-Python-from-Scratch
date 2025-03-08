import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# iris = pd.read_csv('./Iris.csv')

# X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
# y = iris['Species'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)

# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# plt.figure()
# plt.scatter(X[:, 1], X[:, 2], c = y_encoded, cmap = cmap, edgecolors = 'k', s = 20)
# plt.show()

# iris = pd.read_csv('./iris.csv')
# iris = iris.dropna()
# X = iris[['SepalWidthCm']].values
# y = iris['SepalLengthCm'].values

car = pd.read_csv('./car_price_dataset.csv')
car = car.dropna()
min_index = 0
max_index = 100
X = car.loc[:100, ['Mileage']].values
y = car.loc[:100, 'Price'].values

X_all = car.loc[max_index:, ['Mileage']].values
y_all = car.loc[max_index:, 'Price'].values

X, y = datasets.make_regression(
    n_samples=200, n_features=1, noise=40, random_state=100
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

from linear_regression import LinearRegression

regressor = LinearRegression()
regressor.gradient_descent_fit(X_train, y_train)
y_predictions, scratch_weights, scratch_bias = regressor.predict(X_test)

regressor_scikit_learn = linear_model.LinearRegression()
regressor_scikit_learn.fit(X_train, y_train)
print("scikit-learn's solution : weight = ", scratch_weights, "bias = ", scratch_bias)
print("our solution : weight = ", regressor_scikit_learn.coef_[0], "bias = ", regressor_scikit_learn.intercept_)

mse_value = mean_squared_error(y_test, y_predictions)
print(mse_value)


y_pred_line, predicted_weights, predicted_bias = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(10,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
m3 = plt.scatter(X_all, y_all, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()