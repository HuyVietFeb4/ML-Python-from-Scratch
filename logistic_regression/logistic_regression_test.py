import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


breast_cancer = pd.read_csv('./breast_cancer.csv')
examine_feature = 'radius_worst'
X = breast_cancer[[examine_feature]].values
y = breast_cancer['diagnosis'].map({'B': 1, 'M': 0}).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)


def accuracy(y_predict, y_true):
    """
    y_predict: np array of predicted values
    y_true: np array of true values
    """
    accuracy = np.sum(y_predict == y_true)/len(y_true)
    return accuracy
from logistic_regression import LogisticRegression

regressor = LogisticRegression()
regressor.gradient_descent_fit(X_train, y_train)
y_predictions, scratch_weights, scratch_bias = regressor.predict(X_test)
for i in range(len(y_predictions)):
    print(f"Predicted: {y_predictions[i]}. True values: {y_test[i]}")
print(f"From scratch predictions accuracy: {accuracy(y_predictions, y_test)}")

regressor_scikit_learn = linear_model.LogisticRegression()
regressor_scikit_learn.fit(X_train, y_train)
y_scikit_predict = regressor_scikit_learn.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_scikit_predict)
print(cnf_matrix)
print(f"Scikit accuracy: {(cnf_matrix[0][0] + cnf_matrix[1][1]) / (cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[1][0] + cnf_matrix[0][1])}")


# Generate a range of values for X
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# Predict probabilities using the trained model
y_probabilities = regressor_scikit_learn.predict_proba(X_range)[:, 1]  # Probability of class '1' (Benign)

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(10,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# Plot the logistic curve
plt.plot(X_range, y_probabilities, color='red', linewidth=2, label='Logistic Curve')

# Add labels and legend
plt.xlabel(examine_feature)
plt.ylabel('Diagnosis Probability (Benign)')
plt.title('Logistic Regression Curve')
plt.legend()
plt.show()

# y_pred_line, predicted_weights, predicted_bias = regressor.predict(X)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(10,6))
# m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# m3 = plt.scatter(X_all, y_all, color=cmap(0.5), s=10)
# plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
# plt.show()