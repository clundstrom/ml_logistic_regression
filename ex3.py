import numpy as np
import matplotlib.pyplot as plt
from os import path
import util
from sklearn.model_selection import train_test_split

data = np.genfromtxt(path.join('data/breast_cancer.csv'), delimiter=',', dtype=float)

data = np.array(data)
data[data == 2] = 0  # benign
data[data == 4] = 1  # malignant

X = np.array([data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5],
              data[:, 6], data[:, 7], data[:, 8]]).T

y = np.array([data[:, 9]]).T

# Split data int training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Standardize data
X_train_n, feature_mean, feature_std = util.standardizeSet(X_train)
X_train_ne = util.extendMatrix(X_train_n)

# Train model -> Find beta
alpha = 0.2
iterations = 1000
print(f"Hyperparams: alpha {alpha} iter {iterations}")

beta = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)  # Start from beta 0
beta, plots = util.GDLogistic(iterations, alpha, beta, X_train_ne, y_train, True)  # Find new beta
print("Final cost func: ", util.costLogistic(X_train_ne, y_train, beta))

plt.plot(range(0, iterations), plots)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

errs = util.trainingErrs(X_train_ne, beta, y_train)
accuracy = np.divide((X_train_ne.shape[0]-errs), X_train_ne.shape[0])
print(f"Accuracy on training set: {round(accuracy*100, 2)} %")

errs = util.trainingErrs(util.extendMatrix(X_test), beta, util.extendMatrix(y_test))
accuracy = np.divide((X_test.shape[0]-errs), X_test.shape[0])
print(f"Accuracy on test set: {round(accuracy*100, 2)} %")


# The results are around the same for the test set no matter how I shuffle the set.
# Depending on the size of the test set one error will account for
# a larger or smaller accuracy of the total.

