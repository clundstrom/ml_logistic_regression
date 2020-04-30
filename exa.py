import numpy as np
import matplotlib.pyplot as plt
from os import path

data = np.genfromtxt(path.join('data/girls_height.csv'), delimiter='\t', dtype=float)


def createSubScatterPlot(x, y, x_label, y_label, nr):
    ax = plt.subplot(2, 2, nr)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.scatter(x, y, marker='o', color='r')


def standardize(feature, std, mean):
    """
    Normalizes feature by
    (X - u) / std
    :param feature: feature to normalzie
    :return:
    """
    feature_norm = np.divide(np.subtract(feature, mean), std)
    return feature_norm


def stdFeature(feature):
    std = np.std(feature)
    mean = np.mean(feature)
    feature_norm = np.divide(np.subtract(feature, mean), std)
    return feature_norm


def extendMatrix(*features):
    """
    Extends a matrix with ones and x features.
    :param features:
    :return:
    """
    features = np.array(features).T
    Xe = np.c_[np.ones((len(features), 1)), features]
    return Xe


def calcBeta(Xe, y):
    """
    Calculates the beta value based on Extended matrix and expected Y values.
    :param Xe: Untransposed matrix
    :param y: Expected value
    :return: beta-value
    """
    beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
    return beta


def predict(unknown, beta):
    return np.dot(unknown, beta)


def normalEq(Xe, y, unknown):
    """
    Uses Extended matrix and y-matrix to calculate beta and
    predict result of unknown.
    :param Xe: Extended matrix
    :param y: Known results.
    :param unknown: Unknown value
    :return: Result of unknown.
    """
    beta = np.array(calcBeta(Xe, y))
    return predict(unknown, beta)


def cost(Xe, y, beta):
    """
    Cost function based on Xe, y and beta parameters.
    :param Xe: Matrix
    :param y: Target
    :param beta: Beta
    :return: Cost (float)
    """
    j = np.dot(Xe, beta) - y
    return (j.T.dot(j)) / len(Xe)


def minimizeBeta(iterations, alpha, start_beta, Xe):
    for x in range(0, iterations):
        start_beta = np.subtract(start_beta, np.dot(np.dot(alpha, Xe.T), np.subtract(np.dot(Xe, start_beta), y)))
    return start_beta


girl_height = np.array(data[:, 0])
mom_height = np.array(data[:, 1])
dad_height = np.array(data[:, 2])

fig, ax = plt.subplots(nrows=1, ncols=2)

createSubScatterPlot(mom_height, girl_height, 'Mom height', 'Girl height', 1)
createSubScatterPlot(dad_height, girl_height, 'Dad height', 'Girl height', 2)

plt.show()
print()

y = np.array(girl_height).T

# Normal equation
Xe = extendMatrix(mom_height, dad_height)
X = np.array([mom_height, dad_height])
girl = np.array([1, 65, 70])
result = normalEq(Xe, y, girl)
print("Prediction(normal): ", normalEq(Xe, y, girl))

##########

# Mean and Std of features
feature_mean = np.mean(X)
feature_std = np.std(X)

# Standardize features

X_norm = stdFeature(X)
X_norm = X_norm.T
Xe_norm = extendMatrix(X_norm[:, 0], X_norm[:, 1])

girl = np.array([65, 70])
girl_norm = standardize(girl, feature_std, feature_mean)
girl_norm_e = np.array([1, girl_norm[0], girl_norm[1]])

mom_norm = stdFeature(mom_height)
dad_norm = stdFeature(dad_height)

# plot normalized data
createSubScatterPlot(mom_norm, y, 'Mom height norm', 'Girl height', 1)
createSubScatterPlot(dad_norm, y, 'Dad height norm', 'Girl height', 2)
plt.show()

# Print cost of beta for normal eq
beta = calcBeta(Xe, y)
print("Cost(normal): ", cost(Xe, y, beta))
print("Prediction(normalized): ", normalEq(Xe_norm, y, girl_norm_e))

# Implement vectorized version of gradient descent
iterations = 20_000_000
alpha = 0.000001

costs = []
beta = minimizeBeta(iterations, alpha, np.array([0,0,0]), Xe)

print("Prediction(gd): ", predict(girl_norm_e, beta))

