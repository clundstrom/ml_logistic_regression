import numpy as np
import matplotlib.pyplot as plt
import time


def timing(f):
    """
    Function used to wrap other functions and measure time.
    Code borrowed from Stack Overflow and is only used for measuring time taken for a function.
    :param f: Function to be called.
    :return:
    """

    def wrap(*args):
        start = time.time()
        ret = f(*args)
        end = time.time()
        print('{:s} took {:.3f} ms'.format(f.__name__, (end - start) * 1000.0))
        return ret

    return wrap


def createSubScatterPlot(x, y, x_label, y_label, row, col, nr):
    ax = plt.subplot(row, col, nr)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.scatter(x, y, marker='o', color='r', s=.8)
    return ax


def standardizeSet(X):
    """
    Normalizes feature by
    (X - u) / std
    :param feature: feature to normalzie
    :return:
    """
    norm = []
    means = []
    stds = []
    for column in X.T:
        mean = np.mean(column)
        std = np.std(column, ddof=1)
        means.append(mean)
        stds.append(std)
        norm.append(np.divide((np.subtract(column, mean)), std))
    return np.array(norm).T, np.array(means), np.array(stds)


def standardize(target, mean, std):
    """
    Standardizes a target value with a supplied mean and std
    """
    arr = []
    shape = target.shape
    for idx, value in enumerate(target):
        arr.append(np.divide((np.subtract(value, mean[idx])), std[idx]))

    return np.array(arr).reshape(shape)


def extendMatrix(X):
    """
    Extends a matrix with ones and x features.
    :param features:
    :return:
    """
    Xe = np.c_[np.ones((len(X), 1)), X]
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


def predict(X, beta):
    """Return dot product of X and Beta"""
    return np.dot(X, beta)


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


@timing
def minimizeBeta(iterations, alpha, start_beta, Xe, y):
    for x in range(0, iterations):
        start_beta = np.subtract(start_beta, np.dot(np.dot(alpha, Xe.T), np.subtract(np.dot(Xe, start_beta), y)))
    return start_beta


def stdFeature(feature):
    std = np.std(feature)
    mean = np.mean(feature)
    feature_norm = np.divide(np.subtract(feature, mean), std)
    return feature_norm


def sigmoid(X):
    return np.divide(1, (1 + np.exp(-X)))


def costLogistic(X, y, beta):
    """
        Logistic cost function based on X, y and beta parameters.
        :param X: Matrix
        :param y: Target
        :param beta: Beta
        :return: Cost (float)
        """
    g_xb = np.dot(X, beta)
    sig = sigmoid(g_xb)

    first_part = np.dot(y.T, np.log(sig))
    second_part = np.dot(np.subtract(1, y).T, np.log(np.subtract(1, sig)))

    res = np.dot(-(1 / len(X)), np.add(first_part, second_part))
    return res


def GDLogistic(iterations, alpha, beta, X, y, plot=False):
    if plot:
        plots = []
        for x in range(0, iterations):
            dotted = np.dot(X, beta)
            subtract_sig = np.subtract(sigmoid(dotted), y)
            gradient = alpha / len(X) * np.dot(X.T, subtract_sig)
            beta = np.subtract(beta, gradient)
            plots.append(costLogistic(X, y, beta)[0][0])
        return beta, plots
    else:
        for x in range(0, iterations):
            dotted = np.dot(X, beta)
            subtract_sig = np.subtract(sigmoid(dotted), y)
            gradient = alpha / len(X) * np.dot(X.T, subtract_sig)
            beta = np.subtract(beta, gradient)
    print(costLogistic(X, y, beta))
    return beta


def trainingErrs(Xe, beta, y):
    p = np.dot(Xe, beta).reshape(-1, 1)
    p = sigmoid(p)  # Probabilities in range [0,1]
    pp = np.round(p)
    yy = y.reshape(-1, 1)
    errs = np.sum(yy != pp)
    print("Errors: ", errs)
    return errs


def mapFeature(X1, X2, D):
    one = np.ones([len(X1), 1])
    Xe = np.c_[one, X1, X2]  # Start with [1,X1,X2]
    for i in range(2, D + 1):
        for j in range(0, i + 1):
            Xnew = X1 ** (i - j) * X2 ** j  # type (N)
            Xnew = Xnew.reshape(-1, 1)  # type (N,1) required by append
            Xe = np.append(Xe, Xnew, 1)  # axis = 1 ==> append column
    return Xe
