import numpy as np
from os import path
import util
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

data = np.genfromtxt(path.join('data/GPUbenchmark.csv'), delimiter=',', dtype=float)

y = data[:, 6]
X = np.array([data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]]).T
Xe = util.extendMatrix(X)

count = 0
forward_set = np.empty((18, 1))
data_sets = []

for k in range(0, X.shape[1]):
    best_mse = -1  # Starting value for best mean square error
    for p in range(k, X.shape[1]):  # starting from k is the same as p-k

        count += 1  # Keep track of models trained
        temp = forward_set.copy()  # Create deep copy of forward model to avoid changing the optimized model
        X_train = np.c_[temp, X[:, p]]  # Concat previous forward model with new feature for training
        beta = util.calcBeta(X_train, y)  # Calculate beta
        mse = util.cost(X_train, y, beta)  # Calculate cost
        print(f"Training MSE: {mse}, Column: {p}")

        # Save best feature if found
        if mse < best_mse or best_mse < 0:
            best_mse = mse  # assign new best mse
            best_index = p  # index of best column
            keep_feature = X[:, p]  # actual feature

    # Add the column to M+1
    forward_set = np.c_[forward_set, keep_feature]
    # Move chosen feature to index k to avoid singular matrix when dotting and inversing
    X[:, [k, best_index]] = X[:, [best_index, k]]
    print(f"Lowest MSE for {k + 1} features: {best_mse}")
    data_sets.append(forward_set)  # Save the best 18 x n sets to a list

print(f"Feature selection iterations: {count}")

# Cross validation of the best models
linreg = LinearRegression(fit_intercept=False)  # column 1 is already added
kf = KFold(n_splits=3)
for idx, set in enumerate(data_sets):
    M_i = linreg.fit(X=set, y=y)
    score = cross_val_score(M_i, set, y, cv=kf, scoring='neg_mean_squared_error') * -1
    print(f"Model_{idx+1} 3-fold MSE: {score.mean()}")
