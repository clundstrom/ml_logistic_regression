import numpy as np
import matplotlib.pyplot as plt
from os import path
import util

data = np.genfromtxt(path.join('data/GPUbenchmark.csv'), delimiter=',', dtype=float)

# Define X, Xe and Y
y = data[:, 6]
X = np.array([data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]]).T
Xe = util.extendMatrix(X)

# Normalize
X_norm, feature_mean, feature_std = util.standardizeSet(X)

# Extend normalized data
Xe_norm = util.extendMatrix(X_norm)

for i in range(1, 7):
    plt.xlim(-3, 3)
    util.createSubScatterPlot(util.stdFeature(Xe_norm[:, i]), y, f'Feature {i}', 'Y', 2, 3, i)

plt.xlim(-3, 3)
plt.show()

gpu = np.array([2432, 1607, 1683, 8, 8, 256])
gpu_norm = util.standardize(gpu, feature_mean, feature_std)
gpu_norm_e = np.array([1, gpu_norm[0], gpu_norm[1], gpu_norm[2], gpu_norm[3], gpu_norm[4], gpu_norm[5]])

gpu = np.array([1, 2432, 1607, 1683, 8, 8, 256])
beta = util.calcBeta(Xe, y)
print("Benchmark using normal eq: ", util.normalEq(Xe, y, gpu))
print("Cost function: ", util.cost(Xe, y, beta))
# 12.3964
beta2 = util.calcBeta(Xe_norm, y)
print("Cost function normalized: ", util.cost(Xe_norm, y, beta2))
print("Benchmark on normalized data: ", util.normalEq(Xe_norm, y, gpu_norm_e))


# Implement vectorized version of gradient descent
iterations = 10000
alpha = 0.02
start = np.array([0, 0, 0, 0, 0, 0, 0])

beta = util.minimizeBeta(iterations, alpha, start, Xe_norm, y)
print("GD final cost: ", util.cost(Xe_norm, y, beta))
print(f"Parameters: Alpha={alpha}, Iterations={iterations}")
# 12.50950 after 35 mill iterations
# .0091% of previous cost
print("Benchmark(normalized gd): ", util.predict(gpu_norm_e, beta))
