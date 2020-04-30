import numpy as np
import matplotlib.pyplot as plt
from os import path
import util
from matplotlib.colors import ListedColormap

data = np.genfromtxt(path.join('data/admission.csv'), delimiter=',', dtype=float)

X = np.array([data[:, 0], data[:, 1]]).T
y = np.array(data[:, 2])

# Standardize features and extend matrix.
X_norm, feature_mean, feature_std = util.standardizeSet(X)
Xe_norm = util.extendMatrix(X_norm)

# Entire data set X and y. Is used for filtering the data.
normalized_data = np.array([X_norm[:, 0], X_norm[:, 1], y]).T

# Filters samples into lists
failed = list(filter(lambda chip: chip[2] == 0, normalized_data))
ok = list(filter(lambda chip: chip[2] == 1, normalized_data))

# Parse each sample into x,y coordinates
xf, yf, zf = list(zip(*failed))
x_ok, y_ok, z_ok = list(zip(*ok))

# Plot normalized samples
ax = plt.subplot(1, 1, 1)
ax.scatter(xf, yf, color='r', label='Not admitted')
ax.scatter(x_ok, y_ok, color='b', label='Admitted')
plt.show()

# Print sigmoid of matrix
arr = np.array([[0, 1], [2, 3]])
print(util.sigmoid(arr))

# Implement vectorised version of logistic cost function
beta = np.array([0, 0, 0])  # Test value should be 0.6931
print("Cost for [0,0,0]: ", util.costLogistic(Xe_norm, y, beta))

alpha = 0.5
iterations = 1000
beta = util.GDLogistic(iterations, alpha, beta, Xe_norm, y)
student = np.array([[45, 85]])
student_n = util.standardize(student, feature_mean, feature_std)

student_ne = util.extendMatrix(student_n)
print("Probability of admission: ", util.sigmoid(np.dot(student_ne, beta)))
util.trainingErrs(Xe_norm, beta, y)

# PLOT MESH GRID
h = .01 # step size in the mesh

x_min, x_max = X_norm[:, 0].min() - 0.1, X_norm[:, 0].max() + 0.1
y_min, y_max = X_norm[:, 1].min() - 0.1, X_norm[:, 1].max() + 0.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # Mesh Grid
x1, x2 = xx.ravel(), yy.ravel()  # Turn to two Nx1 arrays
XXe = util.mapFeature(x1, x2, 1)  # Extend matrix for degree 2
p = util.sigmoid(np.dot(XXe, beta))  # classify mesh ==> probabilities
classes = p > 0.5  # round off probabilities
clz_mesh = classes.reshape(xx.shape)  # return to mesh format
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  # mesh plot
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # colors
plt.figure(2)
plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, marker='.', cmap=cmap_bold)
plt.show()
