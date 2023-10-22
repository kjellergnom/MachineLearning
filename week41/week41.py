import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from functions import *

np.random.seed(2020)
n = 100
x = np.linspace(-3, 3, n)
noise = np.random.normal(0, 0.5, n)
# y_true = x**4 - 6*x**2 + 7*x + 1
y_true = x**2 + x + 1
y = y_true + noise
y = y.reshape(-1, 1)
poly_deg = 2
iterations = 10000
momentum = 0.3
step_size = 0.0001
n_epochs = 500
mini_batch_size = 10
include_intercept = True
t = (1, 10)
lmbda = 0.01
tol = 1e-8

X = generate_1D_design_matrix(x, poly_deg, intercept=include_intercept)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
x_test = np.linspace(-3, 3, X_test.shape[0])

beta_OLS = GradientDescent(X_train, y_train, n_iter = iterations)
beta_Ridge = GradientDescent(X_train, y_train, n_iter = iterations, lmbda=lmbda)
beta_moment = GradientDescent(X_train, y_train, n_iter = iterations, momentum = momentum)
beta_Ridge_moment = GradientDescent(X_train, y_train, n_iter = iterations, lmbda=lmbda, momentum = momentum)
beta_OLS_SGD = StochasticGradientDescent(X_train, y_train, M = mini_batch_size, n_epochs = n_epochs, time_decay=t, tol = tol)

# Sort data for plotting
if include_intercept:
    indices = np.argsort(X_test[:, 1])
    x_test = X_test[:, 1][indices]
else:
    indices = np.argsort(X_test[:, 0])
    x_test = X_test[:, 0][indices]

indices = np.argsort(X_test[:, 1])
x_test = X_test[:, 1][indices]
y_OLS = X_test[indices] @ beta_OLS
y_Ridge = X_test[indices] @ beta_Ridge
y_moment = X_test[indices] @ beta_moment
y_Ridge_moment = X_test[indices] @ beta_Ridge_moment
y_OLS_SGD = X_test[indices] @ beta_OLS_SGD

# Plot data and prediction
plt.plot(x, y_true, c='k', label='True function')
# plt.scatter(x, y, s=1, c='r', label='Data')
plt.plot(x_test, y_OLS, c ='g', ls='--', marker='x', label='OLS prediction')
# plt.plot(x_test, y_Ridge, c='b', ls='--', marker = '+', label='Ridge prediction')
# plt.plot(x_test, y_moment, c='m', ls='--', marker = '+', label='Momentum prediction')
# plt.plot(x_test, y_Ridge_moment, c='c', ls='--', marker = '+', label='Ridge Momentum prediction')
plt.plot(x_test, y_OLS_SGD, c='y', ls='--', marker = '+', label='OLS SGD prediction')
plt.legend(loc='best')
plt.show()
