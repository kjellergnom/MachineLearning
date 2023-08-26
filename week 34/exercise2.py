import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.random.rand(100, 1)
x = np.sort(x, axis=None)
noise = np.random.randn(100,1)
y = x**2 + 2.0 + 0.1*np.squeeze(noise)

############### 1. ############

design_matrix = np.ones((len(x), 3))
design_matrix[:, 1] = x[:]
design_matrix[:, 2] = x[:]**2
beta = np.linalg.inv(design_matrix.transpose() @ design_matrix) @ design_matrix.transpose() @ y
y_tilde = beta[0] + beta[1]*x + beta[2]*x**2

# plt.scatter(x, y, c='r', label = 'Data points')
# plt.plot(x, y_tilde, label = 'LSM')
# plt.legend(loc='best')
# plt.show()

############### 2. ############

x = np.random.rand(100,1)
y = 2.0+5*x*x+0.1*np.random.randn(100,1)
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(design_matrix)
predict_ = poly.fit

plt.scatter(x, y, c='r', label = 'Data points')
plt.plot(x_new, y_predict, label = 'Sklearn linreg')
plt.legend(loc='best')
plt.show()

############### 3. ############

