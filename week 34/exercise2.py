import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

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

plt.plot(x, y_tilde, lw=5, label = 'LSM')

############### 2. ############

poly = PolynomialFeatures(degree=2)
poly = poly.fit_transform(x.reshape(-1, 1), y.reshape(-1, 1))
poly_model = LinearRegression()
poly_model.fit(poly, y.reshape(-1, 1))
y_predict = poly_model.predict(poly)

plt.scatter(x, y, c='r', label = 'Data points')
plt.plot(x, y_predict, '--', label = 'Sklearn linreg')
plt.legend(loc='best')
plt.show()

############### 3. ############

