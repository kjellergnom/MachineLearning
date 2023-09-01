import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

MSE = mean_squared_error(y, y_predict)
R2 = r2_score(y, y_predict)
print(f'MSE = {MSE}\nR2 = {R2}')
# MSE scales proportionally with noise amplitude, while R2 scales inversely with noise amplitude.
# Too noisy data will give poor results for both MSE and R2, even if the model seems to fit the data well.