import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

x = np.linspace(0, 1, 100)
noise = np.random.randn(100,1)
y = x**2 + 2.0 + 0.1*np.squeeze(noise)
print(np.shape(x))
print(np.shape(y))
############### 1. ############
design_matrix = np.ones((len(x), 3))
design_matrix[:, 1] = x[:]
design_matrix[:, 2] = x[:]**2
beta = np.linalg.inv(design_matrix.transpose() @ design_matrix) @ design_matrix.transpose() @ y
y_model = beta[0] + beta[1]*x + beta[2]*x**2

plt.scatter(x, y, c='r', label = 'Data points')
plt.plot(x, y_model, label = 'LSM')
plt.legend(loc='best')
plt.show()
############### 2. ############

############### 3. ############

