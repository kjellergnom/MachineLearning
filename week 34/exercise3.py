import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed()
n = 100
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

########## a) ##########
X = np.ones(shape=(n, 6))
X[:, 1] = x[:, 0]
X[:, 2] = x[:, 0]**2
X[:, 3] = x[:, 0]**3
X[:, 4] = x[:, 0]**4
X[:, 5] = x[:, 0]**5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

########## b) ##########
beta =  np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
y_tilde = X_train @ beta
y_predict = X_test @ beta
MSE = np.sum((y_test - y_predict)**2)/n
print(f'5th order: MSE = {MSE}')

plt.plot(x[:len(y_tilde)], y_tilde, 'g', label='Train; 5th order')
plt.plot(x[len(y_tilde):], y_predict, 'g--', label='Predict')

########## c) ##########
poly_deg = 15
X = np.ones(shape=(n, poly_deg+1))
for p in range(poly_deg):
    X[:, p+1] = x[:, 0]**(p+1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
beta =  np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
y_tilde = X_train @ beta
y_predict = X_test @ beta
MSE = np.sum((y_test - y_predict)**2)/n
print(f'15th order: MSE = {MSE}')

plt.scatter(x, y, c='r', label='Data')
# plt.plot(x[:len(y_tilde)], y_tilde, 'b', label='Train; 15th order')
# plt.plot(x[len(y_tilde):], y_predict, 'b--', label='Predict')
plt.legend(loc='best')
plt.show()