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
MSE_train = np.sum((y_tilde - y_train)**2)/n
MSE_test = np.sum((y_predict - y_test)**2)/n
print(f'5th order:\nMSE_train = {MSE_train}\nMSE_test = {MSE_test}')

X_train_plot, y_tilde_plot = zip(*sorted(zip(X_train[:, 1], y_tilde)))
X_test_plot, y_predict_plot = zip(*sorted(zip(X_test[:, 1], y_predict)))

plt.plot(X_train_plot, y_tilde_plot, 'g', label='Train; 5th order')
plt.plot(X_test_plot, y_predict_plot, 'g--', label='Predict', zorder=2)

########## c) ##########
poly_deg = 15
X = np.ones(shape=(n, poly_deg+1))
for p in range(poly_deg):
    X[:, p+1] = x[:, 0]**(p+1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
beta =  np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
y_tilde = X_train @ beta
y_predict = X_test @ beta
MSE_train = np.sum((y_tilde - y_train)**2)/n
MSE_test = np.sum((y_predict - y_test)**2)/n
print(f'15th order:\nMSE_train = {MSE_train}\nMSE_test = {MSE_test}')

X_train_plot, y_tilde_plot = zip(*sorted(zip(X_train[:, 1], y_tilde)))
X_test_plot, y_predict_plot = zip(*sorted(zip(X_test[:, 1], y_predict)))

plt.plot(X_train_plot, y_tilde_plot, 'b', label='Train; 15th order')
plt.plot(X_test_plot, y_predict_plot, 'b--', label='Predict', zorder=2)
plt.scatter(x, y, c='r', label='Data')
plt.legend(loc='best')
plt.show()