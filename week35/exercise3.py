import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed()
n = 50
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

########## a) ##########
def generate_design_matrix(x, poly_deg):
    X = np.ones(shape=(n, poly_deg+1))
    for p in range(poly_deg):
        X[:, p+1] = x[:, 0]**(p+1)
    return X

X = generate_design_matrix(x, 5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

########## b) ##########
def MSE(y, y_tilde):
    return np.sum((y_tilde - y)**2)/n

def least_squares_method(X, y):
    beta =  np.linalg.pinv(X.T @ X) @ X.T @ y
    y_tilde = X @ beta
    return y_tilde

y_tilde = least_squares_method(X_train, y_train)
y_predict = least_squares_method(X_test, y_test)
MSE_train = MSE(y_train, y_tilde)
MSE_test = MSE(y_test, y_predict)
print(f'5th order:\nMSE_train = {MSE_train}\nMSE_test = {MSE_test}')

X_train_plot, y_tilde_plot = zip(*sorted(zip(X_train[:, 1], y_tilde)))
X_test_plot, y_predict_plot = zip(*sorted(zip(X_test[:, 1], y_predict)))

plt.plot(X_train_plot, y_tilde_plot, 'g', label='Train; 5th order')
plt.plot(X_test_plot, y_predict_plot, 'g--', label='Predict', zorder=2)
plt.title('task b)')
plt.scatter(x, y, c='r', label='Data')
plt.legend(loc='best')
plt.show()

########## c) ##########
degree = 15
X = generate_design_matrix(x, degree)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
y_tilde = least_squares_method(X_train, y_train)
y_predict = least_squares_method(X_test, y_test)
MSE_train = MSE(y_train, y_tilde)
MSE_test = MSE(y_test, y_predict)
print(f'15th order:\nMSE_train = {MSE_train}\nMSE_test = {MSE_test}')

X_train_plot, y_tilde_plot = zip(*sorted(zip(X_train[:, 1], y_tilde)))
X_test_plot, y_predict_plot = zip(*sorted(zip(X_test[:, 1], y_predict)))

plt.plot(X_train_plot, y_tilde_plot, 'b', label='Train; 15th order')
plt.plot(X_test_plot, y_predict_plot, 'b--', label='Predict', zorder=2)
plt.title('task c)')
plt.scatter(x, y, c='r', label='Data')
plt.legend(loc='best')
plt.show()

plt.figure()
poly_array = np.arange(1, degree+1)
MSE_train_array = np.zeros(len(poly_array))
MSE_test_array = np.zeros(len(poly_array))
for p in poly_array:
    X = generate_design_matrix(x, p)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    y_tilde = least_squares_method(X_train, y_train)
    y_predict = least_squares_method(X_test, y_test)
    MSE_train = MSE(y_train, y_tilde)
    MSE_test = MSE(y_test, y_predict)
    MSE_train_array[p-1] = MSE_train
    MSE_test_array[p-1] = MSE_test
plt.plot(poly_array, MSE_train_array, 'b', label='Train')
plt.plot(poly_array, MSE_test_array, 'r', label='Test')
plt.xlabel('Polynomial degree')
plt.ylabel('MSE')
plt.title('task c)')
plt.legend(loc='best')
plt.show()
# A polynomial of degree 8 seems to suffice for this data set.
# The test MSE did not increase after degree 8 as expected from the bias-variance tradeoff.