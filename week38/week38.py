import numpy as np
import matplotlib.pyplot as plt
import sklearn.pipeline as pipe
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def generate_design_matrix(x, poly_deg):
    X = np.ones(shape=(n, poly_deg+1))
    for p in range(poly_deg):
        X[:, p+1] = x[:, 0]**(p+1)
    return X

def least_squares_method(X, y):
    beta =  np.linalg.pinv(X.T @ X) @ X.T @ y
    y_tilde = X @ beta
    return y_tilde

def get_bias(x, y):
    return (x - np.mean(y))**2

np.random.seed()
n = 200
max_deg = 15
n_bootstraps = 100

x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
poly_degs = [x for x in range(max_deg+1)]
mse_list = np.zeros(len(poly_degs))
var_list = np.zeros(len(poly_degs))
bias_list = np.zeros(len(poly_degs))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

for deg in poly_degs:
    # Fit model
    model = pipe.make_pipeline(StandardScaler(), PolynomialFeatures(degree=deg), LinearRegression(fit_intercept=False))
    y_pred = np.empty((y_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        x_, y_ = resample(x_train, y_train)
        y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()
    
    print(y_pred.shape, y_test.shape)
    MSE = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
    bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
    var = np.mean(np.var(y_pred, axis=1, keepdims=True))
    mse_list[deg] = MSE
    var_list[deg] = var
    bias_list[deg] = bias
    print('Polynomial degree:', deg)
    print('Error:', mse_list[deg])
    print('Bias^2:', bias_list[deg])
    print('Var:', var_list[deg])
    print(f'Error = Bias^2 + Var: {mse_list[deg]} = {bias_list[deg]} + {var_list[deg]}')


    # Plot main plot
    # fig, ax = plt.subplots()
    # ax.scatter(x, y, label='Data', color='r', marker='x')
    # ax.scatter(x_test, np.mean(y_pred, axis=1, keepdims=True), c='b', label='Prediction')
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$y$')
    # ax.set_title(f'Polynomial degree: {deg}')
    # ax.legend(loc='best')
    # plt.show()

# Plot metrics
fig, ax = plt.subplots()
ax.plot(poly_degs, mse_list, label='MSE')
ax.plot(poly_degs, var_list, label='Variance')
ax.plot(poly_degs, bias_list, label=r'Bias$^2$')
ax.set_xlabel(r'Polynomial deg')
plt.legend(loc='best')
plt.show()

#### Discussion ####
# With too few data points the model will not be able to sufficiently fit the data at the ends.
# This leads to extreme prediciton values at the ends. Using 100 data points or more seems to eliminate this problem.
# As expected, the bias starts out large with a very low variance for the first polynomial degrees. 
# The model seems to fit the data the best at around polynomial degree 8-10. 
# Increasing the number of bootstraps seem to make the variance and bias (and consequently the MSE) more stable.
# At 100 bootstraps, the bias-variance tradeoff is very clear. However, a weird spike in the variance appears
# often at a polynomial degree of 12 and 14. 