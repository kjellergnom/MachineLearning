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
n = 100
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
n_bootstraps = 100
deg = 15
poly_degs = [x for x in range(16)]

mse_list = np.zeros(max(poly_degs)+1)
var_list = np.zeros(max(poly_degs)+1)
bias_list = np.zeros(max(poly_degs)+1)

for deg in poly_degs:
    X = generate_design_matrix(x, deg)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scale data
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_scaler = StandardScaler(with_std=False)
    y_scaler.fit(y_train)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # Fit model
    # model = pipe.make_pipeline(PolynomialFeatures(deg=deg), LinearRegression())
    # model.fit(X_train_scaled, y_train_scaled)
    # y_pred = model.predict(X_test_scaled)
    y_pred = np.empty((y_test_scaled.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        X_, y_ = resample(X_train_scaled, y_train_scaled)
        y_pred[:, i] = least_squares_method(X_test_scaled, y_test_scaled).ravel()
    print(y_pred.shape)
    # Reintroduce the intercept
    y_intercept = y_scaler.mean_

    # Sort arrays so that they become line-plottable
    X_test_plot, y_pred_plot = zip(*sorted(zip(X_test_scaled[:, 1], least_squares_method(X_test_scaled, y_test_scaled))))

    # MSE, variance and bias
    # MSE = mean_squared_error(y_test_scaled, y_pred)
    MSE = np.mean(np.mean((y_test_scaled - y_pred)**2, axis=1, keepdims=True))
    # bias = np.mean(get_bias(y_test_scaled, y_pred))
    bias = np.mean((y_test_scaled - np.mean(y_pred, axis=1, keepdims=True))**2)
    var = np.mean(np.var(y_pred, axis=1, keepdims=True))
    mse_list[deg] = MSE
    var_list[deg] = var
    bias_list[deg] = bias
    print('Polynomial degree:', deg)
    print('Error:', mse_list[deg])
    print('Bias^2:', bias_list[deg])
    print('Var:', var_list[deg])
    print('{} >= {} + {} = {}'.format(mse_list[deg], bias_list[deg], var_list[deg], bias_list[deg]+var_list[deg]))


# Plot main plot
fig, ax = plt.subplots()
ax.scatter(x, y, label='Data', color='r', marker='x')
# ax.scatter(X_train_scaled[:, 1], y_train_scaled, label='Train data', color='g', marker='o')
ax.plot(X_test_plot, y_pred_plot+y_intercept, 'b--', label='Prediction')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend(loc='best')
plt.show()

# Plot metrics
fig, ax = plt.subplots()
ax.plot(poly_degs, mse_list, label='MSE')
ax.plot(poly_degs, var_list, label='Variance')
ax.plot(poly_degs, bias_list, label=r'Bias$^2$')
ax.set_xlabel(r'Polynomial deg')
plt.legend(loc='best')
plt.show()
