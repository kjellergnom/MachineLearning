import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_design_matrix(x, poly_deg, include_bias=True):
    if include_bias == True:
        X = np.ones(shape=(n, poly_deg+1))
        for p in range(poly_deg):
            X[:, p+1] = x[:, 0]**(p+1)
    else:
        X = np.zeros(shape=(n, poly_deg))
        for p in range(poly_deg):
            X[:, p] = x[:, 0]**(p+1)
    return X

def OLS(X, y):
    beta =  np.linalg.pinv(X.T @ X) @ X.T @ y
    y_tilde = X @ beta
    return y_tilde

def Ridge_regression(X, y, lmbda):
    beta = np.linalg.pinv(X.T @ X + lmbda*np.identity(X.shape[1])) @ X.T @ y
    y_tilde = X @ beta
    return y_tilde

np.random.seed()
n = 100
poly_degs = [5, 10, 15]
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
lmbda = [0.0001, 0.001, 0.01, 0.1, 1]

for degree in poly_degs:
    # Generate design matrix and split into train and test
    bias_bool = 0       # Includes/excludes bias in design matrix and chooses the corresponding column in X 
    X = generate_design_matrix(x, degree, include_bias=bias_bool)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Scaling X and y
    X_mean = np.mean(X_train, axis=0)
    X_train_scaled = X_train - X_mean
    X_test_scaled = X_test - X_mean
    y_mean = np.mean(y_train)
    y_train_scaled = y_train - y_mean
    y_test_scaled = y_test - y_mean

    #### OLS ####
    print(f'\nOLS:\nPoly deg: {degree}')
    y_tilde_OLS = OLS(X_train_scaled, y_train_scaled)
    y_predict_OLS = OLS(X_test_scaled, y_test_scaled)
    MSE_OLS_train = mean_squared_error(y_train_scaled, y_tilde_OLS)
    MSE_OLS_test = mean_squared_error(y_test_scaled, y_predict_OLS)
    print(f'MSE_OLS_train = {MSE_OLS_train:.4f}\nMSE_OLS_test = {MSE_OLS_test:.4f}')
    X_OLS_train_plot, y_tilde_OLS_plot = zip(*sorted(zip(X_train_scaled[:, bias_bool], y_tilde_OLS)))
    X_OLS_test_plot, y_predict_OLS_plot = zip(*sorted(zip(X_test_scaled[:, bias_bool], y_predict_OLS)))
    print(X_OLS_test_plot)
    plt.plot(X_OLS_train_plot, y_tilde_OLS_plot, 'g', label='Train; OLS')
    plt.plot(X_OLS_test_plot, y_predict_OLS_plot, 'g--', label='Predict', zorder=2)
    plt.legend(loc='best')
    plt.title(f'Poly deg: {degree}')
    plt.plot(x, y - np.mean(y), 'ro', label='Data')
    plt.show()

    #### Ridge ####
    print(f'\nRidge:\nPoly deg: {degree}')
    # Iterate over lambda
    MSE_Ridge_train_arr = np.zeros(len(lmbda))
    MSE_Ridge_test_arr = np.zeros(len(lmbda))
    for l in lmbda:
        y_tilde_Ridge = Ridge_regression(X_train_scaled, y_train_scaled, l)
        y_predict_Ridge = Ridge_regression(X_test_scaled, y_test_scaled, l)
        X_Ridge_train_plot, y_tilde_Ridge_plot = zip(*sorted(zip(X_train_scaled[:, bias_bool], y_tilde_Ridge)))
        X_Ridge_test_plot, y_predict_Ridge_plot = zip(*sorted(zip(X_test_scaled[:, bias_bool], y_predict_Ridge)))
        y_tilde_Ridge = Ridge_regression(X_train_scaled, y_train_scaled, l)
        y_predict_Ridge = Ridge_regression(X_test_scaled, y_test_scaled, l)
        MSE_Ridge_train = mean_squared_error(y_train_scaled, y_tilde_Ridge)
        MSE_Ridge_test = mean_squared_error(y_test_scaled, y_predict_Ridge)
        print(f'Lambda: {l}')
        print(f'MSE_OLS_train = {MSE_OLS_train:.4f}, MSE_OLS_test = {MSE_OLS_test:.4f}')
        print(f'MSE_Ridge_train = {MSE_Ridge_train:.4f}, MSE_Ridge_test = {MSE_Ridge_test:.4f}')
        MSE_Ridge_train_arr[lmbda.index(l)] = MSE_Ridge_train
        MSE_Ridge_test_arr[lmbda.index(l)] = MSE_Ridge_test
    
    plt.plot(lmbda, MSE_Ridge_train_arr, 'b', label=f'Train; Ridge', zorder=1)
    plt.plot(lmbda, MSE_Ridge_test_arr, 'b--', label=f'Test; Ridge', zorder=2)
        # plt.plot(X_Ridge_train_plot, y_tilde_Ridge_plot, 'b', label=f'Train; Ridge, lambda={l}', zorder=1, alpha=0.1*(lmbda.index(1)+1))
        # plt.plot(X_Ridge_test_plot, y_predict_Ridge_plot, 'b--', label=f'Predict, lambda={l}', zorder=2, alpha=0.1*(lmbda.index(1)+1))
    plt.xscale('log')
    plt.title(f'Poly deg: {degree}')
    plt.legend(loc='best')
    plt.show()