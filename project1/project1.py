import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def generate_design_matrix(x, n_features, include_bias=True):
    if include_bias == True:
        X = np.ones(shape=(n, n_features+1))
        for feature in range(n_features):
            X[:, feature+1] = x[:, 0]**(feature+1)
    else:
        X = np.zeros(shape=(n, n_features))
        for feature in range(n_features):
            X[:, feature] = x[:, 0]**(feature + 1)
    return X

def design_matrix(x, y, n_features):
    X = np.zeros(shape=(n, 1))
    for feature in range(n_features+1):
        for deg in range(feature//2+1):
            if (feature == 0 and deg == 0):
                continue
            print(f'x^{feature - deg}, y^{deg}')
            if (feature - deg) != deg:
                print(f'x^{deg}, y^{feature - deg}')
        print('\n')

def beta_OLS(X, y):
    beta =  np.linalg.pinv(X.T @ X) @ X.T @ y
    return beta

def beta_Ridge(X, y, lmbda):
    beta = np.linalg.pinv(X.T @ X + lmbda*np.identity(X.shape[1])) @ X.T @ y
    return beta

# Make data.
n = 50
order = 3
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
print(np.shape([x, y]))
x, y = np.meshgrid(x,y)
noise = np.random.normal(0, 0.02, x.shape)
z = FrankeFunction(x, y) + noise
print(np.shape([x, y]))
# Estimate
model = make_pipeline(
    PolynomialFeatures(degree=order), 
    LinearRegression(fit_intercept=False))
model.fit((x, y), z)
z_model = model.predict(np.c_[x.flatten(), y.flatten()]).reshape(x.shape)

# Plot the surface.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#### KLADD ####
