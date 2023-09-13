import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

### FrankeFunction() and create_X() are taken from the project 1 description
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)
	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))
	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)
	return X

def beta_OLS(X, y):
    beta =  np.linalg.pinv(X.T @ X) @ X.T @ y
    return beta

def beta_Ridge(X, y, lmbda):
    beta = np.linalg.pinv(X.T @ X + lmbda*np.identity(X.shape[1])) @ X.T @ y
    return beta

# Make data.
np.random.seed()
n = 50
order = 2
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
X = create_X(x, y, order)
noise = np.random.normal(0, 0.05, (n,n))
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y) + noise

# Split data into training and test data
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Estimate and evaluate model using sklearn
reg = LinearRegression().fit(X_train_scaled, z_train)
z_tilde_OLS = reg.predict(X_train_scaled)
z_predict_OLS = reg.predict(X_test_scaled)
betaOLS = reg.coef_

# Estimate and evaluate model
# betaOLS = beta_OLS(X_train_scaled, z_train)
# z_tilde_OLS = X_train_scaled @ betaOLS
# z_predict_OLS = X_test_scaled @ betaOLS
MSE_OLS_train = mean_squared_error(z_train, z_tilde_OLS)
MSE_OLS_test = mean_squared_error(z_test, z_predict_OLS)
R2_OLS_train = r2_score(z_train, z_tilde_OLS)
R2_OLS_test = r2_score(z_test, z_predict_OLS)

# Create train and test surfaces
x_surface_train = X_train_scaled[:, 1]
x_surface_test = X_test_scaled[:, 1]
y_surface_train = X_train_scaled[:, 2]
y_surface_test = X_test_scaled[:, 2]
xx_train, yy_train = np.meshgrid(x_surface_train, y_surface_train)
xx_test, yy_test = np.meshgrid(x_surface_test, y_surface_test)

print(np.shape(X))
print(np.shape(X_train_scaled))
print(np.shape(X_test_scaled))
print(np.shape(z_train))
print(np.shape(z_test))
print(np.shape(z_tilde_OLS))
print(np.shape(betaOLS))

# Plot the surface.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, c='r', marker='o', s=1)
surf = ax.plot_surface(xx_train, yy_train, z_tilde_OLS, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()