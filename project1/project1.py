import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

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
order = np.arange(1, 6)
deg = 4
test_size = 0.2
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
noise = np.random.normal(0, 0.05, (n,n))
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y) + noise
X = create_X(x, y, deg)

# Split data into training and test data
X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=test_size)
# Scale data using sklearn
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Estimate and evaluate model
model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
model.fit(X_train_scaled, z_train)
z_fit_OLS = model.predict(X_train_scaled)
z_pred_OLS = model.predict(X_test_scaled)
# z_fit_OLS = X_train_scaled @ beta_OLS(X_train_scaled, z_train)
# z_pred_OLS = X_test_scaled @ beta_OLS(X_test_scaled, z_test)
mse_OLS_train = mean_squared_error(z_train, z_fit_OLS)
mse_OLS_test = mean_squared_error(z_test, z_pred_OLS)
# MSE_train_arr[deg-1] = mse_OLS_train
# MSE_test_arr[deg-1] = mse_OLS_test
r2_OLS = r2_score(z_test, z_pred_OLS)
print(f'Test MSE OLS = {mse_OLS_test:.4f}')
print(f'Test R2 OLS = {r2_OLS:.4f}')

# Create train and test surfaces
x = X_test_scaled[:, 1]
y = X_test_scaled[:, 2]
x_mesh = np.linspace(x.min(), x.max(), 20)
y_mesh = np.linspace(y.min(), y.max(), 20)
x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
zz_test = np.zeros_like(x_mesh)
xy_mesh = np.vstack((x_mesh.ravel(), y_mesh.ravel())).T
zz_test.ravel()[:] = model.predict(xy_mesh)
# plt.plot(order, MSE_train_arr, 'b', label='Train')
# plt.plot(order, MSE_test_arr, 'r', label='Test')
# plt.title('MSE for OLS')
# plt.xlabel('Polynomial order')
# plt.ylabel('MSE')
# plt.legend(loc='best')
# plt.show()

# Plot the surface.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, c='k', marker='o', s=2, label='Franke function')
# ax.scatter(xx_train, yy_train, z_fit_OLS[:, :mesh_len_train], c='r', marker='o', s=2, label='OLS fit')
# ax.scatter(xx_test, yy_test, z_pred_OLS[:, :mesh_len_test], c='g', marker='o', s=2, label='OLS prediction')
# surf_train = ax.plot_surface(xx_train, yy_train, z_fit_OLS[:, :mesh_len_train], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False, label='OLS fit', alpha=0.5)
surf_test = ax.plot_surface(xx_test, yy_test, zz_test, cmap=cm.plasma,
					   linewidth=0, antialiased=False, label='OLS prediction', alpha=0.5)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf_train, shrink=0.5, aspect=5)
fig.colorbar(surf_test, shrink=0.5, aspect=5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# plt.legend(loc='best')
plt.show()