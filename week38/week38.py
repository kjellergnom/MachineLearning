import numpy as np
import matplotlib.pyplot as plt
import sklearn.pipeline as pipe
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def generate_design_matrix(x, poly_deg):
    X = np.ones(shape=(n, poly_deg+1))
    for p in range(poly_deg):
        X[:, p+1] = x[:, 0]**(p+1)
    return X

def bias(x, y):
    return np.mean((x - np.mean(y))**2)

np.random.seed()
n = 200
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
poly_degs = [x for x in range(5, 16)]
X = generate_design_matrix(x, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

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
model = pipe.make_pipeline(PolynomialFeatures(degree=8), LinearRegression())
model.fit(X_train_scaled, y_train_scaled)
y_pred = model.predict(X_test_scaled)

# Reintroduce the intercept
y_intercept = y_scaler.mean_

# Sort arrays so that they become line-plottable
X_test_plot, y_pred_plot = zip(*sorted(zip(X_test_scaled[:, 1], y_pred)))

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
