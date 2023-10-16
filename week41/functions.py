import numpy as np
import matplotlib.pyplot as plt

def test_func_1D(x: float | np.ndarray, coeffs: tuple | list = (1, 1, 1)) -> float | np.ndarray:
    """
    Function to approximate
    """
    a0, a1, a2 = coeffs
    return a2*x**2 + a1*x + a0

def test_func_2D(x: float | np.ndarray, y: float | np.ndarray, coeffs: tuple | list = (1, 1, 1, 0)) -> float | np.ndarray:
    """
    Function to approximate
    """
    a2, a1, a0, b = coeffs
    return a2*x**2 + a1*x*y + a0*y**2 + b

def create_2D_design_matrix(x, y, n, intercept=True):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)

    if intercept:
        X = np.ones((N,l))

        for i in range(1,n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)

    else:
        X = np.ones((N,l-1))

        for i in range(1,n+1):
            q = int((i)*(i+1)/2) - 1
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)

    return X

def create_1D_design_matrix(x, n, poly_deg, include_intercept = True):
    X = np.ones(shape=(n, poly_deg+1))
    for p in range(poly_deg):
        X[:, p+1] = x[:, 0]**(p+1)
    if not include_intercept:
        return X[:, 1:]
    return X

def gradient_descent(beta, X, y, gamma, n_iter = 1000, tol = 1e-3, Ridge = False, lmb = 0.001):
    for i in range(n_iter):
        if Ridge:
            gradient = 2/n * X.T @ (X @ beta - y) + 2*lmb*beta
        else:
            gradient = 2/n * X.T @ (X @ beta - y)
        beta -= gamma * gradient
        if np.linalg.norm(gradient) < tol:
            break
    
    return beta        

if __name__ == '__main__':
    n = 10
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    X = create_1D_design_matrix(x, n, 3, include_intercept=False)
    print(X)