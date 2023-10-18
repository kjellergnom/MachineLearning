import numpy as np
import matplotlib.pyplot as plt

def generate_1D_design_matrix(x, poly_deg, intercept=True):

    n = len(x)
    x = x.reshape(-1, 1)

    X = np.ones(shape=(n, poly_deg+1))
    for p in range(poly_deg):
        X[:, p+1] = x[:, 0]**(p+1)
    
    if intercept == False:
        return np.delete(X, 0, axis=1)
    return X

def generate_2D_design_matrix(x, y, poly_deg, intercept=True):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    n = len(x)
    l = int((poly_deg+1)*(poly_deg+2)/2)

    X = np.ones((n,l))

    for i in range(1,poly_deg+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    if intercept == False:
        X = X[:, 1:]

    return X

def get_beta_OLS(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def get_beta_Ridge(X, y, lmbda):
    return np.linalg.pinv(X.T @ X + lmbda * np.eye(X.shape[1])) @ X.T @ y

def FrankeFunction(x, y):    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def GradientDescent(X, y, n_iter = 1000, lmbda = 1e-2, tol = 1e-8, method = 'OLS', momentum = False):
    n = len(y)
    match method:
        case 'OLS':
            H = (2/n) * X.T @ X
        case 'Ridge':
            H = (2/n) * X.T @ X + 2 * lmbda * np.eye(X.shape[1])

    eigenvalues, _ = np.linalg.eig(H)
    beta = np.random.randn(X.shape[1], 1)
    eta = 1/eigenvalues.max()

    match momentum:
        case False:
            for iter in range(n_iter):
                beta_old = beta
                match method:
                    case 'OLS':
                        gradient = (2/n) * X.T @ (X @ beta - y)
                    case 'Ridge':
                        gradient = (2/n) * X.T @ (X @ beta - y) + 2 * lmbda * beta
                
                beta = beta_old - eta * gradient

                if np.linalg.norm(beta_old - beta) <= tol:  
                    return beta

        case True:
            change = 0
            for iter in range(n_iter):
                
                match method:   
                    case 'OLS':
                        gradient = (2/n) * X.T @ (X @ beta - y)
                    case 'Ridge':
                        gradient = (2/n) * X.T @ (X @ beta - y) + 2 * lmbda * beta
                
                new_change = eta * gradient + momentum * change
                beta -= new_change

                change = new_change

                if np.linalg.norm((beta + change) - beta) <= tol:  
                    return beta

    return beta
    
def StochasticGradientDescent():
    pass