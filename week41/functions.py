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

def cost_func(X, y, beta, lmbda = 0):
    # value of lambda defines if we are using OLS or Ridge
    return np.mean((y - X @ beta)**2) + lmbda * np.sqrt(np.sum(beta**2))**2

def FrankeFunction(x, y):    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def learning_schedule_decay(t, t0, t1):
    return t0/(t1 + t)

def algo_plainGD(X, y, n, theta, n_iter, lmbda = 0, tol = 1e-8, eta = 0.01, momentum = 0):
    change = 0
    for i in range(n_iter):
        gradient = (1/n) * X.T @ (X @ theta - y) + 2 * lmbda * theta
        new_change = eta * gradient + momentum * change
        theta -= new_change
        change = new_change
        if np.linalg.norm(theta) <= tol:  
            return theta
    return theta

def algo_plainSGD(X, y, n, theta, M, m, lmbda = 0, tol = 1e-8, eta = 0.01, momentum = 0, learning_schedule = (None, None, None), t = 0):
    t, t0, t1 = learning_schedule
    change = 0
    for i in range(m):
        k = M*np.random.randint(m)    #k'th mini-batch
        X_k = X[k:k+M]
        y_k = y[k:k+M]
        gradient = (1/M) * X_k.T @ (X_k @ theta - y_k) + 2 * lmbda * theta
        eta = learning_schedule_decay(t, t0, t1)
        new_change = eta * gradient + momentum * change
        theta -= new_change
        change = new_change

    return theta

def algo_adam():
    pass

def algo_RMSprop():
    pass

def algo_adagrad(X, y, n, theta, eta = 0.01, d = 1e-7, momentum = 0):
    gradient = (1/n) * X.T @ (X @ theta - y)
    r += gradient**2
    update = eta * gradient / (d + np.sqrt(r))
    theta -= update
    return theta

def GradientDescent(X, y, n_iter = 1000, lmbda = 0, tol = 1e-8, momentum = 0, eta: float|None = None):
    n = len(y)
    theta = np.random.randn(X.shape[1], 1)

    if eta is None:
        H = (2/n) * X.T @ X + 2 * lmbda * np.eye(X.shape[1])
        eigenvalues, _ = np.linalg.eig(H)
        eta = 1/eigenvalues.max()

    theta = algo_plainGD(X, y, n, theta, eta=eta, n_iter=n_iter, momentum=momentum)
    return theta

def StochasticGradientDescent(X, y, M, n_epochs = 10, lmbda = 0, tol = 1e-8, algo = 'plain', time_decay: tuple = (None, None), momentum = 0):
    n = len(y)
    m = int(n/M)    # mini-batches
    t0, t1 = time_decay

    H = (2/n) * X.T @ X + 2 * lmbda * np.eye(X.shape[1])

    eigenvalues, _ = np.linalg.eig(H)
    theta = np.random.randn(X.shape[1], 1)
    eta = 1/eigenvalues.max()
    

    for epoch in range(n_epochs):
        match algo:
            case 'plain':
                theta = algo_plainSGD(X, y, m, theta, M, m, lmbda = lmbda, eta = eta, tol=tol, momentum=momentum, learning_schedule=())
            
            case 'adagrad':
                theta = algo_adagrad(X, y, n, theta, eta = 0.01, d = 1e-8)


            # t = epoch * m + i
            # if t0 is not None and t1 is not None:
            #     eta = learning_schedule_decay(t, t0, t1)
            # new_change = eta * gradient + momentum * change
            # theta -= new_change
            # change = new_change

        # if np.linalg.norm((theta + change) - theta) <= tol:  
        #     return theta

    return theta