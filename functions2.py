from random import random, seed
import autograd.numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import linear_model
import seaborn as sns
import pandas as pd
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import normal, uniform
from sklearn.preprocessing import MinMaxScaler
import sys
from autograd import grad

def generate_data(noise=True, step_size=0.05 , FrankesFunction=True):
    # Arrange x and y
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)

    # Create meshgrid of x and y
    X, Y = np.meshgrid(x, y)
    
    if FrankesFunction:
        # Calculate the values for Franke function
        z = FrankeFunction(X, Y, noise=noise).flatten()
    else:
        z = TestFunction(X, Y, noise=noise).flatten()

    # Flatten x and y for plotting
    x = X.flatten()
    y = Y.flatten()
    
    return x, y, z


def TestFunction(x, y, noise=False):
    if noise: 
        random_noise = np.random.normal(0, 0.1 , x.shape)
    else: 
        random_noise = 0

    return  x**2 + y**2 + 2*x*y + random_noise

    


def FrankeFunction(x, y, noise=False):
    if noise: 
        random_noise = np.random.normal(0, 0.1 , x.shape)
    else: 
        random_noise = 0
    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + random_noise

def create_X(x, y, n, intercept=True):
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

def GradientDescent( X_train , X_test , z_train, n,  Niterations=1000 , delta_momentum = 0.3 , lmb = 0.001 ,momentum = True , Ridge = False):
    z_train = z_train.reshape(-1,1)
    if Ridge :
        H = (2.0/n)* X_train.T @ X_train + 2* lmb * np.identity(X_train.shape[1])
        def gradient(X_train , beta , lmb):
            return (2.0/n)* X_train.T @ (X_train @ beta-z_train) + 2* lmb * beta
    else:
        H = (2.0/n)* X_train.T @ X_train
        def gradient(X_train , beta , lmb):
            return (2.0/n)* X_train.T @ (X_train @ beta-z_train)

    change = 0.0

    #Get the eigenvalues

    EigValues, _ = np.linalg.eig(H)

    if momentum:
        def algorithm(eta, gradients , beta , change):
            new_change = eta*gradients + delta_momentum * change
            beta-= new_change
            change = new_change
            return beta
    else: 
        def algorithm(eta, gradients ,beta , change):
            beta -= np.abs(eta)*gradients
            return beta

    # Initialize beta and the other parameters
    beta = np.random.randn(X_train.shape[1],1)
    eta = 1.0/np.max(EigValues)
    
    
    for iter in range(Niterations):

        gradients = gradient(X_train , beta , lmb)
        beta = algorithm(eta, gradients , beta , change)     
  
    z_tilde = X_train @ beta
    z_predict = X_test @ beta

    z_tilde = z_tilde.reshape(-1,1)
    z_predict = z_predict.reshape(-1,1)

    return z_tilde , z_predict , beta

def StochasticGD( X_train , X_test , z_train , n , n_epochs=50 , size_batch= 5, Niterations=1000 , delta_momentum = 0.3 , lmb= 0.001 , learningschedule = True , momentum = True , Ridge = False ):
    z_train = z_train.reshape(-1,1)
    M =size_batch
    m = int(n/M )#number of minibatches
    t0, t1 = 5, 50
    
    if Ridge :
        H = (2.0/n)* X_train.T @ X_train + 2* lmb * np.identity(X_train.shape[1])
        def gradient(X_train , z_train , beta , lmb):
            return (2.0/n)* X_train.T @ (X_train @ beta-z_train) + 2* lmb * beta
    else:
        H = (2.0/n)* X_train.T @ X_train
        def gradient(X_train , z_train, beta , lmb):
            return (2.0/n)* X_train.T @ (X_train @ beta-z_train)
    
    EigValues, EigVectors = np.linalg.eig(H)
    change = 0

    if learningschedule:
        def learning_schedule(t):
            return t0/(t+t1)
    else:
        def learning_schedule(t):
            return 1.0/np.max(EigValues)
        
    if momentum:
        
        def algorithm(eta, gradients , beta , change):
            new_change = eta*gradients + delta_momentum * change
            beta-= new_change
            change = new_change
            return beta
    else: 
        def algorithm(eta, gradients ,beta , change):
            beta -= eta*gradients
            return beta
            

    beta = np.random.randn(X_train.shape[1],1)
    z_train = z_train.reshape(-1,1)

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M + np.random.randint(m)
            xi = X_train[random_index:random_index+M]
            zi = z_train[random_index:random_index+M]
            gradients = gradient(xi ,zi,  beta , lmb)
            eta = learning_schedule(epoch*m+i)
            beta =algorithm(eta, gradients , beta , change)

    z_tilde = X_train @ beta
    z_predict = X_test @ beta

    z_tilde = z_tilde.reshape(-1,1)
    z_predict = z_predict.reshape(-1,1)

    return z_tilde , z_predict , beta

def CostOLS(y,X,beta):
    return np.sum((y-X @ beta)**2)

def CostRidge(y,X,beta,lmb=0.001):
    return np.sum((y-X @ beta)**2) + lmb*np.sum(beta**2)


def AdaGradGD(X_train , X_test , z_train , n ,  n_epochs=50 , size_batch= 5, Niterations=1000 , delta_momentum = 0.3 ,delta  = 1e-8 ,lmb=0.001, learningschedule = True , momentum = True , stochastic = True , Ridge = False):
    z_train = z_train.reshape(-1,1)
    M =size_batch
    m = int(n/M )#number of minibatches
    t0, t1 = 5, 50
    Giter = 0.0
    
    H = (2.0/n)* X_train.T @ X_train
    EigValues, EigVectors = np.linalg.eig(H)
    change = 0

    if learningschedule:
        def learning_schedule(t):
            return t0/(t+t1)
    else:
        def learning_schedule(t):
            return np.real(1.0/np.max(EigValues))
        
    if momentum:
        
        def algorithm(eta, gradients , beta , change , Giter):
            Giter += gradients*gradients
            update = gradients*eta/(delta+np.sqrt(Giter))
            new_change = update + delta_momentum * change
            beta-= new_change
            change = new_change
            return beta
    else: 
        def algorithm(eta, gradients ,beta , change , Giter):
            Giter += gradients*gradients
            update = gradients*eta/(delta+np.sqrt(Giter))
            beta -= update
            return beta
        
    if Ridge: 
        training_gradient = grad(CostRidge,2)
    else:
        training_gradient = grad(CostOLS,2)

    beta = np.random.randn(X_train.shape[1],1)
    z_train = z_train.reshape(-1,1)

    
    if stochastic:
        for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                random_index = M + np.random.randint(m)
                xi = X_train[random_index:random_index+M]
                zi = z_train[random_index:random_index+M]
                eta = learning_schedule(epoch*m+i)
                gradients = (1.0/M)*training_gradient(zi, xi, beta)
                beta =algorithm(eta, gradients , beta , change , Giter)
    else:
        for iter in range(Niterations):

            eta = np.real(1.0/np.max(EigValues))
            gradients = (1.0/M)*training_gradient(z_train ,X_train, beta)
            beta =algorithm(eta, gradients , beta , change  , Giter)
            

    z_tilde = X_train @ beta
    z_predict = X_test @ beta

    z_tilde = z_tilde.reshape(-1,1)
    z_predict = z_predict.reshape(-1,1)

    return z_tilde , z_predict , beta

def RMSpropGD(X_train , X_test , z_train , n ,  n_epochs=50 , size_batch= 5, Niterations=1000 , delta_momentum = 0.3 ,delta  = 1e-8 ,rho =0.99,lmb=0.001 ,learningschedule = True , momentum = True , stochastic = True, Ridge = False):
    z_train = z_train.reshape(-1,1) 
    M =size_batch
    m = int(n/M )#number of minibatches
    t0, t1 = 5, 50
    Giter = 0.0
    
    H = (2.0/n)* X_train.T @ X_train
    EigValues, EigVectors = np.linalg.eig(H)
    change = 0

    if learningschedule:
        def learning_schedule(t):
            return t0/(t+t1)
    else:
        def learning_schedule(t):
            return np.real(1.0/np.max(EigValues))
        
    if momentum:
        
        def algorithm(eta, gradients , beta , change , Giter):
            Giter = (rho*Giter+(1-rho)*gradients*gradients)
            update = gradients*eta/(delta+np.sqrt(Giter))
            new_change = update + delta_momentum * change
            beta-= new_change
            change = new_change
            return beta
    else: 
        def algorithm(eta, gradients ,beta , change , Giter):
            Giter = (rho*Giter+(1-rho)*gradients*gradients)
            update = gradients*eta/(delta+np.sqrt(Giter))
            beta -= update
            return beta
            
    if Ridge: 
        training_gradient = grad(CostRidge,2)
    else:
        training_gradient = grad(CostOLS,2)

    beta = np.random.randn(X_train.shape[1],1)
    z_train = z_train.reshape(-1,1)

    
    if stochastic:
        for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                random_index = M + np.random.randint(m)
                xi = X_train[random_index:random_index+M]
                zi = z_train[random_index:random_index+M]
                eta = learning_schedule(epoch*m+i)
                gradients = (1.0/M)*training_gradient(zi, xi, beta)
                beta =algorithm(eta, gradients , beta , change , Giter)
    else:
        for iter in range(Niterations):

            eta = np.real(1.0/np.max(EigValues))
            gradients = (1.0/M)*training_gradient(z_train, X_train, beta)
            beta =algorithm(eta, gradients , beta , change  , Giter)
            

    z_tilde = X_train @ beta
    z_predict = X_test @ beta

    z_tilde = z_tilde.reshape(-1,1)
    z_predict = z_predict.reshape(-1,1)

    return z_tilde , z_predict , beta

def ADAM(X_train , X_test , z_train , n ,  n_epochs=50 , size_batch= 5, Niterations=1000 , delta_momentum = 0.3 ,delta  = 1e-8 ,rho =0.99, beta1 = 0.9 , beta2 = 0.999 ,lmb=0.001, learningschedule = True , momentum = True , stochastic = True, Ridge =False):
    M =size_batch
    z_train = z_train.reshape(-1,1) 
    m = int(n/M )#number of minibatches
    t0, t1 = 5, 50
    first_moment = 0.0
    second_moment = 0.0
    iter= 0.0
    
    H = (2.0/n)* X_train.T @ X_train
    EigValues, EigVectors = np.linalg.eig(H)
    change = 0

    if learningschedule:
        def learning_schedule(t):
            return t0/(t+t1)
    else:
        def learning_schedule(t):
            return np.real(1.0/np.max(EigValues))
        
    if momentum:
        
        def algorithm(eta, gradients , beta , change , first_moment , second_moment , iter):
            first_moment = beta1*first_moment + (1-beta1)*gradients
            second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
            first_term = first_moment/(1.0-beta1**iter)
            second_term = second_moment/(1.0-beta2**iter)
            # Scaling with rho the new and the previous results
            update = eta*first_term/(np.sqrt(second_term)+delta)
            new_change = update + delta_momentum * change
            beta-= new_change
            change = new_change
            return beta
    else: 
        def algorithm(eta, gradients ,beta , change , first_moment , second_moment , iter):
            # Computing moments first
            first_moment = beta1*first_moment + (1-beta1)*gradients
            second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
            first_term = first_moment/(1.0-beta1**iter)
            second_term = second_moment/(1.0-beta2**iter)
            # Scaling with rho the new and the previous results
            update = eta*first_term/(np.sqrt(second_term)+delta)
            beta -= update
            return beta
            
    if Ridge: 
        training_gradient = grad(CostRidge,2)
    else:
        training_gradient = grad(CostOLS,2)
    
    beta = np.random.randn(X_train.shape[1],1)
    z_train = z_train.reshape(-1,1)

    
    if stochastic:
        for epoch in range(n_epochs):
            first_moment = 0.0
            second_moment = 0.0
            iter += 1
            for i in range(m):
                random_index = M + np.random.randint(m)
                xi = X_train[random_index:random_index+M]
                zi = z_train[random_index:random_index+M]
                eta = learning_schedule(epoch*m+i)
                gradients = (1.0/M)*training_gradient(zi, xi, beta)
                beta =algorithm(eta, gradients , beta , change , first_moment , second_moment , iter)
    else:
        for iter in range(Niterations):

            eta = np.real(1.0/np.max(EigValues))
            gradients = (1.0/M)*training_gradient(z_train, X_train, beta)
            beta =algorithm(eta, gradients , beta , change  , first_moment , second_moment , iter)
            

    z_tilde = X_train @ beta
    z_predict = X_test @ beta

    z_tilde = z_tilde.reshape(-1,1)
    z_predict = z_predict.reshape(-1,1)

    return z_tilde , z_predict , beta

    

    
    