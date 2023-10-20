from functions2  import *

MaxPolynomial = 12

x, y, z = generate_data(FrankesFunction = True)
#X = create_X(x, y, MaxPolynomial)
#X_train, X_test, z_train, z_test  = train_test_split(X, z, test_size = 0.2, random_state = 42)

x_train, x_test, y_train, y_test , z_train, z_test  = train_test_split(x, y,z ,test_size=0.2 ,random_state=42)

z_train = z_train.reshape(-1,1)
z_test = z_test.reshape(-1,1)
X_train = create_X(x_train, y_train, MaxPolynomial)
X_test = create_X(x_test, y_test, MaxPolynomial)

mse_test  = np.zeros((MaxPolynomial, 10))
mse_train = np.zeros((MaxPolynomial, 10))
polynomials = np.zeros(MaxPolynomial)
N = 100

for i in range(MaxPolynomial):
    degree = i+1

    c = int((degree+2)*(degree+1)/2)
    X_train = X_train[:,0:c]
    X_test = X_test[:,0:c]

    #X_train = create_X(x_train, y_train, degree)
    #X_test = create_X(x_test, y_test, degree)

    #Fitting the model and predicting NON PUOI MODIFICARE LAMBDA DA QUA
    z_tilde, z_pred , beta = GradientDescent(X_train, X_test , z_train , len(z_train) , N , lmb = 0.001 , momentum = False, Ridge =True)
    z_tilde_sto, z_pred_sto , beta_sto = StochasticGD(X_train, X_test , z_train , len(z_train) , n_epochs=100 , size_batch= 5, Niterations=1000 , delta_momentum=0.3 ,lmb= 0.001 , learningschedule = True , Ridge =True)
    z_tilde_mom, z_pred_mom , beta_mom = StochasticGD(X_train, X_test , z_train , len(z_train) , n_epochs=100 , size_batch= 5, Niterations=1000 ,delta_momentum=0.3 ,lmb=0.001,  momentum = True, Ridge=True)
    z_tilde_ada , z_pred_ada , beta_ada = AdaGradGD(X_train, X_test , z_train , len(z_train) , n_epochs=100 , size_batch= 5, Niterations=1000 ,delta_momentum=0.3,delta  = 1e-8,lmb=0.001, learningschedule = True , momentum = True, stochastic= True, Ridge=True)
    z_tilde_rms , z_pred_rms , beta_rms = RMSpropGD(X_train, X_test , z_train , len(z_train) , n_epochs=100 , size_batch= 5, Niterations=1000 ,delta_momentum=0.3,delta  = 1e-8 ,rho =0.99, lmb = 0.001, learningschedule = True , momentum = True, stochastic= True , Ridge = True)
    z_tilde_adam , z_pred_adam, beta_adam = ADAM(X_train, X_test , z_train , len(z_train) , n_epochs=100 , size_batch= 5, Niterations=1000 ,delta_momentum=0.3,delta  = 1e-8 ,rho =0.99 , beta1 = 0.9 , beta2 = 0.999 , lmb = 0.001, learningschedule = True , momentum = True, stochastic= True , Ridge = True)
 
    # Calculating the mean squared error

    mse_test[i ,0]  = np.mean(np.mean((z_test - z_pred)**2, axis=1, keepdims=True))
    mse_train[i, 0]  = np.mean(np.mean((z_train- z_tilde)**2, axis=1, keepdims=True))
    mse_test[i, 1]  = np.mean(np.mean((z_test - z_pred_sto)**2, axis=1, keepdims=True))
    mse_train[i ,1]  = np.mean(np.mean((z_train- z_tilde_sto)**2, axis=1, keepdims=True))
    mse_test[i, 2]  = np.mean(np.mean((z_test - z_pred_mom)**2, axis=1, keepdims=True))
    mse_train[i, 2]  = np.mean(np.mean((z_train- z_tilde_mom)**2, axis=1, keepdims=True))
    mse_test[i, 3]  = np.mean(np.mean((z_test - z_pred_ada)**2, axis=1, keepdims=True))
    mse_train[i, 3]  = np.mean(np.mean((z_train- z_tilde_ada)**2, axis=1, keepdims=True))
    mse_test[i, 4]  = np.mean(np.mean((z_test - z_pred_rms)**2, axis=1, keepdims=True))
    mse_train[i, 4]  = np.mean(np.mean((z_train- z_tilde_rms)**2, axis=1, keepdims=True))
    mse_test[i, 5]  = np.mean(np.mean((z_test - z_pred_adam)**2, axis=1, keepdims=True))
    mse_train[i, 5]  = np.mean(np.mean((z_train- z_tilde_adam)**2, axis=1, keepdims=True))

    polynomials[i] = degree

#Plotting the MSE and R2 scores
plt.title("MSE with Gradient Descent")
plt.plot(polynomials, mse_test[:,0], label="MSE test set")
plt.plot(polynomials, mse_train[:,0], label="MSE train set")
plt.xlabel("Polynomial order")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.show()

plt.title("MSE with Stochastic Gradient Descent ")
plt.plot(polynomials, mse_test[:,1], label="MSE test set")
plt.plot(polynomials, mse_train[:,1], label="MSE train set")
plt.xlabel("Polynomial order")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.show()

plt.title("MSE with Stochastic Gradient Descent with momentum")
plt.plot(polynomials, mse_test[:,2], label="MSE test set")
plt.plot(polynomials, mse_train[:,2], label="MSE train set")
plt.xlabel("Polynomial order")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.show()

plt.title("MSE with Stochastic Gradient Descent with adagrad")
plt.plot(polynomials, mse_test[:,3], label="MSE test set")
plt.plot(polynomials, mse_train[:,3], label="MSE train set")
plt.xlabel("Polynomial order")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.show()

plt.title("MSE with Stochastic Gradient Descent with RMSprop")
plt.plot(polynomials, mse_test[:,4], label="MSE test set")
plt.plot(polynomials, mse_train[:,4], label="MSE train set")
plt.xlabel("Polynomial order")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.show()

plt.title("MSE with Stochastic Gradient Descent with ADAM")
plt.plot(polynomials, mse_test[:,5], label="MSE test set")
plt.plot(polynomials, mse_train[:,5], label="MSE train set")
plt.xlabel("Polynomial order")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.show()