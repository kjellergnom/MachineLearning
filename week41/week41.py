from functions import *

n = 100
noise = np.random.normal(0, 0.1)
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = test_func_1D(x, coeffs=(1, 1, 1)) #+ noise

#Plot function
plt.plot(x, y, label="Function to approximate")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()