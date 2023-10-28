import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet
from funcs import sigmoid, cost_cross_entropy
import jax
import jax.numpy as jnp


X = np.array([[0,0],
               [0,1],
               [1,0],
               [1,1]])

y_XOR = np.array([[0],[1],[1],[0]]) # XOR
y_AND = np.array([[0],[0],[0],[1]]) # AND
y_OR = np.array([[0],[1],[1],[1]]) # OR

N = 100
layers = (2, 2, 1)
n_epochs = 1000
eta = 0.1
seed = 0

my_NN = NeuralNet(layers, cost_func=cost_cross_entropy, activation_func=sigmoid, jax_seed=seed)
print(my_NN)

# Train the network
weights = my_NN.weights
biases = my_NN.biases
activations = my_NN.activations

trial = my_NN.feed_forward(X, weights, biases, activations)
# Print the outputs
print(f"XOR: {y_XOR}")
print(f"AND: {y_AND}")
print(f"OR: {y_OR}")
print(f"Trial: {trial}")

def cost_forward(X:jnp.ndarray, y_true: jnp.ndarray) -> float:        
    return cost_cross_entropy(X, y_true)

cost_forward(
    my_NN.feed_forward(X, weights, biases, activations), 
    y_XOR
)

cost_and_grad_func = jax.value_and_grad(
    lambda weights, biases: cost_forward(
        my_NN.feed_forward(
            X,
            weights,
            biases,
            activations
        ),
        y_XOR,
    ),
    argnums=(0, 1)
)

initial_cost, (initial_weight_gradients, initial_bias_gradients) = cost_and_grad_func(
    weights,
    biases
)

print(f"Initial cost: {initial_cost}")
