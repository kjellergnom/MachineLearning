import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable

class NeuralNet:
    def __init__(self,
                 layers: tuple,
                 cost_func: Callable,
                 activation_func: Callable,
                 output_func: Callable = lambda x: x,
                 n_batches: int = 100,
                 jax_seed = 0) -> None:
        
        self.layers = layers
        self.cost_func = cost_func
        self.activation_func = activation_func
        self.n_batches = n_batches
        self.weights = []
        self.biases = []
        self.activations = []

        # Xavier Glorot weight initialization
        key = jax.random.PRNGKey(jax_seed)
        for (_in, _out) in zip(layers[:-1], layers[1:]):
            normalized_init_unit = jnp.sqrt(6.0 / (_in + _out))
            key, Wkey = jax.random.split(key)
            W = jax.random.uniform(
                Wkey, 
                (_in, _out), 
                minval=-normalized_init_unit, 
                maxval=+normalized_init_unit
            )
            bias = jnp.zeros(_out) + 0.01
            
            self.weights.append(W)
            self.biases.append(bias)
            self.activations.append(activation_func)
        
        self.activations[-1] = output_func

    def __str__(self) -> str:
        return f"This neural network contains {self.layers[0]} input neurons, {self.layers[0:-1]} hidden layers  and {self.layers[-1]} output neuron(s)."
    
    def feed_forward(self, X: np.ndarray, weights: list, biases: list, activations: list) -> np.ndarray:

        a = X
        for W, b, f in zip(weights, biases, activations):
            print(a.shape, W.shape)
            a = f(jnp.matmul(a, W) + b)

        return a

