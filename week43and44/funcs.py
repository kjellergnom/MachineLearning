import numpy as np
from autograd import elementwise_grad
import jax
import jax.numpy as jnp

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def softmax(X):
    X = X - jnp.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return jnp.exp(X) / (jnp.sum(jnp.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X):
    return jnp.where(X > jnp.zeros(X.shape), X, jnp.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return jnp.where(X > jnp.zeros(X.shape), X, delta * X)


def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return jnp.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return jnp.where(X > 0, 1, delta)

        return func

    else:
        return elementwise_grad(func)

def cost_cross_entropy(X, target: jnp.ndarray):
    return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))