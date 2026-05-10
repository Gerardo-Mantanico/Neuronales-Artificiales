import numpy as np

def relu(Z):
    # ReLU(z) = max(0, z)
    return np.maximum(0.0, Z)

def relu_derivative(Z):
    # Derivada de ReLU: 1 si z > 0, 0 si z <= 0
    return (Z > 0).astype(float)

def softmax(Z):
    # Softmax estable restando el maximo
    Z_estable = Z - np.max(Z)
    exp_Z = np.exp(Z_estable)
    return exp_Z / np.sum(exp_Z)

def sigmoid(Z):
    # Sigmoide
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_derivative(Z):
    # Derivada de Sigmoide
    s = sigmoid(Z)
    return s * (1.0 - s)
