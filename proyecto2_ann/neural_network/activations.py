import numpy as np


def relu(Z: np.ndarray) -> np.ndarray:
    """
    ReLU(z) = max(0, z)  aplicado elemento a elemento.
    Entrada: Z (n,)  |  Salida: A (n,)
    """
    return np.maximum(0.0, Z)


def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """
    ReLU'(z) = 1 si z > 0, 0 si z <= 0
    Retorna máscara binaria de la misma forma que Z.
    """
    return (Z > 0).astype(float)


def softmax(Z: np.ndarray) -> np.ndarray:
    """
    softmax(z_i) = exp(z_i - max(Z)) / sum(exp(z_j - max(Z)))
    Restar max(Z) garantiza estabilidad numérica sin cambiar el resultado.
    Entrada: Z (10,)  |  Salida: A (10,) con sum(A) == 1.0
    """
    Z_estable = Z - np.max(Z)
    exp_Z = np.exp(Z_estable)
    return exp_Z / np.sum(exp_Z)


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    sigmoid(z) = 1 / (1 + exp(-z))
    Alternativa a ReLU para la capa oculta.
    """
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
    """
    sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
    """
    s = sigmoid(Z)
    return s * (1.0 - s)
