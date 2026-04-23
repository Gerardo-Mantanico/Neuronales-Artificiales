import numpy as np


def one_hot(y: int, num_classes: int = 10) -> np.ndarray:
    """
    Convierte etiqueta entera en vector one-hot.
    Ejemplo: y=3, num_classes=10 → [0,0,0,1,0,0,0,0,0,0]
    """
    vector = np.zeros(num_classes)
    vector[y] = 1.0
    return vector


def cross_entropy_loss(A2: np.ndarray, y_onehot: np.ndarray) -> float:
    """
    L = -sum(y_k * log(A2_k))
    Clipping evita log(0) = -inf.
    Entrada: A2 (10,) probabilidades, y_onehot (10,)
    Salida:  escalar — pérdida de la muestra
    """
    A2_seguro = np.clip(A2, 1e-15, 1.0)
    return -float(np.sum(y_onehot * np.log(A2_seguro)))


def accuracy(A2: np.ndarray, y: int) -> int:
    """
    Retorna 1 si la clase predicha coincide con y, 0 en otro caso.
    """
    return int(np.argmax(A2) == y)
