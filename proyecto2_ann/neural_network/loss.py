import numpy as np

def one_hot(y, num_classes=10):
    # Convierte a representacion one-hot
    vector = np.zeros(num_classes)
    vector[y] = 1.0
    return vector

def cross_entropy_loss(A2, y_onehot):
    # Perdida de entropia cruzada con clipping para evitar log(0)
    A2_seguro = np.clip(A2, 1e-15, 1.0)
    return -float(np.sum(y_onehot * np.log(A2_seguro)))

def accuracy(A2, y):
    # Retorna 1 si predice correctamente, si no 0
    return int(np.argmax(A2) == y)
