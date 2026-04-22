import numpy as np


def initialize_weights(seed: int = 42) -> tuple:
    """
    Inicializa pesos y sesgos de la red 784 → 64 → 10.

    W1: He initialization  (recomendado para ReLU)
    W2: Xavier initialization (recomendado para Softmax)
    Sesgos: ceros

    Retorna: (W1, b1, W2, b2)
    """
    np.random.seed(seed)

    # Capa 1: entrada (784) → oculta (64)
    escala_W1 = np.sqrt(2.0 / 784)
    W1 = np.random.randn(64, 784) * escala_W1
    b1 = np.zeros((64,))

    # Capa 2: oculta (64) → salida (10)
    escala_W2 = np.sqrt(1.0 / 64)
    W2 = np.random.randn(10, 64) * escala_W2
    b2 = np.zeros((10,))

    return W1, b1, W2, b2
