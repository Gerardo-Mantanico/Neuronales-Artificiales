import numpy as np

def initialize_weights(seed=42):
    # Inicializa los pesos con semilla aleatoria fija para reproducibilidad
    np.random.seed(seed)

    # Pesos capa 1 (entrada -> oculta) usando He initialization
    escala_W1 = np.sqrt(2.0 / 784)
    W1 = np.random.randn(64, 784) * escala_W1
    b1 = np.zeros((64,))

    # Pesos capa 2 (oculta -> salida) usando Xavier initialization
    escala_W2 = np.sqrt(1.0 / 64)
    W2 = np.random.randn(10, 64) * escala_W2
    b2 = np.zeros((10,))

    return W1, b1, W2, b2
