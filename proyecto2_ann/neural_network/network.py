import numpy as np
from .initializer import initialize_weights
from .activations import relu, relu_derivative, softmax
from .loss import one_hot, cross_entropy_loss, accuracy


class MLP:
    """
    Perceptrón Multicapa: 784 → 64 → 10
    Implementado completamente desde cero con NumPy.
    """

    def __init__(self, lr: float = 0.01, seed: int = 42) -> None:
        self.lr = lr
        self.W1, self.b1, self.W2, self.b2 = initialize_weights(seed)

        # Historial para graficar
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

        # Contador global de iteraciones (muestras procesadas)
        self.iteration: int = 0

    # ──────────────────────────────────────────────────────────
    # Forward Propagation
    # ──────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> tuple:
        """
        Propaga X a través de la red.

        Parámetros:
          X: (784,) — vector de imagen normalizado [0, 1]

        Retorna:
          Z1: (64,)  pre-activación capa oculta
          A1: (64,)  activación capa oculta  (ReLU)
          Z2: (10,)  pre-activación capa salida
          A2: (10,)  probabilidades finales   (Softmax)
        """
        # Capa oculta
        Z1 = self.W1 @ X + self.b1      # (64,784)·(784,) + (64,) → (64,)
        A1 = relu(Z1)                    # (64,)

        # Capa de salida
        Z2 = self.W2 @ A1 + self.b2     # (10,64)·(64,) + (10,) → (10,)
        A2 = softmax(Z2)                 # (10,)

        return Z1, A1, Z2, A2

    # ──────────────────────────────────────────────────────────
    # Backpropagation
    # ──────────────────────────────────────────────────────────

    def backward(
        self,
        X: np.ndarray,          # (784,)
        y_onehot: np.ndarray,   # (10,)
        Z1: np.ndarray,         # (64,)
        A1: np.ndarray,         # (64,)
        A2: np.ndarray,         # (10,)
    ) -> tuple:
        """
        Calcula gradientes usando regla de la cadena.

        Retorna:
          dW1: (64, 784)
          db1: (64,)
          dW2: (10, 64)
          db2: (10,)
        """
        # ── Gradientes capa de salida ──────────────────────────
        # Derivada combinada Softmax + CrossEntropy: dL/dZ2 = A2 - y
        dZ2 = A2 - y_onehot                         # (10,)
        dW2 = np.outer(dZ2, A1)                     # (10,64)
        db2 = dZ2                                    # (10,)

        # ── Propagación hacia capa oculta ──────────────────────
        dA1 = self.W2.T @ dZ2                       # (64,10)·(10,) → (64,)
        dZ1 = dA1 * relu_derivative(Z1)             # (64,) * (64,) → (64,)
        dW1 = np.outer(dZ1, X)                      # (64,784)
        db1 = dZ1                                    # (64,)

        return dW1, db1, dW2, db2

    # ──────────────────────────────────────────────────────────
    # Actualización de pesos (Gradiente Descendente)
    # ──────────────────────────────────────────────────────────

    def update(
        self,
        dW1: np.ndarray,
        db1: np.ndarray,
        dW2: np.ndarray,
        db2: np.ndarray,
    ) -> None:
        """Modifica W1, b1, W2, b2 in-place."""
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    # ──────────────────────────────────────────────────────────
    # Predicción
    # ──────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> int:
        """
        Retorna el dígito predicho (0–9) para una imagen X (784,).
        """
        _, _, _, A2 = self.forward(X)
        return int(np.argmax(A2))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna el vector completo de probabilidades Softmax (10,).
        """
        _, _, _, A2 = self.forward(X)
        return A2

    # ──────────────────────────────────────────────────────────
    # Utilidades
    # ──────────────────────────────────────────────────────────

    def get_weights_snapshot(self) -> dict:
        """
        Retorna copia de los pesos actuales para la visualización web.
        """
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def get_stats(self) -> dict:
        """Estado actual para el dashboard."""
        return {
            "iteration": self.iteration,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
        }
