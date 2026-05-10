import os
import numpy as np
from .initializer import initialize_weights
from .activations import relu, relu_derivative, softmax
from .loss import one_hot, cross_entropy_loss, accuracy


class MLP:
    # Clase principal para el Perceptron Multicapa 784 -> 64 -> 10

    def __init__(self, lr=0.01, seed=42):
        self.lr = lr
        self.W1, self.b1, self.W2, self.b2 = initialize_weights(seed)
        
        # Historial de perdidas y accuracy para las graficas
        self.loss_history = []
        self.accuracy_history = []
        self.iteration = 0

    # --- Forward propagation ---
    def forward(self, X):
        # Capa oculta (ReLU)
        Z1 = self.W1 @ X + self.b1
        A1 = relu(Z1)

        # Capa de salida (Softmax)
        Z2 = self.W2 @ A1 + self.b2
        A2 = softmax(Z2)

        return Z1, A1, Z2, A2

    # --- Backpropagation ---
    def backward(self, X, y_onehot, Z1, A1, A2):
        # Derivada del error respecto a Z2 (Softmax + Cross Entropy)
        dZ2 = A2 - y_onehot
        dW2 = np.outer(dZ2, A1)
        db2 = dZ2

        # Retropropagamos el error a la capa oculta
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.outer(dZ1, X)
        db1 = dZ1

        return dW1, db1, dW2, db2

    # --- Actualizar los pesos ---
    def update(self, dW1, db1, dW2, db2):
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    # --- Prediccion ---
    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return int(np.argmax(A2))

    def predict_proba(self, X):
        _, _, _, A2 = self.forward(X)
        return A2

    # --- Utilidades ---
    def get_weights_snapshot(self):
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def get_stats(self):
        return {
            "iteration": self.iteration,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
        }

    def save(self, path):
        dirname = os.path.dirname(path) or "."
        os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe el archivo de pesos: {path}")
        data = np.load(path)
        self.W1 = data["W1"].astype(np.float64)
        self.b1 = data["b1"].astype(np.float64)
        self.W2 = data["W2"].astype(np.float64)
        self.b2 = data["b2"].astype(np.float64)
