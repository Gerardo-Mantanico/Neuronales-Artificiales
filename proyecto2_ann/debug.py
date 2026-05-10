import sys
import os

# Permitir importaciones locales desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(__file__))

from neural_network.network import MLP
from neural_network.debug import debug_single_sample
from data.mnist_loader import load_mnist

if __name__ == "__main__":
    print("Cargando dataset MNIST para modo debug...")
    X_train, y_train, _, _ = load_mnist()
    
    # Instanciar red neuronal con semilla 42
    network = MLP(lr=0.01, seed=42)
    
    # Primera muestra (índice 0, dígito real 5)
    X = X_train[0]
    y = int(y_train[0])
    
    debug_single_sample(network, X, y)
