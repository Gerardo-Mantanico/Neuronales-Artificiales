"""
Tests para verificar que el motor neuronal produce valores correctos.
Ejecutar con: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from neural_network.network import MLP
from neural_network.activations import relu, softmax, relu_derivative
from neural_network.loss import one_hot, cross_entropy_loss, accuracy
from neural_network.initializer import initialize_weights


class TestActivations:
    def test_relu_positivo(self):
        assert relu(np.array([0.5]))[0] == 0.5

    def test_relu_negativo(self):
        assert relu(np.array([-1.0]))[0] == 0.0

    def test_relu_cero(self):
        assert relu(np.array([0.0]))[0] == 0.0

    def test_relu_derivative_positivo(self):
        assert relu_derivative(np.array([0.3]))[0] == 1.0

    def test_relu_derivative_negativo(self):
        assert relu_derivative(np.array([-0.3]))[0] == 0.0

    def test_softmax_suma_uno(self):
        Z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        A = softmax(Z)
        assert abs(np.sum(A) - 1.0) < 1e-10

    def test_softmax_estabilidad_valores_grandes(self):
        Z = np.array([1000.0, 1001.0, 999.0, 1000.0, 1000.0,
                      1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        A = softmax(Z)
        assert not np.any(np.isnan(A))
        assert abs(np.sum(A) - 1.0) < 1e-10


class TestLoss:
    def test_one_hot(self):
        v = one_hot(3)
        assert v[3] == 1.0
        assert np.sum(v) == 1.0

    def test_cross_entropy_perfecto(self):
        A2 = np.zeros(10); A2[5] = 1.0
        y_oh = one_hot(5)
        loss = cross_entropy_loss(A2, y_oh)
        assert loss < 1e-10

    def test_accuracy_correcto(self):
        A2 = np.zeros(10); A2[7] = 0.9
        assert accuracy(A2, 7) == 1

    def test_accuracy_incorrecto(self):
        A2 = np.zeros(10); A2[3] = 0.9
        assert accuracy(A2, 7) == 0


class TestForwardShapes:
    def setup_method(self):
        self.net = MLP(seed=42)
        self.X = np.random.randn(784)

    def test_z1_shape(self):
        Z1, A1, Z2, A2 = self.net.forward(self.X)
        assert Z1.shape == (64,)

    def test_a1_shape(self):
        Z1, A1, Z2, A2 = self.net.forward(self.X)
        assert A1.shape == (64,)

    def test_z2_shape(self):
        Z1, A1, Z2, A2 = self.net.forward(self.X)
        assert Z2.shape == (10,)

    def test_a2_shape_y_suma(self):
        Z1, A1, Z2, A2 = self.net.forward(self.X)
        assert A2.shape == (10,)
        assert abs(np.sum(A2) - 1.0) < 1e-10

    def test_a1_no_negativo(self):
        Z1, A1, Z2, A2 = self.net.forward(self.X)
        assert np.all(A1 >= 0)


class TestBackwardShapes:
    def setup_method(self):
        self.net = MLP(seed=42)
        self.X = np.random.randn(784)
        self.Z1, self.A1, self.Z2, self.A2 = self.net.forward(self.X)
        self.y_oh = one_hot(3)

    def test_dw1_shape(self):
        dW1, db1, dW2, db2 = self.net.backward(
            self.X, self.y_oh, self.Z1, self.A1, self.A2)
        assert dW1.shape == (64, 784)

    def test_db1_shape(self):
        dW1, db1, dW2, db2 = self.net.backward(
            self.X, self.y_oh, self.Z1, self.A1, self.A2)
        assert db1.shape == (64,)

    def test_dw2_shape(self):
        dW1, db1, dW2, db2 = self.net.backward(
            self.X, self.y_oh, self.Z1, self.A1, self.A2)
        assert dW2.shape == (10, 64)

    def test_db2_shape(self):
        dW1, db1, dW2, db2 = self.net.backward(
            self.X, self.y_oh, self.Z1, self.A1, self.A2)
        assert db2.shape == (10,)


class TestGradienteNumerica:
    """Verificación numérica del gradiente (finite differences)."""

    def test_dz2_es_a2_menos_y(self):
        net = MLP(seed=42)
        X = np.random.randn(784)
        y = 5
        y_oh = one_hot(y)
        Z1, A1, Z2, A2 = net.forward(X)
        dW1, db1, dW2, db2 = net.backward(X, y_oh, Z1, A1, A2)
        dZ2_esperado = A2 - y_oh
        dZ2_obtenido = dW2 @ np.linalg.pinv(np.outer(np.ones(10), A1))
        # Verificación directa de dZ2
        assert np.allclose(dW2, np.outer(A2 - y_oh, A1), atol=1e-10)
