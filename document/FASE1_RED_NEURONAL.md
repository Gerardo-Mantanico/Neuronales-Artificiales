# 🧠 Fase 1 — Red Neuronal desde Cero
## Desglose Técnico Completo: Implementación, Pseudocódigo y Estructura de Funciones

**Proyecto 2 — Inteligencia Artificial 1 | USAC CUNOC**

---

## Índice

1. [Arquitectura y topología](#1-arquitectura-y-topología)
2. [Mapa de archivos y responsabilidades](#2-mapa-de-archivos-y-responsabilidades)
3. [Paso 1 — Inicialización de pesos](#3-paso-1--inicialización-de-pesos)
4. [Paso 2 — Funciones de activación](#4-paso-2--funciones-de-activación)
5. [Paso 3 — Forward Propagation](#5-paso-3--forward-propagation)
6. [Paso 4 — Función de pérdida](#6-paso-4--función-de-pérdida)
7. [Paso 5 — Backpropagation](#7-paso-5--backpropagation)
8. [Paso 6 — Actualización de pesos (Gradiente Descendente)](#8-paso-6--actualización-de-pesos-gradiente-descendente)
9. [Paso 7 — Clase MLP (integración)](#9-paso-7--clase-mlp-integración)
10. [Paso 8 — Bucle de entrenamiento](#10-paso-8--bucle-de-entrenamiento)
11. [Paso 9 — Modo Debug](#11-paso-9--modo-debug)
12. [Paso 10 — Bitácora de matrices](#12-paso-10--bitácora-de-matrices)
13. [Flujo de datos completo](#13-flujo-de-datos-completo)
14. [Derivación matemática completa](#14-derivación-matemática-completa)

---

## 1. Arquitectura y Topología

```
Capa de Entrada     Capa Oculta         Capa de Salida
  (784 nodos)        (64 nodos)           (10 nodos)

   x_0  ──┐
   x_1  ──┤          h_0  ──┐             o_0  → P(dígito=0)
   x_2  ──┤──[W1,b1]─h_1  ──┤──[W2,b2]── o_1  → P(dígito=1)
    ...    │          ...    │             ...
   x_783──┘          h_63 ──┘             o_9  → P(dígito=9)

  Activación: ReLU          Activación: Softmax
```

### Dimensiones de matrices

| Variable | Forma | Descripción |
|---|---|---|
| `X` | `(784,)` | Vector de entrada (1 imagen aplanada) |
| `W1` | `(64, 784)` | Pesos capa entrada → oculta |
| `b1` | `(64,)` | Sesgos capa oculta |
| `Z1` | `(64,)` | Suma ponderada capa oculta (pre-activación) |
| `A1` | `(64,)` | Activación capa oculta (post-ReLU) |
| `W2` | `(10, 64)` | Pesos capa oculta → salida |
| `b2` | `(10,)` | Sesgos capa de salida |
| `Z2` | `(10,)` | Suma ponderada capa salida (pre-activación) |
| `A2` | `(10,)` | Probabilidades finales (post-Softmax) |
| `y_onehot` | `(10,)` | Etiqueta verdadera en formato one-hot |

---

## 2. Mapa de Archivos y Responsabilidades

```
neural_network/
│
├── initializer.py   ← Paso 1: generar W1, b1, W2, b2 con semilla fija
├── activations.py   ← Paso 2: relu, relu_deriv, softmax, sigmoid
├── loss.py          ← Paso 4: cross_entropy, accuracy, one_hot
├── network.py       ← Pasos 3,5,6,7: clase MLP (forward, backward, update)
├── trainer.py       ← Paso 8: bucle de entrenamiento con mini-batches
├── debug.py         ← Paso 9: impresión detallada de una iteración
└── logger.py        ← Paso 10: bitácora de matrices en iter 1, 50, 100
```

---

## 3. Paso 1 — Inicialización de Pesos

### Por qué importa

Los pesos no pueden ser todos cero (la red no aprendería: gradientes simétricos).
Se usan valores pequeños aleatorios. Con **semilla fija** el resultado es reproducible
y puede verificarse contra el cálculo manual.

### Estrategia de inicialización: He Initialization

```
Para ReLU:  W ~ N(0, sqrt(2 / n_entradas))
Para Softmax: W ~ N(0, sqrt(1 / n_entradas))   ← Xavier
```

Esto evita que las activaciones exploten o se saturen desde el inicio.

### Pseudocódigo

```
FUNCIÓN initialize_weights(seed=42):

    fijar_semilla_aleatoria(seed)

    # Capa 1: entrada (784) → oculta (64)
    escala_W1 = sqrt(2.0 / 784)          ← He init para ReLU
    W1 = muestras_normal(media=0, desv=escala_W1, forma=(64, 784))
    b1 = ceros(forma=(64,))              ← sesgos inician en 0

    # Capa 2: oculta (64) → salida (10)
    escala_W2 = sqrt(1.0 / 64)           ← Xavier init para Softmax
    W2 = muestras_normal(media=0, desv=escala_W2, forma=(10, 64))
    b2 = ceros(forma=(10,))

    RETORNAR W1, b1, W2, b2
```

### Estructura de la función

```python
# initializer.py

def initialize_weights(seed: int = 42) -> tuple:
    """
    Retorna: (W1, b1, W2, b2)
      W1: (64, 784) — He init
      b1: (64,)     — zeros
      W2: (10, 64)  — Xavier init
      b2: (10,)     — zeros
    """
    ...
```

---

## 4. Paso 2 — Funciones de Activación

### 4.1 ReLU (Rectified Linear Unit)

**Propósito:** Introducir no-linealidad en la capa oculta. Simple y evita el problema del gradiente desvaneciente.

**Fórmula:**
```
ReLU(z) = max(0, z)

Derivada:
ReLU'(z) = 1   si z > 0
ReLU'(z) = 0   si z ≤ 0
```

**Pseudocódigo:**
```
FUNCIÓN relu(Z):
    RETORNAR max(0, Z)          ← aplicado elemento a elemento

FUNCIÓN relu_derivative(Z):
    RETORNAR 1 donde Z > 0, 0 en otro caso
```

---

### 4.2 Softmax

**Propósito:** Convertir los 10 valores de salida en probabilidades que sumen 1.0. Necesario para clasificación multiclase.

**Fórmula:**
```
softmax(z_i) = exp(z_i) / Σ exp(z_j)   para j = 0..9
```

**Problema numérico:** `exp(z)` puede explotar con valores grandes.

**Solución — estabilidad numérica:**
```
softmax(z_i) = exp(z_i - max(Z)) / Σ exp(z_j - max(Z))
```

Restar `max(Z)` no cambia el resultado matemático pero evita overflow.

**Pseudocódigo:**
```
FUNCIÓN softmax(Z):
    Z_estable = Z - max(Z)              ← estabilidad numérica
    exp_Z     = exponencial(Z_estable)
    RETORNAR exp_Z / suma(exp_Z)
```

---

### 4.3 Sigmoid (opcional, alternativa a ReLU)

```
sigmoid(z) = 1 / (1 + exp(-z))

Derivada:
sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
```

**Pseudocódigo:**
```
FUNCIÓN sigmoid(Z):
    RETORNAR 1.0 / (1.0 + exponencial(-Z))

FUNCIÓN sigmoid_derivative(Z):
    s = sigmoid(Z)
    RETORNAR s * (1 - s)
```

### Estructura de las funciones

```python
# activations.py

def relu(Z: np.ndarray) -> np.ndarray:
    """Entrada: Z (64,) | Salida: A (64,)"""
    ...

def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """Entrada: Z (64,) | Salida: máscara binaria (64,)"""
    ...

def softmax(Z: np.ndarray) -> np.ndarray:
    """Entrada: Z (10,) | Salida: probabilidades (10,) que suman 1"""
    ...

def sigmoid(Z: np.ndarray) -> np.ndarray:
    """Entrada: Z (n,) | Salida: A (n,) en rango (0,1)"""
    ...

def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
    """Entrada: Z (n,) | Salida: derivada (n,)"""
    ...
```

---

## 5. Paso 3 — Forward Propagation

### Qué hace

Toma el vector de entrada X y lo "propaga hacia adelante" capa a capa hasta
producir un vector de probabilidades A2. No modifica ningún peso.

### Flujo matemático completo

```
CAPA OCULTA:
  Z1 = W1 · X + b1          (producto punto: (64,784)·(784,) + (64,) → (64,))
  A1 = ReLU(Z1)             (aplicado elemento a elemento → (64,))

CAPA DE SALIDA:
  Z2 = W2 · A1 + b2         (producto punto: (10,64)·(64,) + (10,) → (10,))
  A2 = Softmax(Z2)          (normalización → (10,), suma=1.0)
```

### Pseudocódigo detallado

```
FUNCIÓN forward(X, W1, b1, W2, b2):

    # ── Capa oculta ──────────────────────────────────────────
    Z1 = producto_punto(W1, X) + b1
    # Z1[j] = Σ(W1[j,i] * X[i]) + b1[j]   para j = 0..63
    
    A1 = relu(Z1)
    # A1[j] = max(0, Z1[j])

    # ── Capa de salida ────────────────────────────────────────
    Z2 = producto_punto(W2, A1) + b2
    # Z2[k] = Σ(W2[k,j] * A1[j]) + b2[k]   para k = 0..9

    A2 = softmax(Z2)
    # A2[k] = exp(Z2[k]) / Σ exp(Z2[m])

    RETORNAR Z1, A1, Z2, A2
    # Se guardan Z1 y A1 porque Backprop los necesita
```

### Ejemplo numérico (1 nodo, 3 entradas simplificadas)

```
Suponer: X = [0.5, 0.2, 0.8]  (simplificado, real es 784)
         W1[0] = [0.1, -0.3, 0.5]
         b1[0] = 0.0

Z1[0] = 0.1*0.5 + (-0.3)*0.2 + 0.5*0.8 + 0.0
      = 0.05 - 0.06 + 0.40
      = 0.3900

A1[0] = ReLU(0.3900) = 0.3900   ← positivo, pasa igual
```

### Estructura de la función

```python
# network.py  (método de clase MLP)

def forward(self, X: np.ndarray) -> tuple:
    """
    Parámetros:
      X: vector de entrada, forma (784,)

    Retorna:
      Z1: pre-activación oculta,  (64,)
      A1: activación oculta,      (64,)
      Z2: pre-activación salida,  (10,)
      A2: probabilidades Softmax, (10,)
    """
    ...
```

---

## 6. Paso 4 — Función de Pérdida

### Cross-Entropy Loss

**Por qué Cross-Entropy:** Mide cuánto se equivoca la red comparando la distribución
predicha (Softmax) con la etiqueta real. Penaliza fuertemente las predicciones
incorrectas con alta confianza.

### One-Hot Encoding

Antes de calcular la pérdida, la etiqueta `y` (entero 0–9) se convierte a vector:

```
y = 3  →  y_onehot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

**Pseudocódigo one-hot:**
```
FUNCIÓN one_hot(y, num_clases=10):
    vector = ceros(num_clases)
    vector[y] = 1.0
    RETORNAR vector
```

### Fórmula Cross-Entropy

```
L = -Σ y_k * log(A2_k)   para k = 0..9

Como y_onehot tiene solo un 1 (en posición de la clase correcta):
L = -log(A2[clase_correcta])
```

**Problema numérico:** `log(0)` es infinito. Solución: clipping.

```
L = -log(clip(A2[clase_correcta], mínimo=1e-15))
```

### Pseudocódigo

```
FUNCIÓN one_hot(y, num_clases=10):
    vector = ceros(num_clases)
    vector[y] = 1.0
    RETORNAR vector

FUNCIÓN cross_entropy_loss(A2, y_onehot):
    A2_seguro = clip(A2, 1e-15, 1.0)       ← evitar log(0)
    loss = -suma(y_onehot * log(A2_seguro))
    RETORNAR loss                            ← escalar

FUNCIÓN accuracy(A2, y_real):
    prediccion = argmax(A2)                  ← índice con mayor probabilidad
    RETORNAR 1 si prediccion == y_real, sino 0
```

### Estructura de las funciones

```python
# loss.py

def one_hot(y: int, num_classes: int = 10) -> np.ndarray:
    """
    Entrada: y (entero 0-9)
    Salida:  vector (10,) con un 1 en posición y
    """
    ...

def cross_entropy_loss(A2: np.ndarray, y_onehot: np.ndarray) -> float:
    """
    Entrada: A2 (10,) probabilidades, y_onehot (10,) etiqueta
    Salida:  escalar — pérdida de la muestra
    """
    ...

def accuracy(A2: np.ndarray, y: int) -> int:
    """
    Entrada: A2 (10,), y (entero)
    Salida:  1 si correcto, 0 si incorrecto
    """
    ...
```

---

## 7. Paso 5 — Backpropagation

### Qué hace

Calcula cuánto contribuyó **cada peso** al error total, usando la regla de la cadena
(cálculo diferencial). Estos gradientes indican en qué dirección y magnitud ajustar
cada peso para reducir el error.

### Regla de la cadena — intuición

```
Para ajustar W1[j,i], necesitamos:
  ∂L/∂W1[j,i] = ∂L/∂Z1[j] * ∂Z1[j]/∂W1[j,i]
                            = X[i]

Entonces:   ∂L/∂W1 = ∂L/∂Z1 ⊗ Xᵀ

El truco es calcular ∂L/∂Z1 usando la cadena desde la salida hacia atrás.
```

### Derivación completa capa por capa

#### Gradiente en la capa de salida

```
∂L/∂Z2 = A2 - y_onehot          ← derivada combinada Softmax + CrossEntropy
                                    (simplificación algebraica elegante)

∂L/∂W2 = ∂L/∂Z2 ⊗ A1ᵀ          ← producto externo: (10,) ⊗ (64,) → (10,64)
∂L/∂b2 = ∂L/∂Z2                  ← (10,)
```

**Por qué `∂L/∂Z2 = A2 - y_onehot`:**

La derivada de CrossEntropy respecto a Softmax, combinada con la derivada de
Softmax respecto a Z2, se simplifica algebraicamente a esta forma. No es
coincidencia: es una propiedad de usar la función log con Softmax.

#### Gradiente en la capa oculta

```
∂L/∂A1 = W2ᵀ · ∂L/∂Z2          ← "propagar el error hacia atrás"
                                    (64,10)·(10,) → (64,)

∂L/∂Z1 = ∂L/∂A1 * ReLU'(Z1)    ← multiplicación elemento a elemento
                                    (64,) * (64,) → (64,)
                                    ReLU' es 0 donde Z1≤0 (apaga el gradiente)

∂L/∂W1 = ∂L/∂Z1 ⊗ Xᵀ           ← producto externo: (64,) ⊗ (784,) → (64,784)
∂L/∂b1 = ∂L/∂Z1                  ← (64,)
```

### Pseudocódigo completo

```
FUNCIÓN backward(X, y_onehot, Z1, A1, A2, W1, W2):

    # ── Gradientes de la capa de salida ──────────────────────
    dZ2 = A2 - y_onehot
    # dZ2[k] = A2[k] - y_onehot[k]   para k = 0..9
    # Forma: (10,)

    dW2 = producto_externo(dZ2, A1)
    # dW2[k,j] = dZ2[k] * A1[j]
    # Forma: (10, 64)

    db2 = dZ2
    # Forma: (10,)

    # ── Propagación hacia la capa oculta ─────────────────────
    dA1 = producto_punto(transpuesta(W2), dZ2)
    # dA1 = W2ᵀ · dZ2
    # (64,10) · (10,) → (64,)

    dZ1 = dA1 * relu_derivative(Z1)
    # Multiplicación elemento a elemento
    # Donde Z1[j] ≤ 0 → dZ1[j] = 0  (neurona "muerta" no propaga gradiente)
    # Forma: (64,)

    dW1 = producto_externo(dZ1, X)
    # dW1[j,i] = dZ1[j] * X[i]
    # Forma: (64, 784)

    db1 = dZ1
    # Forma: (64,)

    RETORNAR dW1, db1, dW2, db2
```

### Tabla resumen de gradientes

| Gradiente | Fórmula | Forma |
|---|---|---|
| `dZ2` | `A2 - y_onehot` | `(10,)` |
| `dW2` | `dZ2 ⊗ A1ᵀ` | `(10, 64)` |
| `db2` | `dZ2` | `(10,)` |
| `dA1` | `W2ᵀ · dZ2` | `(64,)` |
| `dZ1` | `dA1 * ReLU'(Z1)` | `(64,)` |
| `dW1` | `dZ1 ⊗ Xᵀ` | `(64, 784)` |
| `db1` | `dZ1` | `(64,)` |

### Estructura de la función

```python
# network.py  (método de clase MLP)

def backward(self,
             X: np.ndarray,        # (784,)
             y_onehot: np.ndarray, # (10,)
             Z1: np.ndarray,       # (64,)  — guardado del forward
             A1: np.ndarray,       # (64,)  — guardado del forward
             A2: np.ndarray        # (10,)  — guardado del forward
             ) -> tuple:
    """
    Retorna:
      dW1: (64, 784)
      db1: (64,)
      dW2: (10, 64)
      db2: (10,)
    """
    ...
```

---

## 8. Paso 6 — Actualización de Pesos (Gradiente Descendente)

### Fórmula

```
W = W - lr * dW
b = b - lr * db
```

El **learning rate (lr)** controla el tamaño del paso. Demasiado grande → diverge.
Demasiado pequeño → aprende muy lento.

### Pseudocódigo

```
FUNCIÓN update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):

    W1 = W1 - lr * dW1     ← actualizar cada uno de los 64*784 = 50,176 pesos
    b1 = b1 - lr * db1     ← actualizar 64 sesgos

    W2 = W2 - lr * dW2     ← actualizar cada uno de los 10*64 = 640 pesos
    b2 = b2 - lr * db2     ← actualizar 10 sesgos

    RETORNAR W1, b1, W2, b2
```

### Verificación manual de un peso

```
Suponer: W2[0,0] = 0.0500,   dW2[0,0] = 0.0123,   lr = 0.01

W2_nuevo[0,0] = 0.0500 - 0.01 * 0.0123
              = 0.0500 - 0.000123
              = 0.049877   ← el peso se ajustó ligeramente
```

### Estructura de la función

```python
# network.py  (método de clase MLP)

def update(self,
           dW1: np.ndarray,  # (64, 784)
           db1: np.ndarray,  # (64,)
           dW2: np.ndarray,  # (10, 64)
           db2: np.ndarray,  # (10,)
           lr: float
           ) -> None:
    """Modifica W1, b1, W2, b2 in-place."""
    ...
```

---

## 9. Paso 7 — Clase MLP (Integración)

### Pseudocódigo de la clase completa

```
CLASE MLP:

    CONSTRUCTOR(lr=0.01, seed=42):
        W1, b1, W2, b2 = initialize_weights(seed)
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.lr = lr
        self.loss_history    = []     ← pérdida por epoch
        self.accuracy_history = []    ← accuracy por epoch

    MÉTODO forward(X):
        Z1 = producto_punto(self.W1, X) + self.b1
        A1 = relu(Z1)
        Z2 = producto_punto(self.W2, A1) + self.b2
        A2 = softmax(Z2)
        RETORNAR Z1, A1, Z2, A2

    MÉTODO backward(X, y_onehot, Z1, A1, A2):
        dZ2 = A2 - y_onehot
        dW2 = producto_externo(dZ2, A1)
        db2 = dZ2
        dA1 = producto_punto(transpuesta(self.W2), dZ2)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = producto_externo(dZ1, X)
        db1 = dZ1
        RETORNAR dW1, db1, dW2, db2

    MÉTODO update(dW1, db1, dW2, db2):
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    MÉTODO predict(X):
        _, _, _, A2 = self.forward(X)
        RETORNAR argmax(A2)          ← dígito predicho (0–9)

    MÉTODO get_weights_snapshot():
        RETORNAR {
            'W1': self.W1,
            'W2': self.W2,
            'b1': self.b1,
            'b2': self.b2
        }
```

### Estructura de la clase

```python
# network.py

import numpy as np
from activations import relu, relu_derivative, softmax
from loss import cross_entropy_loss, accuracy, one_hot
from initializer import initialize_weights

class MLP:
    def __init__(self, lr: float = 0.01, seed: int = 42) -> None: ...
    def forward(self, X: np.ndarray) -> tuple: ...
    def backward(self, X, y_onehot, Z1, A1, A2) -> tuple: ...
    def update(self, dW1, db1, dW2, db2) -> None: ...
    def predict(self, X: np.ndarray) -> int: ...
    def get_weights_snapshot(self) -> dict: ...
```

---

## 10. Paso 8 — Bucle de Entrenamiento

### Lógica del entrenamiento con mini-batches

```
FUNCIÓN train(red, X_train, y_train, epochs, batch_size, logger):

    PARA cada epoch en 1..epochs:
        
        # Mezclar datos aleatoriamente (shuffle)
        indices = permutacion_aleatoria(longitud(X_train))
        X_mezclado = X_train[indices]
        y_mezclado = y_train[indices]

        loss_total    = 0
        correctas     = 0
        iteracion_global = (epoch - 1) * num_batches

        # Dividir en mini-batches
        PARA cada batch de tamaño batch_size en (X_mezclado, y_mezclado):
            
            loss_batch = 0
            iteracion_global += 1

            PARA cada (X_i, y_i) en el batch:

                # 1. Forward
                Z1, A1, Z2, A2 = red.forward(X_i)

                # 2. Pérdida
                y_oh = one_hot(y_i)
                loss_batch += cross_entropy_loss(A2, y_oh)
                correctas  += accuracy(A2, y_i)

                # 3. Backward
                dW1, db1, dW2, db2 = red.backward(X_i, y_oh, Z1, A1, A2)

                # 4. Actualizar pesos
                red.update(dW1, db1, dW2, db2)

                # 5. Bitácora en iteraciones 1, 50 y 100
                SI iteracion_global EN {1, 50, 100}:
                    logger.log_weights(iteracion_global, red.W1)

            loss_total += loss_batch

        # Métricas de la epoch
        loss_promedio = loss_total / longitud(X_train)
        acc_promedio  = correctas  / longitud(X_train)
        red.loss_history.agregar(loss_promedio)
        red.accuracy_history.agregar(acc_promedio)

        imprimir(f"Epoch {epoch} | Loss: {loss_promedio:.4f} | Acc: {acc_promedio:.4f}")
```

### Estructura de la función

```python
# trainer.py

def train(
    network: MLP,
    X_train: np.ndarray,   # (60000, 784)
    y_train: np.ndarray,   # (60000,)
    epochs: int = 20,
    batch_size: int = 32,
    logger = None
) -> dict:
    """
    Retorna:
      {'loss_history': [...], 'accuracy_history': [...]}
    """
    ...
```

---

## 11. Paso 9 — Modo Debug

### Propósito

Imprimir con exactamente 4 decimales cada variable intermedia de **una sola iteración**
para que el cálculo manual pueda verificarse contra el código.

### Qué debe imprimir

```
════════════════════════════════════════════════════
  MODO DEBUG — Iteración única (seed=42, muestra #0)
════════════════════════════════════════════════════

[ENTRADA]
  X[:5]        = [0.0000  0.0000  0.0000  0.0000  0.0000]
  y_real       = 5
  y_onehot     = [0 0 0 0 0 1 0 0 0 0]

[FORWARD — CAPA OCULTA]
  W1[0,:5]     = [ 0.0523  -0.0312   0.0891  -0.0156   0.0634]
  b1[0]        = 0.0000
  Z1[0]        = 0.3742         ← producto punto de W1[0] con X + b1[0]
  Z1[1]        = -0.1823
  Z1[2]        = 0.2219
  A1[0]        = 0.3742         ← ReLU(0.3742) = 0.3742
  A1[1]        = 0.0000         ← ReLU(-0.1823) = 0.0000
  A1[2]        = 0.2219

[FORWARD — CAPA DE SALIDA]
  W2[0,:5]     = [ 0.0211  -0.0445   0.0133   0.0892  -0.0321]
  b2[0]        = 0.0000
  Z2[0]        = 0.1234
  Z2[1]        = -0.0523
  ...
  Z2[9]        = 0.0891
  A2           = [0.1023  0.0892  0.0934  0.1102  0.0967
                  0.1241  0.0856  0.0978  0.1031  0.0976]
  suma(A2)     = 1.0000         ← verificación Softmax

[PÉRDIDA]
  Loss         = 2.2891         ← -log(A2[5]) = -log(0.1241)

[BACKWARD — CAPA DE SALIDA]
  dZ2          = [0.1023  0.0892  0.0934  0.1102  0.0967
                 -0.8759  0.0856  0.0978  0.1031  0.0976]
  dW2[0,0]     = dZ2[0] * A1[0] = 0.1023 * 0.3742 = 0.0383
  db2[0]       = dZ2[0] = 0.1023

[BACKWARD — CAPA OCULTA]
  dA1[0]       = 0.0234         ← W2[:,0]ᵀ · dZ2
  dZ1[0]       = 0.0234         ← dA1[0] * ReLU'(Z1[0]) = 0.0234 * 1 = 0.0234
  dZ1[1]       = 0.0000         ← dA1[1] * ReLU'(Z1[1]) = ? * 0 = 0.0000
  dW1[0,0]     = dZ1[0] * X[0] = 0.0234 * 0.0000 = 0.0000

[ACTUALIZACIÓN DE PESOS (lr=0.01)]
  W2_nuevo[0,0] = W2[0,0] - lr * dW2[0,0]
                = 0.0211 - 0.01 * 0.0383
                = 0.0211 - 0.0004
                = 0.0207
════════════════════════════════════════════════════
```

### Pseudocódigo

```
FUNCIÓN debug_single_sample(red, X, y, lr):

    imprimir encabezado

    # Forward completo con impresión
    Z1, A1, Z2, A2 = red.forward(X)
    y_oh = one_hot(y)
    loss = cross_entropy_loss(A2, y_oh)

    imprimir "X[:5]   =", formatear_4dec(X[:5])
    imprimir "y_real  =", y
    imprimir "Z1[0:3] =", formatear_4dec(Z1[0:3])
    imprimir "A1[0:3] =", formatear_4dec(A1[0:3])
    imprimir "Z2      =", formatear_4dec(Z2)
    imprimir "A2      =", formatear_4dec(A2)
    imprimir "sum(A2) =", formatear_4dec(suma(A2))  ← debe ser 1.0000
    imprimir "Loss    =", formatear_4dec(loss)

    # Backward con impresión
    dW1, db1, dW2, db2 = red.backward(X, y_oh, Z1, A1, A2)

    imprimir "dZ2     =", formatear_4dec(A2 - y_oh)
    imprimir "dW2[0,0]=", formatear_4dec(dW2[0,0])
    imprimir "dA1[0]  =", formatear_4dec(transpuesta(red.W2)[:,0] · (A2-y_oh))
    imprimir "dZ1[0]  =", formatear_4dec(dZ1_calculado[0])
    imprimir "dW1[0,0]=", formatear_4dec(dW1[0,0])

    # Verificación de actualización
    W2_nuevo_00 = red.W2[0,0] - lr * dW2[0,0]
    imprimir "W2_nuevo[0,0] =", formatear_4dec(W2_nuevo_00)
```

### Estructura de la función

```python
# debug.py

def debug_single_sample(
    network: MLP,
    X: np.ndarray,   # (784,) — muestra de índice 0 de MNIST
    y: int,          # etiqueta real (ej. 5)
    lr: float = 0.01
) -> None:
    """
    Imprime todos los cálculos intermedios con 4 decimales.
    NO modifica los pesos de la red.
    """
    ...
```

---

## 12. Paso 10 — Bitácora de Matrices

### Propósito

Evidenciar cómo los pesos de la red **cambian matemáticamente** durante el entrenamiento.
El archivo `.log` debe generarse automáticamente.

### Qué se guarda

En las iteraciones exactas 1, 50 y 100: una sección representativa de W1 (ej. primeras 5 filas × 5 columnas).

### Pseudocódigo

```
FUNCIÓN log_weights(iteracion, W1, archivo="reports/weight_matrix.log"):

    sección = W1[0:5, 0:5]        ← submatriz 5×5 representativa

    contenido = "
    ╔══════════════════════════════════════════════╗
    ║  BITÁCORA DE PESOS — Iteración {iteracion}   ║
    ╚══════════════════════════════════════════════╝
    Sección W1[0:5, 0:5]:
    {sección formateada con 6 decimales}
    Fecha/Hora: {timestamp}
    "

    SI iteracion == 1:
        abrir archivo en modo escritura ('w')
    SINO:
        abrir archivo en modo append ('a')

    escribir contenido en archivo
    cerrar archivo
```

### Formato del archivo generado

```
╔══════════════════════════════════════════════╗
║  BITÁCORA DE PESOS — Iteración 1             ║
╚══════════════════════════════════════════════╝
Sección W1[0:5, 0:5]:
[[ 0.052341  -0.031267   0.089123  -0.015623   0.063412]
 [-0.023456   0.041892  -0.067234   0.028934   0.054321]
 [ 0.078912  -0.012345   0.034567  -0.089012   0.023456]
 [-0.045678   0.067890  -0.023456   0.056789  -0.012345]
 [ 0.034567  -0.078901   0.012345  -0.034567   0.089012]]
Timestamp: 2026-05-03 14:32:01

╔══════════════════════════════════════════════╗
║  BITÁCORA DE PESOS — Iteración 50            ║
╚══════════════════════════════════════════════╝
Sección W1[0:5, 0:5]:
[[ 0.048234  -0.028934   0.082341  -0.011234   0.059023]
...
```

### Estructura de la clase

```python
# logger.py

class WeightLogger:

    def __init__(self, filepath: str = "reports/weight_matrix.log") -> None:
        """Crea/limpia el archivo al instanciar."""
        ...

    def log_weights(self, iteration: int, W1: np.ndarray) -> None:
        """
        Guarda W1[0:5, 0:5] en el archivo si iteration in {1, 50, 100}.
        No hace nada si la iteración no corresponde.
        """
        ...
```

---

## 13. Flujo de Datos Completo

```
                    ┌─────────────────────────────────────────────────────┐
                    │              BUCLE DE ENTRENAMIENTO                  │
                    │                                                       │
   X (784,) ───────►│  forward()                                           │
   y (int)  ───────►│    Z1 = W1·X + b1                                   │
                    │    A1 = ReLU(Z1)              Z1,A1,Z2,A2            │
                    │    Z2 = W2·A1 + b2  ──────────────────────►         │
                    │    A2 = Softmax(Z2)            backward()            │
                    │                                  dZ2 = A2-y_oh      │
                    │  cross_entropy(A2, y_oh)         dW2 = dZ2⊗A1ᵀ     │
                    │    Loss ◄──────────────           db2 = dZ2         │
                    │                                  dA1 = W2ᵀ·dZ2     │
                    │  accuracy(A2, y)                 dZ1 = dA1*ReLU'    │
                    │    1 / 0 ◄────────────           dW1 = dZ1⊗Xᵀ     │
                    │                                  db1 = dZ1          │
                    │                       gradientes ►                   │
                    │                                  update()            │
                    │                                  W1 -= lr*dW1       │
                    │                                  b1 -= lr*db1       │
                    │                                  W2 -= lr*dW2       │
                    │                                  b2 -= lr*db2       │
                    │                                                       │
                    │  SI iter in {1, 50, 100}: logger.log_weights()      │
                    └─────────────────────────────────────────────────────┘
                                           │
                                    métricas por epoch
                                           │
                                    ┌──────▼──────┐
                                    │ loss_history │
                                    │  acc_history │
                                    └─────────────┘
```

---

## 14. Derivación Matemática Completa

### Por qué `∂L/∂Z2 = A2 - y_onehot` (demostración)

```
L = -Σ_k  y_k * log(softmax(Z2)_k)

softmax(Z2)_k = exp(Z2_k) / Σ_m exp(Z2_m)

Para la clase i:
  ∂L/∂Z2_i = Σ_k  (-y_k) * ∂log(A2_k)/∂Z2_i
            = Σ_k  (-y_k) * (1/A2_k) * ∂A2_k/∂Z2_i

La derivada de softmax es:
  ∂A2_k/∂Z2_i = A2_k*(1-A2_k)  si k=i
  ∂A2_k/∂Z2_i = -A2_k*A2_i     si k≠i

Sustituyendo y simplificando:
  ∂L/∂Z2_i = -y_i*(1-A2_i) + Σ_{k≠i} y_k*A2_i
            = -y_i + y_i*A2_i + A2_i*Σ_{k≠i} y_k
            = -y_i + A2_i * Σ_k y_k
            = -y_i + A2_i * 1          ← Σy_k = 1 (one-hot)
            = A2_i - y_i

∴  dZ2 = A2 - y_onehot   ✓
```

### Por qué la derivada de ReLU "apaga" neuronas

```
ReLU(z) = max(0, z)

∂ReLU/∂z = 1   si z > 0
          = 0   si z ≤ 0

En dZ1 = dA1 * ReLU'(Z1):
  - Donde Z1[j] > 0: el gradiente fluye normalmente
  - Donde Z1[j] ≤ 0: el gradiente se vuelve 0
    → ese peso NO se actualizará en esta iteración
    → fenómeno llamado "neurona muerta" (dead neuron)
```

### Verificación de dimensiones (shape check)

```
forward:
  Z1 = (64,784)·(784,) + (64,) = (64,)  ✓
  A1 = ReLU(64,) = (64,)                ✓
  Z2 = (10,64)·(64,) + (10,) = (10,)   ✓
  A2 = softmax(10,) = (10,)             ✓

backward:
  dZ2 = (10,) - (10,) = (10,)           ✓
  dW2 = outer(10,),(64,) = (10,64)      ✓
  dA1 = (64,10)·(10,) = (64,)          ✓
  dZ1 = (64,)*(64,) = (64,)             ✓
  dW1 = outer(64,),(784,) = (64,784)    ✓
```

---

## Resumen de la Interfaz Pública del Módulo

```python
# API completa del módulo neural_network/

# initializer.py
initialize_weights(seed=42) → (W1, b1, W2, b2)

# activations.py
relu(Z)                → np.ndarray
relu_derivative(Z)     → np.ndarray
softmax(Z)             → np.ndarray
sigmoid(Z)             → np.ndarray
sigmoid_derivative(Z)  → np.ndarray

# loss.py
one_hot(y, num_classes=10)          → np.ndarray
cross_entropy_loss(A2, y_onehot)    → float
accuracy(A2, y)                     → int  (0 o 1)

# network.py
class MLP:
    __init__(lr=0.01, seed=42)
    forward(X)                       → (Z1, A1, Z2, A2)
    backward(X, y_onehot, Z1, A1, A2) → (dW1, db1, dW2, db2)
    update(dW1, db1, dW2, db2)      → None
    predict(X)                       → int  (0–9)
    get_weights_snapshot()           → dict

# trainer.py
train(network, X_train, y_train,
      epochs=20, batch_size=32,
      logger=None)                   → dict  {loss_history, accuracy_history}

# debug.py
debug_single_sample(network, X, y, lr=0.01) → None  (solo imprime)

# logger.py
class WeightLogger:
    __init__(filepath="reports/weight_matrix.log")
    log_weights(iteration, W1)       → None
```
