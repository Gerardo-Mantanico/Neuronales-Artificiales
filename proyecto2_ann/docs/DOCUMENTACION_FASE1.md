# Documentación Técnica — Fase 1: Red Neuronal desde Cero
**Proyecto 2 — Inteligencia Artificial 1 | USAC CUNOC**  
**Autor:** Gerardo-Mantanico | **Docente:** Ing. Daniel González

---

## Índice

1. [Resumen de lo implementado](#1-resumen-de-lo-implementado)
2. [Estructura de archivos](#2-estructura-de-archivos)
3. [Archivo: `initializer.py`](#3-archivo-initializerpy)
4. [Archivo: `activations.py`](#4-archivo-activationspy)
5. [Archivo: `loss.py`](#5-archivo-losspy)
6. [Archivo: `network.py` — Clase MLP](#6-archivo-networkpy--clase-mlp)
7. [Archivo: `trainer.py`](#7-archivo-trainerpy)
8. [Archivo: `logger.py`](#8-archivo-loggerpy)
9. [Archivo: `debug.py`](#9-archivo-debugpy)
10. [Flujo completo de una iteración](#10-flujo-completo-de-una-iteración)
11. [Resultados de entrenamiento obtenidos](#11-resultados-de-entrenamiento-obtenidos)
12. [Restricciones cumplidas](#12-restricciones-cumplidas)

---

## 1. Resumen de lo Implementado

Se construyó un **Perceptrón Multicapa (MLP)** completo desde cero, sin ninguna librería de Machine Learning. El motor matemático usa únicamente **NumPy** para operaciones matriciales.

### Topología de la red

```
Capa de Entrada   →   Capa Oculta   →   Capa de Salida
   784 nodos            64 nodos            10 nodos
  (píxeles de         (activación          (probabilidades
   imagen 28×28)         ReLU)              Softmax, dígitos 0–9)
```

### Capacidades implementadas

| Capacidad | Estado |
|---|---|
| Inicialización de pesos (He + Xavier) | Implementado |
| Forward Propagation (capa a capa) | Implementado |
| Función de activación ReLU | Implementado |
| Función de activación Sigmoid (alternativa) | Implementado |
| Función de salida Softmax (estabilidad numérica) | Implementado |
| Función de pérdida Cross-Entropy (con clipping) | Implementado |
| Métrica de accuracy por muestra | Implementado |
| Backpropagation con regla de la cadena | Implementado |
| Gradiente Descendente (actualización de pesos) | Implementado |
| Bucle de entrenamiento con mini-batches y shuffle | Implementado |
| Modo paso a paso (una iteración a la vez) | Implementado |
| Modo debug (impresión con 4 decimales) | Implementado |
| Bitácora automática en iteraciones 1, 50 y 100 | Implementado |
| Callback para WebSocket (UI en tiempo real) | Implementado |

---

## 2. Estructura de Archivos

```
neural_network/
├── __init__.py       ← Exporta todo el módulo
├── initializer.py    ← Generación de pesos iniciales con semilla fija
├── activations.py    ← relu, relu_derivative, softmax, sigmoid
├── loss.py           ← one_hot, cross_entropy_loss, accuracy
├── network.py        ← Clase MLP: forward, backward, update, predict
├── logger.py         ← Bitácora de matrices W1 en iter 1, 50, 100
└── debug.py          ← Impresión detallada de una iteración completa

trainer.py            ← Bucle de entrenamiento (mini-batch) + modo paso a paso
```

---

## 3. Archivo: `initializer.py`

### Propósito

Genera los pesos y sesgos iniciales de la red con una **semilla fija** para reproducibilidad. Esto permite que el cálculo manual pueda verificarse contra la primera iteración del programa.

### Función implementada

```python
def initialize_weights(seed: int = 42) -> tuple
```

### Decisiones de diseño

| Parámetro | Estrategia | Por qué |
|---|---|---|
| `W1` (64×784) | **He initialization**: `N(0, √(2/784))` | ReLU funciona mejor con He; evita activaciones saturadas |
| `b1` (64,) | Ceros | Simetría correcta en el inicio; ReLU se encarga de la asimetría |
| `W2` (10×64) | **Xavier initialization**: `N(0, √(1/64))` | Softmax no tiene no-linealidad tan agresiva; Xavier es más apropiado |
| `b2` (10,) | Ceros | Igual que b1 |
| Semilla | `np.random.seed(42)` | Reproducibilidad; permite comparar contra cálculo manual |

### Valores que produce (con seed=42)

```
escala_W1 = √(2/784)  ≈ 0.05051
escala_W2 = √(1/64)   ≈ 0.12500

W1.shape = (64, 784)   — 50,176 parámetros
b1.shape = (64,)       — 64 parámetros
W2.shape = (10, 64)    — 640 parámetros
b2.shape = (10,)       — 10 parámetros

Total de parámetros entrenables: 50,890
```

---

## 4. Archivo: `activations.py`

### Funciones implementadas

#### `relu(Z)` y `relu_derivative(Z)`

```python
def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, Z)

def relu_derivative(Z: np.ndarray) -> np.ndarray:
    return (Z > 0).astype(float)
```

**Fórmula matemática:**
```
ReLU(z)  = max(0, z)
ReLU'(z) = 1  si z > 0
           0  si z ≤ 0
```

**Comportamiento clave:** `relu_derivative` retorna una **máscara binaria** del mismo shape que Z. En Backpropagation, multiplicar por esta máscara **apaga el gradiente** de los nodos que estaban inactivos (Z ≤ 0). Esto es el fenómeno de "dead neurons": un nodo que nunca se activa no contribuye al aprendizaje.

**Implementación vectorizada:** `np.maximum` y la comparación `(Z > 0)` operan sobre arrays completos sin bucles Python, lo que hace la operación eficiente.

---

#### `softmax(Z)`

```python
def softmax(Z: np.ndarray) -> np.ndarray:
    Z_estable = Z - np.max(Z)
    exp_Z = np.exp(Z_estable)
    return exp_Z / np.sum(exp_Z)
```

**Fórmula matemática:**
```
softmax(z_i) = exp(z_i) / Σ exp(z_j)   para j = 0..9
```

**Propiedad garantizada:** `sum(A2) == 1.0` siempre.

**Problema que resuelve la estabilidad numérica:** Si `Z = [1000, 999, ...]`, entonces `exp(1000)` produce `inf` en punto flotante (overflow). Restar `max(Z)` antes de exponenciar transforma los valores a un rango manejable sin cambiar el resultado matemático, porque:

```
softmax(z_i - c) = exp(z_i - c) / Σ exp(z_j - c)
                 = exp(z_i)*exp(-c) / (Σ exp(z_j)*exp(-c))
                 = exp(z_i) / Σ exp(z_j)   ← c se cancela
```

---

#### `sigmoid(Z)` y `sigmoid_derivative(Z)`

```python
def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
    s = sigmoid(Z)
    return s * (1.0 - s)
```

**Incluida como alternativa** a ReLU para la capa oculta. No se usa en la configuración por defecto, pero está disponible si se desea comparar comportamientos.

---

## 5. Archivo: `loss.py`

### Funciones implementadas

#### `one_hot(y, num_classes=10)`

```python
def one_hot(y: int, num_classes: int = 10) -> np.ndarray:
    vector = np.zeros(num_classes)
    vector[y] = 1.0
    return vector
```

**Propósito:** Convierte la etiqueta entera en el vector que necesita Cross-Entropy.

```
Ejemplo: y = 5  →  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                                    ↑ índice 5
```

---

#### `cross_entropy_loss(A2, y_onehot)`

```python
def cross_entropy_loss(A2: np.ndarray, y_onehot: np.ndarray) -> float:
    A2_seguro = np.clip(A2, 1e-15, 1.0)
    return -float(np.sum(y_onehot * np.log(A2_seguro)))
```

**Fórmula matemática:**
```
L = -Σ y_k * log(A2_k)   para k = 0..9
```

Como `y_onehot` tiene un único 1 (en la posición de la clase correcta), esto simplifica a:
```
L = -log(A2[clase_correcta])
```

**Interpretación:** Cuando la predicción es perfecta `A2[correcta] = 1.0`, la pérdida es `0`. Cuando la predicción es completamente incorrecta `A2[correcta] ≈ 0`, la pérdida tiende a infinito.

**Clipping:** `np.clip(A2, 1e-15, 1.0)` evita que `log(0)` produzca `-inf`, lo cual corrompería los gradientes.

---

#### `accuracy(A2, y)`

```python
def accuracy(A2: np.ndarray, y: int) -> int:
    return int(np.argmax(A2) == y)
```

Retorna `1` si el índice de mayor probabilidad coincide con la etiqueta real, `0` en caso contrario. Acumular estos valores y dividir por el total de muestras da el accuracy porcentual.

---

## 6. Archivo: `network.py` — Clase MLP

### Descripción general

La clase `MLP` integra todos los componentes del motor neuronal. Mantiene el estado de los pesos y el historial de métricas. No tiene dependencias de librerías ML.

```python
class MLP:
    W1: np.ndarray   # (64, 784)
    b1: np.ndarray   # (64,)
    W2: np.ndarray   # (10, 64)
    b2: np.ndarray   # (10,)
    lr: float
    iteration: int
    loss_history: list[float]
    accuracy_history: list[float]
```

---

### Método `forward(X)`

```python
def forward(self, X: np.ndarray) -> tuple:
    Z1 = self.W1 @ X + self.b1      # (64,784)·(784,) + (64,) → (64,)
    A1 = relu(Z1)                    # (64,)
    Z2 = self.W2 @ A1 + self.b2     # (10,64)·(64,) + (10,) → (10,)
    A2 = softmax(Z2)                 # (10,)
    return Z1, A1, Z2, A2
```

**Flujo matemático capa a capa:**

```
X (784,)
  │
  │  Z1[j] = Σᵢ W1[j,i] · X[i] + b1[j]    para j = 0..63
  ▼
Z1 (64,)
  │
  │  A1[j] = max(0, Z1[j])
  ▼
A1 (64,)
  │
  │  Z2[k] = Σⱼ W2[k,j] · A1[j] + b2[k]   para k = 0..9
  ▼
Z2 (10,)
  │
  │  A2[k] = exp(Z2[k] - max(Z2)) / Σₘ exp(Z2[m] - max(Z2))
  ▼
A2 (10,)  ← suma = 1.0  (probabilidades finales)
```

**Por qué se retornan Z1 y A1:** Backpropagation los necesita. Si no se guardan aquí, habría que recalcularlos en backward (doble cómputo).

---

### Método `backward(X, y_onehot, Z1, A1, A2)`

```python
def backward(self, X, y_onehot, Z1, A1, A2) -> tuple:
    # Capa de salida
    dZ2 = A2 - y_onehot                  # (10,)
    dW2 = np.outer(dZ2, A1)              # (10,64)
    db2 = dZ2                             # (10,)

    # Capa oculta
    dA1 = self.W2.T @ dZ2               # (64,)
    dZ1 = dA1 * relu_derivative(Z1)     # (64,)
    dW1 = np.outer(dZ1, X)              # (64,784)
    db1 = dZ1                            # (64,)

    return dW1, db1, dW2, db2
```

**Derivación de cada gradiente:**

```
┌─────────────────────────────────────────────────────────────────┐
│ GRADIENTE CAPA DE SALIDA                                         │
│                                                                   │
│ ∂L/∂Z2 = A2 - y_onehot                                          │
│                                                                   │
│ Origen: derivada combinada de CrossEntropy(Softmax(Z2), y)       │
│ Al aplicar la regla de la cadena y simplificar algebraicamente:   │
│   ∂L/∂Z2_i = A2_i - y_i   (forma elegante y eficiente)          │
│                                                                   │
│ ∂L/∂W2[k,j] = ∂L/∂Z2[k] · ∂Z2[k]/∂W2[k,j] = dZ2[k] · A1[j]  │
│ → dW2 = outer(dZ2, A1)   forma: (10, 64)                        │
│                                                                   │
│ ∂L/∂b2[k]  = ∂L/∂Z2[k] · 1 = dZ2[k]                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ GRADIENTE CAPA OCULTA                                            │
│                                                                   │
│ ∂L/∂A1 = W2ᵀ · dZ2   ← "propagar el error hacia atrás"         │
│   forma: (64,10)·(10,) → (64,)                                  │
│                                                                   │
│ ∂L/∂Z1 = ∂L/∂A1 * ReLU'(Z1)   ← multiplicación elem a elem    │
│   Donde Z1[j] ≤ 0: gradiente = 0 (neurona inactiva)             │
│   Donde Z1[j] > 0: gradiente fluye normalmente                  │
│                                                                   │
│ ∂L/∂W1[j,i] = dZ1[j] · X[i]                                    │
│ → dW1 = outer(dZ1, X)   forma: (64, 784)                        │
│                                                                   │
│ ∂L/∂b1[j]  = dZ1[j]                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Verificación de dimensiones:**

| Gradiente | Operación | Entrada | Salida |
|---|---|---|---|
| `dZ2` | `A2 - y_oh` | (10,) - (10,) | (10,) |
| `dW2` | `outer(dZ2, A1)` | (10,), (64,) | (10, 64) |
| `db2` | `= dZ2` | (10,) | (10,) |
| `dA1` | `W2.T @ dZ2` | (64,10)·(10,) | (64,) |
| `dZ1` | `dA1 * relu_deriv(Z1)` | (64,) * (64,) | (64,) |
| `dW1` | `outer(dZ1, X)` | (64,), (784,) | (64, 784) |
| `db1` | `= dZ1` | (64,) | (64,) |

---

### Método `update(dW1, db1, dW2, db2)`

```python
def update(self, dW1, db1, dW2, db2) -> None:
    self.W1 -= self.lr * dW1
    self.b1 -= self.lr * db1
    self.W2 -= self.lr * dW2
    self.b2 -= self.lr * db2
```

**Fórmula del Gradiente Descendente:**
```
W ← W - η · ∂L/∂W
b ← b - η · ∂L/∂b
```

Donde `η` (eta) es el **learning rate**, configurado en `lr=0.01` por defecto.

**Operación in-place (`-=`):** Modifica los arrays de pesos directamente en memoria sin crear copias intermedias. Más eficiente en memoria que `W1 = W1 - lr * dW1`.

---

### Métodos de predicción

```python
def predict(self, X: np.ndarray) -> int:
    _, _, _, A2 = self.forward(X)
    return int(np.argmax(A2))            # índice 0–9 con mayor probabilidad

def predict_proba(self, X: np.ndarray) -> np.ndarray:
    _, _, _, A2 = self.forward(X)
    return A2                             # vector completo (10,) para mostrar en UI
```

---

## 7. Archivo: `trainer.py`

### Función `train()`

Implementa el **bucle de entrenamiento completo** con mini-batch Gradient Descent.

```python
def train(network, X_train, y_train, epochs=20, batch_size=32, logger=None, callback=None)
```

**Flujo interno por epoch:**

```
1. Shuffle: np.random.permutation(n_muestras)
   → Evita que el orden de los datos cause sesgo en el aprendizaje

2. División en mini-batches de tamaño batch_size=32
   → Cada batch agrupa muestras para actualizaciones más frecuentes

3. Por cada muestra (X_i, y_i) dentro del batch:
   a. network.iteration += 1
   b. Z1, A1, Z2, A2 = network.forward(X_i)
   c. y_oh = one_hot(y_i)
   d. loss_epoch += cross_entropy_loss(A2, y_oh)
   e. correctas_epoch += accuracy(A2, y_i)
   f. dW1, db1, dW2, db2 = network.backward(X_i, y_oh, Z1, A1, A2)
   g. network.update(dW1, db1, dW2, db2)
   h. logger.log_weights(network.iteration, network.W1)  ← si iter in {1,50,100}

4. Métricas al final de la epoch:
   loss_avg = loss_epoch / n_muestras
   acc_avg  = correctas / n_muestras

5. callback(epoch, loss_avg, acc_avg)  ← emite datos al WebSocket en tiempo real
```

**Nota:** Se usa **Stochastic Gradient Descent por muestra** dentro del mini-batch (actualización después de cada imagen, no acumulando gradientes del batch completo). Esto es más simple matemáticamente y converge bien en MNIST.

---

### Función `train_single_step()`

```python
def train_single_step(network, X_i, y_i, logger=None) -> dict
```

Ejecuta exactamente **un paso** completo (forward + backward + update) y retorna un diccionario con:

```python
{
    "iteration":     int,          # número de iteración global
    "loss":          float,        # pérdida de esta muestra
    "accuracy":      int,          # 1 o 0
    "prediction":    int,          # dígito predicho (0–9)
    "probabilities": list[float],  # vector A2 completo (10 valores)
    "weights_sample": {
        "W1_sample": list,         # W1[0:5, 0:5] para visualización
        "W2_sample": list,         # W2[0:5, 0:5] para visualización
    },
    "activations": {
        "A1": list,                # activaciones capa oculta (64,)
        "A2": list,                # probabilidades finales (10,)
    }
}
```

Usado por el **modo educativo** de la UI: cada clic en "Siguiente Paso" llama a este método y actualiza la visualización con los datos retornados.

---

## 8. Archivo: `logger.py`

### Clase `WeightLogger`

```python
class WeightLogger:
    ITERACIONES_OBJETIVO = {1, 50, 100}

    def __init__(self, filepath="reports/weight_matrix.log")
    def log_weights(self, iteration: int, W1: np.ndarray) -> None
```

**Comportamiento:**
- Al instanciar: crea o sobreescribe el archivo `.log` con encabezado y timestamp.
- En `log_weights`: si `iteration` pertenece a `{1, 50, 100}`, extrae la sección `W1[0:5, 350:355]` (5 filas × 5 columnas del área central de la imagen 28×28) y la escribe en el archivo en modo append con 8 decimales.
- Columnas 350–355 corresponden a la zona central de la imagen (no al borde, que siempre es negro en MNIST y no recibe gradiente).

**Ejemplo de salida generada:**
```
╔══════════════════════════════════════════════════════╗
║  ITERACIÓN 1                                          ║
╚══════════════════════════════════════════════════════╝
Sección W1[0:5, 350:355]  (zona central activa):
  [+0.02508785  -0.00698340  +0.03271321  +0.07692462  -0.01182653]
  ...
Timestamp: 2026-05-03 23:14:56
```

**La variación entre iteraciones 1, 50 y 100 evidencia el aprendizaje:** los pesos se alejan de sus valores iniciales conforme la red ajusta su representación interna.

---

## 9. Archivo: `debug.py`

### Función `debug_single_sample(network, X, y)`

**Propósito:** Imprimir cada variable intermedia de **una sola iteración** con exactamente 4 decimales, para que el docente pueda comparar contra el cálculo manual.

**NO modifica los pesos**: solo lee el estado actual de la red y calcula, sin llamar a `update()`.

**Secciones que imprime:**

```
[ENTRADA]
  X[:10]       primeros 10 píxeles de la imagen
  y_real       etiqueta entera (0–9)
  y_onehot     vector one-hot correspondiente

[FORWARD — CAPA OCULTA]
  W1[0,:5]     primeros 5 pesos de la neurona 0
  b1[0]        sesgo de la neurona 0
  Z1[:5]       pre-activaciones de las primeras 5 neuronas ocultas
  A1[:5]       activaciones ReLU de las primeras 5 neuronas
  Nodos activos (Z1>0): X / 64   ← cuántas neuronas se activaron

[FORWARD — CAPA DE SALIDA]
  W2[0,:5]     primeros 5 pesos de la neurona de salida 0
  b2[0]        sesgo de la neurona de salida 0
  Z2           todos los 10 valores pre-Softmax
  A2           todas las 10 probabilidades finales
  sum(A2)      verificación: debe ser exactamente 1.0000000000

[PÉRDIDA]
  Loss         Cross-Entropy de esta muestra
  Predicción   dígito predicho vs. correcto

[BACKWARD — CAPA DE SALIDA]
  dZ2          gradiente combinado Softmax + CrossEntropy
  dW2[0,:3]    primeras 3 actualizaciones de W2
  db2[0]       actualización del sesgo de salida 0

[BACKWARD — CAPA OCULTA]
  dA1[:5]      gradiente propagado hacia A1
  dZ1[:5]      gradiente en Z1 (después de ReLU')
  dW1[0,:3]    primeras 3 actualizaciones de W1
  db1[0]       actualización del sesgo oculto 0

[ACTUALIZACIÓN (lr=0.01)]
  W2[0,0] antes / dW2[0,0] / W2[0,0] después
  W1[0,0] antes / dW1[0,0] / W1[0,0] después
```

---

## 10. Flujo Completo de una Iteración

Diagrama de cómo interactúan todos los archivos durante una iteración de entrenamiento:

```
trainer.py
    │
    │  X_i, y_i  (una imagen del dataset MNIST)
    │
    ▼
network.py → forward(X_i)
    │  initializer.py proveyó W1, b1, W2, b2
    │
    │  Z1 = W1 @ X_i + b1
    │  A1 = relu(Z1)               ← activations.py
    │  Z2 = W2 @ A1 + b2
    │  A2 = softmax(Z2)            ← activations.py
    │
    ▼
loss.py → cross_entropy_loss(A2, one_hot(y_i))
    │  pérdida escalar + accuracy (1 o 0)
    │
    ▼
network.py → backward(X_i, y_oh, Z1, A1, A2)
    │  dZ2 = A2 - y_oh
    │  dW2 = outer(dZ2, A1)
    │  dA1 = W2.T @ dZ2
    │  dZ1 = dA1 * relu_derivative(Z1)   ← activations.py
    │  dW1 = outer(dZ1, X_i)
    │
    ▼
network.py → update(dW1, db1, dW2, db2)
    │  W1 -= lr * dW1  ←  los pesos cambian aquí
    │  W2 -= lr * dW2
    │
    ▼
logger.py → log_weights(iteration, W1)
    │  Si iteration in {1, 50, 100}: escribe W1[0:5, 350:355] en .log
    │
    ▼
callback(epoch, loss, acc)
    │  Emite datos al WebSocket para actualizar la UI en tiempo real
```

---

## 11. Resultados de Entrenamiento Obtenidos

Entrenamiento de prueba ejecutado: **2 epochs, lr=0.01, batch_size=32, seed=42, dataset MNIST (60,000 muestras de entrenamiento).**

| Epoch | Loss promedio | Accuracy |
|---|---|---|
| 1 | 0.2435 | 92.63% |
| 2 | 0.1175 | 96.37% |

**Interpretación:**
- En solo 2 epochs la red alcanza **96.37% de accuracy** en entrenamiento, lo que demuestra que Forward Propagation, Backpropagation y el Gradiente Descendente están correctamente implementados.
- La curva de pérdida descendente confirma que los pesos se están ajustando en la dirección correcta.
- Con 20 epochs (configuración completa) se espera superar 98% en entrenamiento y ~97% en test.

---

## 12. Restricciones Cumplidas

| Restricción del enunciado | Cumplimiento |
|---|---|
| Red MLP con topología exacta 784 → 64 → 10 | ✓ `initializer.py` y `network.py` |
| Función de activación Softmax en capa de salida | ✓ `activations.softmax()` |
| **PROHIBIDO** TensorFlow, PyTorch, Keras, Scikit-Learn | ✓ Solo se usa `numpy` y `math` |
| Permitido: `numpy`, `math` | ✓ Única librería matemática usada |
| Cálculo manual verificable con modo debug | ✓ `debug.py` imprime con 4 decimales |
| Bitácora de matrices en iteraciones 1, 50 y 100 | ✓ `logger.py` — archivo `weight_matrix.log` |
| Código en Python | ✓ Totalmente en Python 3.10+ |
| Callback para visualización en tiempo real | ✓ `trainer.train()` — parámetro `callback` |
| Modo paso a paso para interfaz educativa | ✓ `trainer.train_single_step()` |
