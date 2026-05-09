# Manual Técnico
## ANN desde Cero — Proyecto 2 | Inteligencia Artificial 1 | USAC CUNOC
**Ing. Daniel González | Autor: Gerardo Mantanico**

---

## 1. Descripción General del Sistema

El proyecto implementa un **Perceptrón Multicapa (MLP)** desde cero usando únicamente NumPy. La red reconoce dígitos escritos a mano (0–9) capturados por la cámara del dispositivo.

**Tecnologías:**

| Capa | Tecnología | Propósito |
|---|---|---|
| Motor neuronal | Python + NumPy | Álgebra lineal y cálculo diferencial |
| Backend web | Flask + Flask-SocketIO | API REST y WebSocket |
| Procesamiento imagen | OpenCV + Pillow | Pipeline cámara → vector 784 |
| Frontend | HTML5 + JavaScript | Interfaz interactiva |
| Comunicación tiempo real | Socket.IO + gevent | Actualización en vivo |

**Restricción cumplida:** No se usa TensorFlow, PyTorch, Keras ni Scikit-Learn.

---

## 2. Arquitectura de la Red Neuronal

### 2.1 Topología

```
Capa de Entrada    Capa Oculta      Capa de Salida
   784 nodos   →    64 nodos    →     10 nodos
   (imagen)       (ReLU)           (Softmax → P(dígito))
```

| Capa | Dimensión | Función de Activación |
|---|---|---|
| Entrada | 784 | — (vector de píxeles) |
| Oculta | 64 | ReLU |
| Salida | 10 | Softmax |

### 2.2 Parámetros (pesos y sesgos)

| Matriz | Dimensiones | Inicialización |
|---|---|---|
| W1 | (64 × 784) | He: σ = √(2/784) |
| b1 | (64,) | Ceros |
| W2 | (10 × 64) | Xavier: σ = √(1/64) |
| b2 | (10,) | Ceros |

**Total de parámetros entrenables:**
- W1: 64 × 784 = **50,176**
- b1: 64
- W2: 10 × 64 = **640**
- b2: 10
- **Total: 50,890 parámetros**

---

## 3. Inicialización de Pesos

### 3.1 Inicialización He (para ReLU)

Se aplica a W1 porque la capa oculta usa ReLU. La inicialización He evita que los gradientes se desvanezcan o exploten con ReLU:

```
W1 ~ N(0, σ²)    donde    σ = √(2 / n_entrada) = √(2 / 784) ≈ 0.0505
```

**Código:**
```python
np.random.seed(42)
W1 = np.random.randn(64, 784) * np.sqrt(2.0 / 784)
```

**Por qué He:** Con ReLU, aproximadamente la mitad de las neuronas se anulan (Z < 0 → A = 0), por lo que se necesita el factor 2 para compensar la varianza perdida.

### 3.2 Inicialización Xavier/Glorot (para Softmax)

Se aplica a W2 porque la capa de salida usa Softmax:

```
W2 ~ N(0, σ²)    donde    σ = √(1 / n_oculta) = √(1 / 64) = 0.125
```

**Código:**
```python
W2 = np.random.randn(10, 64) * np.sqrt(1.0 / 64)
```

---

## 4. Propagación Hacia Adelante (Forward Propagation)

### 4.1 Capa Oculta

**Paso 1 — Pre-activación:**
```
Z1 = W1 · X + b1
```

- W1: matriz (64 × 784)
- X: vector de entrada (784,)
- b1: sesgo (64,)
- Z1: resultado (64,) — producto punto + sesgo para cada neurona oculta

**Paso 2 — Activación ReLU:**
```
A1 = ReLU(Z1) = max(0, Z1)
```

ReLU aplica elemento a elemento. Cualquier Z1 negativo se convierte en 0.

**Derivada de ReLU:**
```
ReLU'(Z1) = 1  si Z1 > 0
             0  si Z1 ≤ 0
```

### 4.2 Capa de Salida

**Paso 3 — Pre-activación:**
```
Z2 = W2 · A1 + b2
```

- W2: matriz (10 × 64)
- A1: activaciones ocultas (64,)
- Z2: resultado (10,) — puntuación bruta para cada dígito

**Paso 4 — Activación Softmax (numéricamente estable):**
```
Z2_estable = Z2 - max(Z2)          ← estabilidad numérica
A2_i = exp(Z2_estable_i) / Σ exp(Z2_estable_j)    para j = 0..9
```

Softmax convierte las puntuaciones brutas en probabilidades que suman exactamente 1:

```
Σ A2_i = 1.0     para i = 0..9
```

El índice con mayor probabilidad es la predicción:
```
ŷ = argmax(A2)
```

---

## 5. Función de Pérdida (Cross-Entropy Loss)

La función de pérdida mide qué tan equivocada está la red. Para la clasificación multiclase usamos **Entropía Cruzada**:

```
L = -Σ y_i · log(A2_i)     para i = 0..9
```

Donde `y_i` es la etiqueta one-hot:
- y = 5 → y_onehot = [0, 0, 0, 0, 0, **1**, 0, 0, 0, 0]

Esto simplifica a:
```
L = -log(A2[clase_correcta])
```

**Clipping para estabilidad numérica:**
```python
A2 = np.clip(A2, 1e-15, 1.0)   # evita log(0) = -inf
```

---

## 6. Retropropagación (Backpropagation)

El objetivo de backpropagation es calcular el gradiente de la pérdida L respecto a cada peso, usando la **regla de la cadena** del cálculo diferencial.

### 6.1 Gradiente combinado Softmax + Cross-Entropy

Esta es la fórmula más elegante del proyecto. Cuando se combina la derivada de la Cross-Entropy con la derivada de Softmax, el resultado simplifica a:

```
∂L/∂Z2 = dZ2 = A2 - y_onehot
```

**Demostración:**

Para j = clase correcta (j = 5):
```
∂L/∂Z2_j = A2_j - 1
```

Para j ≠ clase correcta:
```
∂L/∂Z2_j = A2_j - 0 = A2_j
```

En forma vectorial: `dZ2 = A2 - y_onehot`

### 6.2 Gradientes de la Capa de Salida

```
dZ2 = A2 - y_onehot                    → (10,)

dW2 = dZ2 ⊗ A1 = outer(dZ2, A1)       → (10, 64)
      dW2[i,j] = dZ2[i] · A1[j]

db2 = dZ2                               → (10,)
```

**Álgebra:** `dW2` es el **producto exterior** (outer product) de dos vectores. Cada elemento dW2[i,j] indica cuánto contribuyó la conexión W2[i,j] al error.

### 6.3 Propagación del Error hacia la Capa Oculta

```
dA1 = W2ᵀ · dZ2        → (64,)
```

Donde W2ᵀ es la transpuesta de W2 (64×10). Esto "retropropaga" el error de la salida hacia la capa oculta.

### 6.4 Gradientes de la Capa Oculta

La derivada de ReLU actúa como una "compuerta": solo pasa el gradiente donde Z1 > 0.

```
dZ1 = dA1 ⊙ ReLU'(Z1)                 → (64,)
      dZ1[i] = dA1[i]  si Z1[i] > 0
               0         si Z1[i] ≤ 0

dW1 = dZ1 ⊗ X = outer(dZ1, X)         → (64, 784)
      dW1[i,j] = dZ1[i] · X[j]

db1 = dZ1                               → (64,)
```

### 6.5 Resumen de la Regla de la Cadena

```
L → ∂L/∂Z2 → ∂L/∂W2, ∂L/∂b2
             → ∂L/∂A1 → ∂L/∂Z1 → ∂L/∂W1, ∂L/∂b1
```

---

## 7. Descenso del Gradiente (Actualización de Pesos)

Una vez calculados los gradientes, los pesos se actualizan restando una fracción (tasa de aprendizaje η = 0.01) del gradiente:

```
W1 ← W1 - η · dW1
b1 ← b1 - η · db1
W2 ← W2 - η · dW2
b2 ← b2 - η · db2
```

**Intuición:** El gradiente señala la dirección de mayor aumento de la pérdida. Restarlo mueve los pesos en la dirección de menor pérdida.

---

## 8. Mini-batch Gradient Descent

El entrenamiento no procesa las 60,000 imágenes juntas. Las agrupa en lotes (batches) de 32:

```
Por cada época:
  1. Mezclar aleatoriamente el dataset (shuffle)
  2. Dividir en batches de 32
  3. Por cada batch:
     - Forward → calcula A2 y la pérdida promedio
     - Backward → calcula gradientes
     - Update → actualiza W1, b1, W2, b2
  4. Registrar loss y accuracy del epoch
```

**Ventajas:** Más rápido que actualizar muestra por muestra, más estable que usar todo el dataset a la vez.

---

## 9. Pipeline de Procesamiento de Imagen

Para predecir un dígito capturado por la cámara, se ejecutan 7 pasos:

```
Frame (base64)
    │
    ▼ decode_frame()
BGR (H×W×3)
    │
    ▼ cvtColor(BGR → GRAY)
Escala de grises (H×W)
    │
    ▼ GaussianBlur(kernel 5×5)
Imagen suavizada
    │
    ▼ threshold(Otsu + BINARY_INV)
Binaria: dígito=blanco, fondo=negro
    │
    ▼ findContours() → boundingRect()
Región de interés (ROI) recortada
    │
    ▼ padding cuadrado + resize(28×28, INTER_AREA)
Imagen 28×28 normalizada
    │
    ▼ flatten() / 255.0
Vector (784,) en [0.0, 1.0]
    │
    ▼ MLP.predict_proba(vector)
Probabilidades Softmax (10,)
```

**Por qué BINARY_INV:** MNIST usa dígito blanco sobre fondo negro. Si la cámara captura dígito oscuro sobre papel blanco, la inversión corrige la polaridad.

**Por qué INTER_AREA:** Al reducir de alta resolución a 28×28, INTER_AREA calcula el promedio de los píxeles del área correspondiente, manteniendo mejor calidad que INTER_LINEAR o INTER_NEAREST.

---

## 10. Arquitectura de la Aplicación Web

### 10.1 Comunicación en Tiempo Real

```
Navegador                    Servidor Flask
    │                              │
    │──── POST /train/start ──────►│ Hilo daemon de entrenamiento
    │                              │     │ epoch 1 done
    │◄─── training_update (WS) ───│◄────┘
    │      { epoch, loss, acc }    │
    │◄─── training_update (WS) ───│ epoch 2, 3, ...
    │                              │
    │──── POST /train/step ───────►│ train_single_step()
    │◄─── { activaciones, pesos }─│ respuesta JSON inmediata
    │                              │
    │──── POST /predict ──────────►│ preprocess + predict_proba
    │◄─── { digit, probs, thumb }─│
```

### 10.2 Modo Paso a Paso (Educativo)

El endpoint `/train/step` ejecuta exactamente **una iteración** de entrenamiento sobre la siguiente muestra del dataset. Retorna:

```json
{
  "iteration": 1,
  "loss": 1.7924,
  "accuracy": 1,
  "prediction": 5,
  "probabilities": [0.1177, 0.1021, ...],
  "activations": {
    "A1": [0.6481, 0.6299, 0.0, ...],
    "A2": [0.1177, 0.1021, ...]
  },
  "weights_sample": {
    "W2_sample": [[...]]
  }
}
```

Estos datos animan la visualización: los nodos se iluminan con la activación real, las conexiones cambian de color y grosor.

### 10.3 Visualización de la Red

La clase `NetViz` (Canvas API) dibuja:
- **Nodos de entrada (azul):** 20 representativos de los 784
- **Nodos ocultos (verde):** 64 neuronas, brillo ∝ activación A1
- **Nodos de salida (rojo):** 10, brillo ∝ A2 (probabilidad)
- **Conexiones:** Azul si peso > 0, Rojo si peso < 0, grosor ∝ |peso|

---

## 11. Reportes Generados Automáticamente

### 11.1 Bitácora de Matrices (`weight_matrix.log`)

El sistema registra la sección `W1[0:5, 350:355]` (zona central activa de la imagen 28×28) exactamente en las iteraciones 1, 50 y 100. Esto evidencia cómo los pesos evolucionan con el entrenamiento.

**Por qué columnas 350-355:** Las columnas 0-27 y similares corresponden a los bordes de la imagen MNIST, que siempre son 0. Los píxeles centrales (columnas ~300-500) son los que realmente reciben señal y actualizan sus pesos.

### 11.2 Gráfica de Pérdida (`loss_chart.png`)

La interfaz web muestra en tiempo real la curva de pérdida y accuracy por epoch usando Chart.js. El botón "Exportar PNG" guarda la gráfica en `reports/loss_chart.png`.

---

## 12. Resultados Obtenidos

Entrenando con los 60,000 ejemplos de MNIST (lr=0.01, batch=32):

| Época | Loss | Accuracy |
|---|---|---|
| 1 | ~0.39 | 92.63% |
| 2 | ~0.27 | 96.37% |
| 5 | ~0.20 | 97.5% (estimado) |

La red alcanza ~96% de precisión en apenas 2 épocas, demostrando que el algoritmo implementado desde cero es correcto y eficiente.

---

## 13. Decisiones de Diseño Clave

| Decisión | Alternativa rechazada | Razón |
|---|---|---|
| ReLU en capa oculta | Sigmoid | Sigmoid tiene gradiente casi cero en extremos → vanishing gradient |
| Softmax + Cross-Entropy combinados | Derivadas separadas | La combinación simplifica dZ2 = A2 - y, más eficiente y numéricamente estable |
| Mini-batch de 32 | SGD puro (batch=1) | Balance entre velocidad y estabilidad |
| He init para W1 | Random uniforme | Varianza calibrada para ReLU → entrenamiento más rápido |
| gevent como async | eventlet | Eventlet está en deprecation warning en Flask-SocketIO moderno |
| Columnas 350-355 en logger | Columnas 0-5 | Los bordes de MNIST son siempre 0 → nunca recibirían gradiente |

---

## 14. Estructura de Archivos del Motor Neuronal

```
neural_network/
├── initializer.py    He + Xavier initialization, seed=42
├── activations.py    relu, relu_derivative, softmax (estable)
├── loss.py           one_hot, cross_entropy_loss, accuracy
├── network.py        Clase MLP: forward, backward, update, predict
├── logger.py         WeightLogger: bitácora en iter {1, 50, 100}
└── debug.py          debug_single_sample: imprime con 4 decimales
```

Cada archivo tiene una sola responsabilidad. Los módulos no tienen dependencias circulares.
