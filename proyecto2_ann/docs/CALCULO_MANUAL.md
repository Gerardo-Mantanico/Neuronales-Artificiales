# Documento de Cálculo Manual
## ANN desde Cero — Proyecto 2 | Inteligencia Artificial 1 | USAC CUNOC
**Ing. Daniel González | Autor: Gerardo Mantanico**

> **Verificación:** Todos los valores aquí mostrados fueron generados por el modo `debug` del programa (`debug.py`) con `seed=42`, y pueden reproducirse exactamente ejecutando:
> ```bash
> venv/bin/python -c "
> import sys; sys.path.insert(0,'.')
> from neural_network.network import MLP
> from neural_network.debug import debug_single_sample
> from data.mnist_loader import load_mnist
> X_train, y_train, _, _ = load_mnist()
> net = MLP(lr=0.01, seed=42)
> debug_single_sample(net, X_train[0], int(y_train[0]))
> "
> ```

---

## Configuración de la Red (Iteración 0 — Pesos Iniciales)

| Parámetro | Valor |
|---|---|
| Imagen | MNIST muestra #0 (dígito **5**) |
| Tasa de aprendizaje (η) | 0.01 |
| Arquitectura | 784 → 64 → 10 |
| Inicialización W1 | He: σ = √(2/784) ≈ 0.0505, seed=42 |
| Inicialización W2 | Xavier: σ = √(1/64) = 0.125, seed=42 |
| b1, b2 | Todos ceros |

---

## PASO 1 — Entrada (X)

La imagen MNIST del dígito **5** es una matriz 28×28 de píxeles en escala de grises, aplanada a un vector de 784 valores normalizados en [0.0, 1.0].

**Primeros 10 valores del vector X:**
```
X[0:10] = [0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000]
```
(Los bordes de las imágenes MNIST son siempre 0 — el dígito está centrado.)

**Píxeles no-cero:** 166 de 784 tienen valor mayor que 0.

**Top 5 píxeles con mayor contribución al nodo oculto #0:**

| Índice pixel | X[i] | W1[0,i] | Contribución = X[i] × W1[0,i] |
|---|---|---|---|
| 209 | 0.9922 | 0.1946 | **+0.1931** |
| 654 | 0.9922 | 0.1300 | **+0.1290** |
| 575 | 0.9922 | −0.1117 | **−0.1108** |
| 234 | 0.9922 | 0.1083 | **+0.1074** |
| 236 | 0.9922 | −0.1023 | **−0.1015** |

---

## PASO 2 — Forward Propagation: Capa Oculta

### 2a. Pre-activación Z1 = W1 · X + b1

Fórmula para el nodo oculto j:
```
Z1[j] = Σ(W1[j,i] × X[i])  para i = 0..783   +  b1[j]
```

**Pesos iniciales usados (W1[0, :5]):**
```
W1[0,:5] = [0.0251  -0.0070  0.0327  0.0769  -0.0118]
b1[0] = 0.0000
```

**Cálculo de Z1[0] (primer nodo oculto):**
```
Z1[0] = W1[0,0]×X[0] + W1[0,1]×X[1] + ... + W1[0,783]×X[783] + b1[0]
Z1[0] = 0.0251×0.0000 + (-0.0070)×0.0000 + ... + (términos de pixeles centrales) + 0
Z1[0] = 0.6481
```

**Primeros 5 valores de Z1:**
```
Z1[:5] = [0.6481   0.6299   -0.3533   -0.3817   0.2758]
```

**Vector Z1 completo (64 valores):**
```
Z1 = [ 0.6481,  0.6299, -0.3533, -0.3817,  0.2758,  0.4008, -0.3472,  0.2153,
      -0.2033, -0.4541, -0.6313, -0.4262, -0.3842,  0.4658,  0.2967,  0.3320,
       0.0837, -0.4088,  0.0668, -0.8676,  0.5192, -0.3851, -0.4054, -0.3091,
       0.7378,  0.4934,  0.2393, -0.1467, -0.3120, -0.6704, -0.8449, -0.2498,
       0.0392, -0.4788, -0.2381, -0.4312, -0.0580, -0.5089, -1.1727,  0.5692,
      -0.7044,  0.1751,  0.0781, -0.8016, -1.3385,  0.4476, -0.0970, -0.0720,
      -0.1389,  0.3013,  0.5834,  0.9147,  0.0306, -0.0109,  0.0784,  0.1436,
       0.0427, -0.5209,  0.7520, -0.5032, -1.1966,  0.5098, -0.1529,  0.4849 ]
```

### 2b. Activación ReLU: A1 = max(0, Z1)

```
A1[j] = Z1[j]   si Z1[j] > 0
         0        si Z1[j] ≤ 0
```

**Cálculo explícito para los 5 primeros:**
```
A1[0] = max(0,  0.6481) = 0.6481  ✓ activo
A1[1] = max(0,  0.6299) = 0.6299  ✓ activo
A1[2] = max(0, -0.3533) = 0.0000  ✗ inactivo
A1[3] = max(0, -0.3817) = 0.0000  ✗ inactivo
A1[4] = max(0,  0.2758) = 0.2758  ✓ activo
```

**Nodos activos:** 29 de 64 (45% — razonable para ReLU)

**Vector A1 completo (64 valores):**
```
A1 = [0.6481, 0.6299, 0.0000, 0.0000, 0.2758, 0.4008, 0.0000, 0.2153,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4658, 0.2967, 0.3320,
      0.0837, 0.0000, 0.0668, 0.0000, 0.5192, 0.0000, 0.0000, 0.0000,
      0.7378, 0.4934, 0.2393, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0392, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5692,
      0.0000, 0.1751, 0.0781, 0.0000, 0.0000, 0.4476, 0.0000, 0.0000,
      0.0000, 0.3013, 0.5834, 0.9147, 0.0306, 0.0000, 0.0784, 0.1436,
      0.0427, 0.0000, 0.7520, 0.0000, 0.0000, 0.5098, 0.0000, 0.4849]
```

---

## PASO 3 — Forward Propagation: Capa de Salida

### 3a. Pre-activación Z2 = W2 · A1 + b2

Fórmula para el nodo de salida k:
```
Z2[k] = Σ(W2[k,j] × A1[j])  para j = 0..63   +  b2[k]
```

**Pesos iniciales W2[0,:5]:**
```
W2[0,:5] = [0.1168  0.0027  0.0892  -0.1025  -0.0416]
b2[0] = 0.0000
```

**Cálculo de Z2[0] (nodo para dígito "0"):**
```
Z2[0] = 0.1168×0.6481 + 0.0027×0.6299 + 0.0892×0.0000 + (-0.1025)×0.0000 + (-0.0416)×0.2758 + ...
Z2[0] = 0.0757 + 0.0017 + 0.0000 + 0.0000 + (-0.0115) + (resto)
Z2[0] = 0.2091
```

**Vector Z2 completo (10 valores — uno por dígito):**
```
Z2 = [0.2091, 0.0668, -0.0681, -0.0039, -0.0692, 0.5562, -0.2685, -0.2321, -0.1217, 0.1119]
     [  "0"    "1"      "2"      "3"      "4"      "5"      "6"      "7"      "8"      "9"  ]
```

### 3b. Activación Softmax: A2

**Paso 1 — Estabilidad numérica (restar el máximo):**
```
max(Z2) = 0.5562   (corresponde al dígito "5")
Z2_estable = Z2 - 0.5562
Z2_estable = [-0.3471, -0.4894, -0.6243, -0.5601, -0.6254, 0.0000, -0.8247, -0.7883, -0.6779, -0.4443]
```

**Paso 2 — Exponencial:**
```
exp(Z2_estable) = [0.7069, 0.6130, 0.5357, 0.5712, 0.5351, 1.0000, 0.4384, 0.4547, 0.5079, 0.6414]
```

**Paso 3 — Suma total:**
```
Σ exp(Z2_estable) = 0.7069 + 0.6130 + 0.5357 + 0.5712 + 0.5351 + 1.0000 + 0.4384 + 0.4547 + 0.5079 + 0.6414
Σ = 6.0043
```

**Paso 4 — Dividir para obtener probabilidades:**
```
A2[k] = exp(Z2_estable[k]) / Σ
```

**Vector A2 (probabilidades Softmax):**
```
A2 = [0.1177, 0.1021, 0.0892, 0.0951, 0.0891, 0.1666, 0.0730, 0.0757, 0.0846, 0.1068]
     [ "0"     "1"     "2"     "3"     "4"     "5"     "6"     "7"     "8"     "9"  ]
```

**Verificación:** sum(A2) = 1.0000000000 ✓

**Predicción:** argmax(A2) = dígito **5** ← CORRECTO (dígito real = 5)

---

## PASO 4 — Función de Pérdida (Cross-Entropy)

La etiqueta real es y = 5, codificada como one-hot:
```
y_onehot = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```

La pérdida es:
```
L = -Σ y_i × log(A2[i])  = -1 × log(A2[5])    (el resto son cero)
L = -log(0.1666)
L = 1.7924
```

**La red acierta el dígito en la primera iteración,** pero con baja confianza (16.66%). Después del entrenamiento, A2[5] debería acercarse a 1.0.

---

## PASO 5 — Retropropagación: Gradientes de la Capa de Salida

### 5a. Gradiente combinado Softmax + Cross-Entropy

```
dZ2 = A2 - y_onehot
```

**Cálculo elemento a elemento:**
```
dZ2[0] = A2[0] - y[0] = 0.1177 - 0 = +0.1177
dZ2[1] = A2[1] - y[1] = 0.1021 - 0 = +0.1021
dZ2[2] = A2[2] - y[2] = 0.0892 - 0 = +0.0892
dZ2[3] = A2[3] - y[3] = 0.0951 - 0 = +0.0951
dZ2[4] = A2[4] - y[4] = 0.0891 - 0 = +0.0891
dZ2[5] = A2[5] - y[5] = 0.1666 - 1 = -0.8334  ← clase correcta (negativo → reduce pérdida)
dZ2[6] = A2[6] - y[6] = 0.0730 - 0 = +0.0730
dZ2[7] = A2[7] - y[7] = 0.0757 - 0 = +0.0757
dZ2[8] = A2[8] - y[8] = 0.0846 - 0 = +0.0846
dZ2[9] = A2[9] - y[9] = 0.1068 - 0 = +0.1068
```

```
dZ2 = [0.1177, 0.1021, 0.0892, 0.0951, 0.0891, -0.8334, 0.0730, 0.0757, 0.0846, 0.1068]
```

### 5b. Gradiente de W2 (producto exterior)

```
dW2 = dZ2 ⊗ A1 = outer(dZ2, A1)
dW2[i,j] = dZ2[i] × A1[j]
```

**Cálculo explícito de dW2[0, 0:3]:**
```
dW2[0,0] = dZ2[0] × A1[0] = 0.1177 × 0.6481 = 0.0763
dW2[0,1] = dZ2[0] × A1[1] = 0.1177 × 0.6299 = 0.0741
dW2[0,2] = dZ2[0] × A1[2] = 0.1177 × 0.0000 = 0.0000
```

**dW2[5, 0] (conexión del nodo de salida "5" al nodo oculto 0):**
```
dW2[5,0] = dZ2[5] × A1[0] = (-0.8334) × 0.6481 = -0.5401
```

**db2:**
```
db2 = dZ2 = [0.1177, 0.1021, 0.0892, 0.0951, 0.0891, -0.8334, 0.0730, 0.0757, 0.0846, 0.1068]
```

---

## PASO 6 — Retropropagación: Gradientes de la Capa Oculta

### 6a. Propagar el error hacia la capa oculta

```
dA1 = W2ᵀ · dZ2
```

**Cálculo de dA1[0] (primer nodo oculto):**
```
dA1[0] = Σ(W2[k,0] × dZ2[k])  para k = 0..9
dA1[0] = W2[0,0]×0.1177 + W2[1,0]×0.1021 + ... + W2[5,0]×(-0.8334) + ...
dA1[0] = -0.1364
```

**Primeros 5 valores de dA1:**
```
dA1[:5] = [-0.1364  -0.0165  0.1401  0.0688  -0.1053]
```

### 6b. Aplicar derivada de ReLU

La derivada de ReLU actúa como compuerta: pasa el gradiente donde Z1 > 0, lo bloquea donde Z1 ≤ 0.

```
dZ1[j] = dA1[j] × ReLU'(Z1[j])
         = dA1[j] × 1   si Z1[j] > 0
         = 0             si Z1[j] ≤ 0
```

**Cálculo para los 5 primeros:**
```
dZ1[0] = -0.1364 × 1 = -0.1364   (Z1[0]=0.6481 > 0 → activo)
dZ1[1] = -0.0165 × 1 = -0.0165   (Z1[1]=0.6299 > 0 → activo)
dZ1[2] =  0.1401 × 0 =  0.0000   (Z1[2]=-0.3533 < 0 → bloqueado)
dZ1[3] =  0.0688 × 0 =  0.0000   (Z1[3]=-0.3817 < 0 → bloqueado)
dZ1[4] = -0.1053 × 1 = -0.1053   (Z1[4]=0.2758 > 0 → activo)
```

```
dZ1[:5] = [-0.1364  -0.0165  0.0000  0.0000  -0.1053]
```

### 6c. Gradiente de W1 (producto exterior)

```
dW1 = dZ1 ⊗ X = outer(dZ1, X)
dW1[j,i] = dZ1[j] × X[i]
```

**Cálculo de dW1[0, 0:3]:**
```
dW1[0,0] = dZ1[0] × X[0] = -0.1364 × 0.0000 = 0.0000
dW1[0,1] = dZ1[0] × X[1] = -0.1364 × 0.0000 = 0.0000
dW1[0,2] = dZ1[0] × X[2] = -0.1364 × 0.0000 = 0.0000
```

**Nota:** Los primeros píxeles de X son 0 (bordes de MNIST), por eso dW1 comienza en 0.

**db1:**
```
db1[:5] = [-0.1364  -0.0165  0.0000  0.0000  -0.1053]
```

---

## PASO 7 — Actualización de Pesos (Gradiente Descendente, η = 0.01)

```
W ← W - η × dW
```

### Actualización de W2[0,0]:
```
W2[0,0]_nuevo = W2[0,0]_anterior - η × dW2[0,0]
W2[0,0]_nuevo = 0.1168 - 0.01 × 0.0763
W2[0,0]_nuevo = 0.1168 - 0.0008
W2[0,0]_nuevo = 0.1160
```

### Actualización de W2[5,0] (conexión más importante — clase correcta):
```
W2[5,0]_nuevo = W2[5,0]_anterior - η × dW2[5,0]
W2[5,0]_nuevo = 0.1663 - 0.01 × (-0.5401)
W2[5,0]_nuevo = 0.1663 + 0.0054
W2[5,0]_nuevo = 0.1717
```

**Interpretación:** El peso W2[5,0] aumentó porque dZ2[5] fue negativo (la red subestimó el dígito "5"), y A1[0] fue positivo. El gradiente descendente corrige esto aumentando W2[5,0] para que en el siguiente paso la red favorezca más la clase "5".

### Actualización de W1[0,0]:
```
W1[0,0]_nuevo = W1[0,0]_anterior - η × dW1[0,0]
W1[0,0]_nuevo = 0.0251 - 0.01 × 0.0000
W1[0,0]_nuevo = 0.0251   (sin cambio — X[0] = 0)
```

---

## PASO 8 — Verificación con el Modo Debug del Programa

Ejecutando `debug_single_sample(net, X_train[0], 5)` con `seed=42`, el programa produce exactamente:

```
════════════════════════════════════════════════════════════
  MODO DEBUG — Una iteración (sin modificar pesos)
════════════════════════════════════════════════════════════

[ENTRADA]
  X[:10]     = [0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000]
  y_real     = 5
  y_onehot   = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

[FORWARD — CAPA OCULTA]
  W1[0,:5]   = [0.0251  -0.0070  0.0327  0.0769  -0.0118]
  b1[0]      = 0.0000
  Z1[:5]     = [0.6481  0.6299  -0.3533  -0.3817  0.2758]
  A1[:5]     = [0.6481  0.6299  0.0000  0.0000  0.2758]
  Nodos activos (Z1>0): 29 / 64

[FORWARD — CAPA DE SALIDA]
  W2[0,:5]   = [0.1168  0.0027  0.0892  -0.1025  -0.0416]
  b2[0]      = 0.0000
  Z2         = [0.2091  0.0668  -0.0681  -0.0039  -0.0692  0.5562  -0.2685  -0.2321  -0.1217  0.1119]
  A2         = [0.1177  0.1021  0.0892  0.0951  0.0891  0.1666  0.0730  0.0757  0.0846  0.1068]
  sum(A2)    = 1.0000000000  ← debe ser 1.0

[PÉRDIDA]
  Loss       = 1.7924
  Predicción = 5  (correcto: 5)

[BACKWARD — CAPA DE SALIDA]
  dZ2        = [0.1177  0.1021  0.0892  0.0951  0.0891  -0.8334  0.0730  0.0757  0.0846  0.1068]
  dW2[0,:3]  = [0.0763  0.0741  0.0000]
  db2[0]     = 0.1177

[BACKWARD — CAPA OCULTA]
  dA1[:5]    = [-0.1364  -0.0165  0.1401  0.0688  -0.1053]
  dZ1[:5]    = [-0.1364  -0.0165  0.0000  0.0000  -0.1053]
  dW1[0,:3]  = [-0.0000  -0.0000  -0.0000]
  db1[0]     = -0.1364

[ACTUALIZACIÓN (lr=0.01)]
  W2[0,0]     antes  = 0.1168
  dW2[0,0]           = 0.0763
  W2[0,0]     después = 0.1160

  W1[0,0]     antes  = 0.0251
  dW1[0,0]           = -0.0000
  W1[0,0]     después = 0.0251
```

**Concordancia:** Todos los valores del cálculo manual en los Pasos 1–7 coinciden con la salida del programa con precisión de 4 decimales. ✓

---

## Resumen de la Iteración Completa

| Variable | Valor calculado manualmente | Valor del programa | Diferencia |
|---|---|---|---|
| Z1[0] | 0.6481 | 0.6481 | 0.0000 ✓ |
| A1[0] | 0.6481 | 0.6481 | 0.0000 ✓ |
| Z1[2] | −0.3533 | −0.3533 | 0.0000 ✓ |
| A1[2] | 0.0000 | 0.0000 | 0.0000 ✓ |
| Z2[5] | 0.5562 | 0.5562 | 0.0000 ✓ |
| A2[5] | 0.1666 | 0.1666 | 0.0000 ✓ |
| sum(A2) | 1.0000 | 1.0000 | 0.0000 ✓ |
| Loss | 1.7924 | 1.7924 | 0.0000 ✓ |
| dZ2[5] | −0.8334 | −0.8334 | 0.0000 ✓ |
| dW2[0,0] | 0.0763 | 0.0763 | 0.0000 ✓ |
| dZ1[0] | −0.1364 | −0.1364 | 0.0000 ✓ |
| W2[0,0] actualizado | 0.1160 | 0.1160 | 0.0000 ✓ |
| W2[5,0] actualizado | 0.1717 | 0.1717 | 0.0000 ✓ |
