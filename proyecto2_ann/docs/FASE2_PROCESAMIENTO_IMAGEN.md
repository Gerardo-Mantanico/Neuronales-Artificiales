# 🖼️ Fase 2 — Procesamiento de Imagen
## Desglose Técnico Completo: Cámara → 28×28 → Vector

**Proyecto 2 — Inteligencia Artificial 1 | USAC CUNOC**

---

## Índice

1. [Qué problema resuelve esta fase](#1-qué-problema-resuelve-esta-fase)
2. [Mapa de archivos y responsabilidades](#2-mapa-de-archivos-y-responsabilidades)
3. [Pipeline completo de un vistazo](#3-pipeline-completo-de-un-vistazo)
4. [Paso 1 — Captura del frame (cámara → base64 → BGR)](#4-paso-1--captura-del-frame-cámara--base64--bgr)
5. [Paso 2 — Conversión a escala de grises](#5-paso-2--conversión-a-escala-de-grises)
6. [Paso 3 — Suavizado con Blur Gaussiano](#6-paso-3--suavizado-con-blur-gaussiano)
7. [Paso 4 — Umbralización de Otsu](#7-paso-4--umbralización-de-otsu)
8. [Paso 5 — Detección de contornos y bounding box](#8-paso-5--detección-de-contornos-y-bounding-box)
9. [Paso 6 — Cuadrado y centrado del dígito](#9-paso-6--cuadrado-y-centrado-del-dígito)
10. [Paso 7 — Redimensionar a 28×28](#10-paso-7--redimensionar-a-2828)
11. [Paso 8 — Aplanar y normalizar → vector (784,)](#11-paso-8--aplanar-y-normalizar--vector-784)
12. [Integración: función `preprocess_image`](#12-integración-función-preprocess_image)
13. [Flujo de datos con formas en cada etapa](#13-flujo-de-datos-con-formas-en-cada-etapa)
14. [Consideraciones para coincidir con MNIST](#14-consideraciones-para-coincidir-con-mnist)
15. [Estructura de funciones — API pública](#15-estructura-de-funciones--api-pública)

---

## 1. Qué Problema Resuelve Esta Fase

La red neuronal espera **exactamente** un vector de `784` números flotantes en el rango `[0.0, 1.0]`, correspondientes a los píxeles de una imagen en **escala de grises de 28×28 píxeles**.

La cámara web entrega un frame **a color, de alta resolución y con fondo ruidoso**. Esta fase es el puente:

```
Cámara web                              Red Neuronal
(ej. 1280×720, color RGB)  ──────►  Vector (784,) normalizado
```

Sin esta transformación, la red no puede recibir la imagen. Y si la transformación no replica las mismas condiciones que MNIST (fondo blanco, dígito negro, centrado), la predicción será incorrecta aunque la red esté bien entrenada.

---

## 2. Mapa de Archivos y Responsabilidades

```
image_processing/
│
├── __init__.py       ← Exporta las 3 funciones públicas del módulo
├── capture.py        ← Paso 1: decodifica el frame base64 del navegador → imagen BGR
├── roi_detector.py   ← Pasos 2–5: grises → blur → Otsu → contornos → recorte ROI
└── preprocessor.py   ← Pasos 6–8: cuadrado → 28×28 → aplanar → normalizar
```

| Archivo | Función pública | Entrada | Salida |
|---|---|---|---|
| `capture.py` | `decode_frame(b64)` | `str` base64 | `ndarray (H,W,3)` BGR |
| `roi_detector.py` | `detect_digit_region(bgr)` | `ndarray (H,W,3)` | `ndarray (h,w)` binaria |
| `preprocessor.py` | `preprocess_image(b64)` | `str` base64 | `(ndarray (784,), ndarray (28,28))` |

---

## 3. Pipeline Completo de un Vistazo

```
[Cámara Web]
     │  stream de video en el navegador
     ▼
[JS: canvas.toDataURL('image/jpeg')]
     │  frame como string base64
     │  "data:image/jpeg;base64,/9j/4AAQ..."
     ▼
─────────────────────────────────── capture.py ───────
decode_frame(base64_string)
  1. Eliminar prefijo "data:image/jpeg;base64,"
  2. base64.b64decode → bytes crudos
  3. np.frombuffer → array de uint8
  4. cv2.imdecode  → imagen BGR (H, W, 3)
──────────────────────────────────────────────────────
     │  imagen BGR (ej. 720×1280×3)
     ▼
─────────────────────────────── roi_detector.py ──────
detect_digit_region(imagen_bgr)
  5. cv2.cvtColor  → escala de grises   (H, W)
  6. GaussianBlur  → suavizado          (H, W)
  7. threshold Otsu → binaria invertida (H, W)  0/255
  8. findContours  → lista de contornos
  9. max por área  → contorno del dígito
  10. boundingRect → (x, y, w, h)
  11. margen 15%   → recorte con contexto
──────────────────────────────────────────────────────
     │  ROI binaria (h_roi, w_roi)   valores: 0 o 255
     ▼
──────────────────────────────── preprocessor.py ─────
preprocess_image (continúa)
  12. Cuadrado: padding para que h == w
  13. Centrar el dígito dentro del cuadrado
  14. cv2.resize → 28×28  (interpolación INTER_AREA)
  15. flatten()  → (784,)  valores: 0..255
  16. / 255.0    → (784,)  valores: 0.0..1.0  float64
──────────────────────────────────────────────────────
     │
     ▼
Vector X (784,)  →  MLP.forward(X)  →  predicción del dígito
```

---

## 4. Paso 1 — Captura del Frame (Cámara → Base64 → BGR)

### Qué hace el navegador (JavaScript — `camera.js`)

```javascript
// El navegador accede a la cámara con la API estándar
stream = await navigator.mediaDevices.getUserMedia({ video: true })
videoEl.srcObject = stream

// Al capturar: copia el frame del <video> a un <canvas> temporal
tmpCanvas.getContext("2d").drawImage(videoEl, 0, 0)

// Convierte el canvas a base64 JPEG (calidad 0.85)
const base64 = tmpCanvas.toDataURL("image/jpeg", 0.85)
// resultado: "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."
```

---

### Qué hace el backend (Python — `capture.py`)

```python
def decode_frame(base64_string: str) -> np.ndarray:
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]   # quitar prefijo MIME

    datos  = base64.b64decode(base64_string)             # str → bytes
    buffer = np.frombuffer(datos, dtype=np.uint8)        # bytes → array uint8
    imagen = cv2.imdecode(buffer, cv2.IMREAD_COLOR)      # array → imagen BGR
    return imagen
```

### Pseudocódigo

```
FUNCIÓN decode_frame(base64_string):

    SI base64_string contiene ",":
        base64_string = parte después de la primera ","
        # elimina "data:image/jpeg;base64,"

    bytes_crudos = decodificar_base64(base64_string)
    # transforma caracteres ASCII → bytes originales del JPEG

    buffer_uint8 = interpretar_como_enteros_sin_signo(bytes_crudos)
    # el JPEG comprimido como array de bytes numéricos

    imagen_bgr = decodificar_imagen_comprimida(buffer_uint8)
    # OpenCV descomprime el JPEG → matriz de píxeles (H, W, 3)
    # canal 0: Azul, canal 1: Verde, canal 2: Rojo  (orden BGR de OpenCV)

    RETORNAR imagen_bgr   # forma: (H, W, 3), dtype: uint8, valores: 0–255
```

### Por qué BGR y no RGB

OpenCV usa el orden de canales **BGR** (azul-verde-rojo) por razones históricas. No afecta a los pasos siguientes porque el paso inmediato es convertir a escala de grises, donde el orden de canales no importa.

### Estructura de la función

```python
# capture.py

def decode_frame(base64_string: str) -> np.ndarray:
    """
    Entrada:  string base64 (con o sin prefijo 'data:image/...')
    Salida:   imagen BGR  (H, W, 3)  dtype=uint8
    """
```

---

## 5. Paso 2 — Conversión a Escala de Grises

### Por qué

La red neuronal fue entrenada con imágenes en **escala de grises**. El color no aporta información para reconocer dígitos manuscritos y triplica el tamaño del dato innecesariamente.

### Matemática de la conversión

OpenCV usa la ponderación estándar ITU-R BT.601:

```
Gray = 0.114 × B  +  0.587 × G  +  0.299 × R
```

Los coeficientes reflejan la sensibilidad del ojo humano: el verde aporta más luminancia percibida que el rojo, y el azul aporta menos.

### Código y pseudocódigo

```python
gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
```

```
FUNCIÓN convertir_a_grises(imagen_bgr):

    PARA cada píxel (i, j):
        B = imagen_bgr[i, j, 0]    # canal azul   (0–255)
        G = imagen_bgr[i, j, 1]    # canal verde  (0–255)
        R = imagen_bgr[i, j, 2]    # canal rojo   (0–255)

        gris[i, j] = round(0.114*B + 0.587*G + 0.299*R)

    RETORNAR gris   # forma: (H, W), dtype: uint8, valores: 0–255
```

### Transformación de dimensiones

```
(H, W, 3)  →  (H, W)
  BGR           grises
  3 canales     1 canal
```

---

## 6. Paso 3 — Suavizado con Blur Gaussiano

### Por qué

La cámara introduce **ruido de imagen**: pequeñas variaciones de píxeles causadas por iluminación irregular, compresión JPEG y sensor electrónico. Si se umbraliza sin suavizar, el ruido genera contornos falsos que confunden la detección del dígito.

El blur gaussiano promedia cada píxel con sus vecinos, suavizando esas variaciones.

### Matemática del filtro gaussiano

El filtro gaussiano aplica una **convolución** de la imagen con un kernel de forma campana (distribución gaussiana 2D):

```
G(x, y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)

Kernel 5×5 con σ≈1 (aproximación):
┌─────────────────────────────────┐
│  2   4   5   4   2  │  / 159   │
│  4   9  12   9   4  │          │
│  5  12  15  12   5  │          │
│  4   9  12   9   4  │          │
│  2   4   5   4   2  │          │
└─────────────────────────────────┘

Para cada píxel (i,j):
  suavizado[i,j] = suma_ponderada(vecindad_5x5, kernel) / 159
```

### Código y pseudocódigo

```python
blur = cv2.GaussianBlur(gris, (5, 5), 0)
# kernel 5×5, sigma calculado automáticamente por OpenCV (sigmaX=0)
```

```
FUNCIÓN blur_gaussiano(gris, tamaño_kernel=5):

    PARA cada píxel (i, j):
        blur[i, j] = suma_ponderada(
            vecindad de tamaño 5×5 centrada en (i,j),
            pesos gaussianos 5×5
        )

    RETORNAR blur   # misma forma que gris: (H, W)
```

### Efecto visual

```
Antes del blur:  valores irregulares  [120, 123, 118, 245, 119, 122, ...]
Después del blur: valores suavizados  [121, 121, 121, 165, 122, 122, ...]
                                              ↑
                               el pico de 245 se "diluye" con sus vecinos
```

---

## 7. Paso 4 — Umbralización de Otsu

### Por qué

Después del blur, la imagen sigue siendo en escala de grises (valores 0–255 continuos). Se necesita una imagen **binaria**: cada píxel es exactamente 0 (fondo) o 255 (dígito). Esto hace que la detección de contornos sea robusta e independiente de la iluminación.

El **método de Otsu** determina automáticamente el umbral óptimo analizando el histograma de la imagen, sin que el programador tenga que hardcodear un valor fijo.

### Matemática del método de Otsu

Otsu busca el umbral `T` que **minimiza la varianza intra-clase** (o equivalentemente, maximiza la varianza inter-clase) entre los píxeles del fondo y los del dígito:

```
Para cada umbral T posible (0..255):
  clase_0 = píxeles con valor < T   (fondo)
  clase_1 = píxeles con valor ≥ T   (dígito)

  ω_0 = proporción de píxeles en clase_0
  ω_1 = proporción de píxeles en clase_1
  μ_0 = media de clase_0
  μ_1 = media de clase_1

  varianza_inter(T) = ω_0 × ω_1 × (μ_0 - μ_1)²

T_óptimo = argmax(varianza_inter(T))
```

OpenCV calcula este valor en O(256) iteraciones sobre el histograma.

### Código y pseudocódigo

```python
_, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

```
FUNCIÓN umbralizar_otsu(blur):

    histograma = contar_frecuencias(blur, bins=256)
    # histograma[v] = número de píxeles con valor v

    T_optimo = calcular_otsu(histograma)
    # T_optimo maximiza varianza_inter(T)

    PARA cada píxel (i, j):
        SI blur[i, j] >= T_optimo:
            binaria[i, j] = 0     # BINARY_INV: fondo blanco → negro
        SINO:
            binaria[i, j] = 255   # dígito oscuro → blanco

    RETORNAR binaria   # forma: (H, W), valores: 0 o 255
```

### Por qué `THRESH_BINARY_INV`

MNIST tiene **dígito blanco sobre fondo negro**. Pero en la cámara el dígito escrito en papel es **oscuro sobre fondo claro**. `BINARY_INV` invierte la relación: lo oscuro se vuelve blanco (valor 255) y lo claro se vuelve negro (valor 0), igualando la convención de MNIST.

```
Imagen real:    fondo claro (200)  →  dígito oscuro (30)
Después Otsu:   fondo = 0          ←  dígito = 255
Igual que MNIST: fondo negro,          dígito blanco
```

---

## 8. Paso 5 — Detección de Contornos y Bounding Box

### Qué hace

Sobre la imagen binaria, OpenCV traza los **contornos** (bordes cerrados de regiones blancas). Se selecciona el contorno de mayor área —que corresponde al dígito— y se extrae su rectángulo envolvente.

### Código y pseudocódigo

```python
contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contornos:
    return gris     # fallback: si no detecta nada, retorna la imagen gris completa

contorno_mayor = max(contornos, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(contorno_mayor)

margen = int(max(w, h) * 0.15)
x1 = max(0, x - margen)
y1 = max(0, y - margen)
x2 = min(gris.shape[1], x + w + margen)
y2 = min(gris.shape[0], y + h + margen)

recorte = binaria[y1:y2, x1:x2]
```

```
FUNCIÓN detectar_region(binaria):

    contornos = encontrar_contornos_externos(binaria)
    # traza los bordes de todas las regiones blancas conectadas

    SI contornos está vacío:
        RETORNAR imagen gris completa   # sin dígito detectado

    contorno_mayor = contorno con mayor área encerrada
    # el dígito es la región blanca más grande

    (x, y, w, h) = rectángulo_envolvente(contorno_mayor)
    # x,y: esquina superior izquierda
    # w,h: ancho y alto del rectángulo

    margen = 15% del lado mayor entre w y h
    # contexto extra para no cortar el trazo del dígito

    x1 = max(0,             x - margen)       # límite izquierdo
    y1 = max(0,             y - margen)       # límite superior
    x2 = min(ancho_imagen,  x + w + margen)  # límite derecho
    y2 = min(alto_imagen,   y + h + margen)  # límite inferior

    RETORNAR binaria[y1:y2, x1:x2]
    # recorte rectangular de la imagen binaria alrededor del dígito
```

### Parámetros de `findContours`

| Parámetro | Valor usado | Significado |
|---|---|---|
| `mode` | `RETR_EXTERNAL` | Solo contornos externos (ignora huecos internos como el del "8") |
| `method` | `CHAIN_APPROX_SIMPLE` | Comprime segmentos rectos, almacena solo los extremos |

### Por qué el margen del 15%

Sin margen, el bounding box ajustado exactamente al trazo puede cortar los extremos del dígito (la punta superior del "1", los extremos del "7"). Un 15% del lado mayor añade contexto proporcional sin importar el tamaño del dígito capturado.

---

## 9. Paso 6 — Cuadrado y Centrado del Dígito

### El problema

El recorte del bounding box generalmente **no es cuadrado**: un "1" produce un rectángulo muy alto y estrecho, un "-" produce uno muy ancho y bajo. Si se redimensiona directamente a 28×28, el dígito se **deforma**.

```
Recorte de "1":          Deformado si resize directo:
  ┌──┐                        ┌──────────┐
  │  │  (20px × 80px)  →      │ 1 estirado│  (28px × 28px)
  │ 1│                        │horizontaml│
  │  │                        └──────────┘
  └──┘
```

### La solución: lienzo cuadrado con padding

```python
h, w = roi.shape[:2]
lado = max(h, w)
cuadrado = np.zeros((lado, lado), dtype=roi.dtype)   # lienzo negro cuadrado
y_off = (lado - h) // 2
x_off = (lado - w) // 2
cuadrado[y_off:y_off + h, x_off:x_off + w] = roi    # pegar el dígito centrado
```

```
FUNCIÓN hacer_cuadrado(roi):

    h, w = dimensiones del roi
    lado = max(h, w)

    cuadrado = matriz_de_ceros(lado × lado)   # fondo negro

    desplaz_y = (lado - h) // 2   # margen superior para centrar verticalmente
    desplaz_x = (lado - w) // 2   # margen izquierdo para centrar horizontalmente

    cuadrado[desplaz_y : desplaz_y+h,
             desplaz_x : desplaz_x+w] = roi   # pegar el ROI centrado

    RETORNAR cuadrado   # forma: (lado, lado), dígito centrado
```

### Pseudocódigo visual

```
ROI del "1" (20×80):          Cuadrado resultante (80×80):
                                ┌────────────────────┐
                                │                    │
  ┌──┐                          │     ┌──┐           │
  │  │         →                │     │  │           │
  │ 1│                          │     │ 1│           │
  │  │                          │     │  │           │
  └──┘                          │     └──┘           │
                                │                    │
                                └────────────────────┘
     20×80                              80×80
                           ↑ el "1" queda centrado y sin deformarse
```

---

## 10. Paso 7 — Redimensionar a 28×28

### Código y pseudocódigo

```python
thumbnail = cv2.resize(cuadrado, (28, 28), interpolation=cv2.INTER_AREA)
```

```
FUNCIÓN redimensionar_28x28(cuadrado):

    thumbnail = interpolación_area(cuadrado, destino=28×28)
    # INTER_AREA: promedia los píxeles originales que "caen" en cada píxel destino
    # Recomendado para reducción de tamaño (anti-aliasing natural)

    RETORNAR thumbnail   # forma: (28, 28), dtype: uint8, valores: 0–255
```

### Por qué `INTER_AREA` y no otras interpolaciones

| Interpolación | Método | Cuándo usar |
|---|---|---|
| `INTER_NEAREST` | Píxel más cercano | Imágenes pequeñas → grandes (más rápido, pero pixelado) |
| `INTER_LINEAR` | Bilineal | Uso general, buena calidad |
| `INTER_AREA` | Promedio de área | **Reducción de tamaño** — evita el efecto moiré y suaviza de forma natural |
| `INTER_CUBIC` | Bicúbica | Ampliación de alta calidad (más lento) |

Para pasar de (ej.) 200×200 a 28×28, `INTER_AREA` es el más apropiado porque cada píxel de destino recibe el promedio de los múltiples píxeles de origen que le corresponden, produciendo el resultado más fiel.

### Qué produce: el thumbnail

El thumbnail de 28×28 se retorna junto al vector para mostrarlo en la interfaz web, permitiendo al usuario ver exactamente qué imagen procesó la red.

```
Antes:             Después:
┌────────────┐     ┌───┐
│            │     │   │
│    ╔═╗     │  →  │ 5 │  28×28 píxeles
│    ╚═╝     │     │   │
│            │     └───┘
└────────────┘
  varios cientos     exactamente como MNIST
  de píxeles
```

---

## 11. Paso 8 — Aplanar y Normalizar → Vector (784,)

### Aplanado

```python
vector = thumbnail.flatten().astype(np.float64) / 255.0
```

```
FUNCIÓN aplanar_y_normalizar(thumbnail):

    # Aplanar: matriz (28, 28) → vector (784,)
    # Lee los píxeles fila por fila, de izquierda a derecha, de arriba a abajo
    vector_uint8 = thumbnail.flatten()
    # vector_uint8[0]   = thumbnail[0, 0]   ← píxel fila 0, columna 0
    # vector_uint8[1]   = thumbnail[0, 1]   ← píxel fila 0, columna 1
    # ...
    # vector_uint8[27]  = thumbnail[0, 27]  ← último píxel fila 0
    # vector_uint8[28]  = thumbnail[1, 0]   ← primer píxel fila 1
    # ...
    # vector_uint8[783] = thumbnail[27, 27] ← último píxel

    # Convertir a float64 y normalizar a [0.0, 1.0]
    vector_float = vector_uint8.astype(float64) / 255.0

    RETORNAR vector_float   # forma: (784,), dtype: float64, valores: 0.0–1.0
```

### Por qué dividir entre 255

Los píxeles son enteros en el rango `[0, 255]`. La red fue entrenada con MNIST normalizado a `[0.0, 1.0]`. Si se pasan valores sin normalizar:

```
Entrada sin normalizar:   X = [0, 128, 255, 200, ...]
Z1 = W1 @ X + b1  →  valores enormes → Softmax saturado → gradientes nulos
```

Al normalizar, los productos punto producen valores en un rango manejable donde ReLU y Softmax funcionan correctamente.

### Verificación del resultado

```
thumbnail.shape   = (28, 28)     ✓
vector.shape      = (784,)       ✓  = 28 × 28
vector.dtype      = float64      ✓
vector.min()      ≥ 0.0          ✓
vector.max()      ≤ 1.0          ✓
```

---

## 12. Integración: Función `preprocess_image`

Esta función es el **punto de entrada único** del módulo. Orquesta todos los pasos anteriores.

```python
def preprocess_image(base64_string: str) -> tuple:
    imagen_bgr = decode_frame(base64_string)        # Paso 1
    roi        = detect_digit_region(imagen_bgr)    # Pasos 2–5

    # Paso 6: cuadrado y centrado
    h, w  = roi.shape[:2]
    lado  = max(h, w)
    cuadrado = np.zeros((lado, lado), dtype=roi.dtype)
    y_off = (lado - h) // 2
    x_off = (lado - w) // 2
    cuadrado[y_off:y_off + h, x_off:x_off + w] = roi

    thumbnail = cv2.resize(cuadrado, (28, 28), interpolation=cv2.INTER_AREA)  # Paso 7
    vector    = thumbnail.flatten().astype(np.float64) / 255.0                # Paso 8

    return vector, thumbnail
```

### Pseudocódigo completo integrado

```
FUNCIÓN preprocess_image(base64_string):

    # ── Paso 1: Decodificar frame ──────────────────────────────
    imagen_bgr = decode_frame(base64_string)
    # Entrada: "data:image/jpeg;base64,..."
    # Salida:  (H, W, 3)  dtype=uint8

    # ── Pasos 2–5: Detectar dígito ────────────────────────────
    roi = detect_digit_region(imagen_bgr)
    # Internamente:
    #   gris    = cvtColor(BGR → GRAY)         (H, W)
    #   blur    = GaussianBlur(gris, 5×5)      (H, W)
    #   binaria = threshold_otsu(blur, INV)    (H, W)  valores: 0 o 255
    #   contornos = findContours(binaria)
    #   recorte = binaria[y1:y2, x1:x2]       (h_roi, w_roi)
    # Salida: región binaria alrededor del dígito

    # ── Paso 6: Hacer cuadrado ────────────────────────────────
    h, w = roi.shape
    lado = max(h, w)
    cuadrado = ceros(lado × lado)             # lienzo negro
    cuadrado[centrado_v : centrado_v+h,
             centrado_h : centrado_h+w] = roi  # pegar centrado
    # Salida: (lado, lado)  dígito centrado sin deformación

    # ── Paso 7: Redimensionar ─────────────────────────────────
    thumbnail = resize(cuadrado, 28×28, INTER_AREA)
    # Salida: (28, 28)  dtype=uint8  valores: 0–255

    # ── Paso 8: Aplanar y normalizar ──────────────────────────
    vector = thumbnail.flatten().astype(float64) / 255.0
    # Salida: (784,)  dtype=float64  valores: 0.0–1.0

    RETORNAR vector, thumbnail
    # vector    → entra directo a MLP.forward(X)
    # thumbnail → se muestra en la UI como imagen de 28×28
```

---

## 13. Flujo de Datos con Formas en Cada Etapa

```
Etapa                     Forma              Tipo      Rango de valores
──────────────────────────────────────────────────────────────────────
Frame de la cámara        (H, W, 3)          uint8     0–255  (RGB/BGR)
  ej. 720×1280×3

Después de decode_frame   (720, 1280, 3)     uint8     0–255  (BGR)

Después de cvtColor       (720, 1280)        uint8     0–255  (grises)

Después de GaussianBlur   (720, 1280)        uint8     0–255  (suavizado)

Después de threshold Otsu (720, 1280)        uint8     0 o 255 (binaria)

Después de recorte ROI    (h_roi, w_roi)     uint8     0 o 255
  ej. (180, 90)

Después de hacer cuadrado (180, 180)         uint8     0 o 255

Después de resize 28×28   (28, 28)           uint8     0–255

Después de flatten        (784,)             uint8     0–255

Después de / 255.0        (784,)             float64   0.0–1.0  ✓ listo
──────────────────────────────────────────────────────────────────────
```

---

## 14. Consideraciones para Coincidir con MNIST

MNIST tiene características específicas que hay que replicar para que la red generalice correctamente:

| Característica MNIST | Cómo la replicamos |
|---|---|
| Escala de grises | `cvtColor(BGR→GRAY)` — paso 2 |
| Fondo negro (valor 0) | `THRESH_BINARY_INV` — paso 4 invierte el fondo claro |
| Dígito blanco (valor 255) | `THRESH_BINARY_INV` — el trazo oscuro se vuelve 255 |
| 28×28 píxeles exactos | `cv2.resize(28, 28)` — paso 7 |
| Dígito centrado y con margen | Cuadrado + padding 15% — pasos 5–6 |
| Normalizado [0.0, 1.0] | `/ 255.0` — paso 8 |

**Punto crítico:** Si el dígito se captura en condiciones de iluminación muy diferentes, la umbralización de Otsu sigue funcionando porque es adaptativa al contenido de cada imagen específica.

---

## 15. Estructura de Funciones — API Pública

```python
# image_processing/__init__.py
from .capture      import decode_frame
from .roi_detector import detect_digit_region
from .preprocessor import preprocess_image


# capture.py
def decode_frame(base64_string: str) -> np.ndarray:
    """
    Convierte un frame base64 del navegador en imagen BGR de OpenCV.
    Entrada:  str  — "data:image/jpeg;base64,..."  o solo el payload
    Salida:   np.ndarray  (H, W, 3)  dtype=uint8
    """


# roi_detector.py
def detect_digit_region(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Detecta y recorta la región del dígito en la imagen.
    Pipeline: grises → blur → Otsu → contornos → bounding box con margen.
    Entrada:  np.ndarray  (H, W, 3)  BGR
    Salida:   np.ndarray  (h, w)     binaria (0 o 255)
              Si no detecta dígito, retorna imagen gris completa como fallback.
    """


# preprocessor.py
def preprocess_image(base64_string: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completo: base64 → (vector 784, thumbnail 28×28).
    Orquesta decode_frame + detect_digit_region + cuadrado + resize + normalizar.

    Entrada:  str           — frame base64 de la cámara
    Salida:   tuple con:
                vector:    np.ndarray (784,)    float64  [0.0, 1.0]  → MLP.forward()
                thumbnail: np.ndarray (28, 28)  uint8    [0, 255]    → mostrar en UI
    """
```

---

## Resumen Visual de los 8 Pasos

```
[1] base64 del navegador
     │
     ▼  decode_frame()
[2] imagen BGR (H×W×3)
     │
     ▼  cvtColor(BGR→GRAY)
[3] escala de grises (H×W)
     │
     ▼  GaussianBlur(5×5)
[4] imagen suavizada (H×W)
     │
     ▼  threshold Otsu + BINARY_INV
[5] imagen binaria 0/255 (H×W)
     │
     ▼  findContours → boundingRect → margen 15%
[6] recorte ROI del dígito (h×w variable)
     │
     ▼  padding → cuadrado centrado
[7] cuadrado (lado×lado)
     │
     ▼  resize INTER_AREA
[8] thumbnail 28×28 (uint8)
     │
     ▼  flatten + /255.0
 vector (784,) float64 [0.0–1.0]
     │
     ▼
 MLP.forward(X)  →  predicción del dígito
```
