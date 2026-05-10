# Plan Técnico Estructurado — Proyecto 2: ANN desde Cero
**Curso:** Inteligencia Artificial 1 | **Docente:** Ing. Daniel González  
**Universidad:** USAC — CUNOC | **Ingeniería en Ciencias y Sistemas**

---

## 1. Análisis del Problema

### Objetivo General

Construir una Red Neuronal Artificial (MLP) **completamente desde cero** —sin librerías de Machine Learning— capaz de reconocer dígitos manuscritos (0–9) capturados por cámara web, y exponerla dentro de una aplicación web educativa interactiva desarrollada en Python que visualice en tiempo real los cálculos matemáticos del proceso de aprendizaje.

### Objetivos Específicos

1. Implementar matemáticamente Forward Propagation con funciones de activación ReLU/Sigmoide y Softmax.
2. Implementar Backpropagation con Gradiente Descendente usando solo NumPy.
3. Documentar y verificar los cálculos manuales de una iteración completa con precisión ≥ 4 decimales.
4. Capturar imagen via cámara web y transformarla al vector de entrada (784 valores normalizados).
5. Desarrollar interfaz web con visualización gráfica de la topología de la red, panel de estadísticas y modos de ejecución.
6. Generar reportes automáticos: bitácora de pesos (iteraciones 1, 50, 100) y gráfica de la función de costo.

### Alcance del Sistema

| Componente | Incluido | Excluido |
|---|---|---|
| Red neuronal MLP 784→64→10 | Sí | CNNs, RNNs, redes más profundas |
| Forward + Backprop desde cero | Sí | Autograd, librerías ML |
| Captura de cámara web | Sí | Carga de imágenes por archivo |
| Modo rápido + modo educativo | Sí | Entrenamiento distribuido |
| Visualización de pesos/nodos | Sí | Visualización 3D |
| Exportar gráfica de pérdida | Sí | Guardado de modelo a archivo |
| Dataset de entrenamiento | MNIST (descarga directa) | Datasets propios o alternativos |

### Restricciones Técnicas

- **Lenguaje único:** Python (backend + frontend vía framework web).
- **Librerías prohibidas:** TensorFlow, PyTorch, Keras, Scikit-Learn y cualquier librería de alto nivel de ML.
- **Librerías permitidas:** `numpy`, `math`, `os`, `json`, framework web Python (Flask o FastAPI), librería de captura de cámara (OpenCV o js nativo del navegador).
- **Topología fija:** exactamente 784 → 64 → 10 nodos.
- **Función de salida:** obligatoriamente Softmax.
- **Precisión del debug:** los resultados del código deben coincidir con el cálculo manual a ≥ 4 decimales.

### Entradas y Salidas del Sistema

```
ENTRADA:
  - Imagen de cámara web (RGB, resolución variable)
  - Dataset MNIST para entrenamiento (60,000 imágenes 28x28)
  - Comandos del usuario (modo, velocidad, capturar)

SALIDAS:
  - Predicción del dígito (0–9) con probabilidades Softmax
  - Visualización gráfica de la red (nodos + pesos coloreados)
  - Dashboard: epoch, loss, accuracy en tiempo real
  - Gráfica de curva de pérdida (exportable)
  - Bitácora de matrices de pesos (iterations 1, 50, 100)
  - Modo debug: impresión de cálculos de una iteración
```

---

## 2. División del Sistema en Módulos Principales

---

### 🧠 Fase 1 — Red Neuronal desde Cero

**Problema que resuelve:** Clasificar un vector de 784 valores (píxeles) en una de 10 categorías (dígitos 0–9) aprendiendo a partir de ejemplos sin usar librerías de ML.

**Responsabilidades:**
- Inicializar pesos y sesgos aleatoriamente.
- Ejecutar Forward Propagation capa a capa.
- Calcular la pérdida con Cross-Entropy.
- Ejecutar Backpropagation calculando gradientes analíticamente.
- Actualizar pesos con Gradiente Descendente.
- Proveer modo debug con impresión de cálculos intermedios.
- Generar la bitácora de matrices en iteraciones 1, 50 y 100.

**Entradas:**
- Vector `X` de forma `(784,)` con valores normalizados `[0, 1]`.
- Etiqueta `y` como entero `0–9`.
- Hiperparámetros: `learning_rate`, `epochs`, `batch_size`.

**Salidas:**
- Vector de probabilidades `(10,)` — resultado Softmax.
- Gradientes `dW1`, `db1`, `dW2`, `db2`.
- Historial de pérdida y accuracy por epoch.
- Impresión debug de una iteración.

**Dependencias:** Recibe datos procesados de la Fase 2. Expone API interna para la Fase 3.

---

### 🖼️ Fase 2 — Procesamiento de Imagen (Cámara → 28×28 → Vector)

**Problema que resuelve:** Transformar la imagen cruda de la cámara web (RGB, alta resolución) en el vector de 784 valores flotantes que acepta la red neuronal.

**Responsabilidades:**
- Acceder al stream de la cámara web del dispositivo.
- Capturar un frame bajo demanda del usuario.
- Convertir imagen a escala de grises.
- Detectar y recortar la región de interés (dígito escrito).
- Redimensionar a exactamente 28×28 píxeles.
- Aplanar la matriz `(28,28)` → vector `(784,)`.
- Normalizar píxeles al rango `[0.0, 1.0]`.

**Entradas:**
- Frame de cámara web (JPEG/RGB desde el navegador o OpenCV).
- Señal de captura del usuario.

**Salidas:**
- Vector NumPy `(784,)` normalizado listo para inferencia.
- Thumbnail 28×28 para mostrar en la UI.

**Dependencias:** Alimenta directamente la Fase 1 para predicción. La UI de la Fase 3 controla cuándo capturar.

---

### 🌐 Fase 3 — Aplicación Web Interactiva

**Problema que resuelve:** Proveer al usuario una interfaz visual que controle el entrenamiento, visualice la red neuronal en tiempo real, capture dígitos y muestre la predicción con trazabilidad educativa.

**Responsabilidades:**
- Servir la aplicación web desde Python.
- Exponer endpoints REST/WebSocket para comunicar frontend ↔ backend.
- Renderizar la topología de la red con nodos y conexiones coloreadas.
- Implementar modo rápido (entrenamiento continuo) y modo educativo (paso a paso).
- Mostrar dashboard: epoch, loss, accuracy.
- Graficar la curva de pérdida en tiempo real.
- Gestionar la captura de cámara e invocar el pipeline de procesamiento.
- Mostrar el resultado de predicción con barras de probabilidad Softmax.

**Entradas:**
- Eventos del usuario (botones, sliders, captura).
- Datos del motor neuronal (pesos, gradientes, métricas).

**Salidas:**
- HTML/CSS/JS renderizado en el navegador.
- Respuestas JSON con estado de la red.
- Gráfica de pérdida exportable (PNG o SVG).

**Dependencias:** Orquesta Fase 1 y Fase 2. Depende de que el motor neuronal esté instanciado.

---

## 3. Extracción de Requerimientos

### Requerimientos Funcionales

| ID | Descripción |
|---|---|
| RF-01 | La red debe tener exactamente 784 nodos de entrada, 64 ocultos y 10 de salida. |
| RF-02 | Forward Propagation debe implementarse sin librerías ML, usando solo NumPy. |
| RF-03 | Backpropagation debe calcular gradientes analíticos usando regla de la cadena. |
| RF-04 | El sistema debe entrenar con el dataset MNIST. |
| RF-05 | La cámara web debe activarse desde el navegador y capturar frames. |
| RF-06 | La imagen capturada debe pasar por: grises → 28×28 → vector 784 → normalización. |
| RF-07 | El modo debug debe imprimir exactamente el mismo resultado que el cálculo manual (≥4 decimales). |
| RF-08 | El modo rápido entrena a máxima velocidad sin pausas. |
| RF-09 | El modo educativo muestra cada paso de Forward y Backward con botón "siguiente". |
| RF-10 | El dashboard debe mostrar epoch actual, loss y accuracy en vivo. |
| RF-11 | La visualización de la red debe colorear/engrosar conexiones según valor del peso. |
| RF-12 | La bitácora debe guardar secciones de la matriz W1 en iteraciones 1, 50 y 100. |
| RF-13 | La gráfica de pérdida debe actualizarse en tiempo real y ser exportable. |
| RF-14 | La predicción debe mostrar las 10 probabilidades Softmax como barras. |

### Requerimientos No Funcionales

| ID | Descripción |
|---|---|
| RNF-01 | El backend debe estar completamente en Python. |
| RNF-02 | El tiempo de respuesta de una predicción (inferencia) no debe superar 500ms. |
| RNF-03 | El código debe ser modular y separar motor neuronal, procesamiento e interfaz. |
| RNF-04 | Los pesos iniciales deben ser reproducibles con una semilla fija (`np.random.seed`). |
| RNF-05 | El sistema debe funcionar localmente sin conexión a internet (salvo descarga inicial de MNIST). |
| RNF-06 | La bitácora de matrices debe generarse automáticamente sin intervención manual. |
| RNF-07 | El código debe incluir comentarios que expliquen cada operación matemática. |

### Restricciones Obligatorias del Proyecto

- **PROHIBIDO:** TensorFlow, PyTorch, Keras, Scikit-Learn o cualquier librería que abstraiga la red neuronal.
- **OBLIGATORIO:** El cálculo manual debe coincidir con la primera iteración del programa a ≥ 4 decimales.
- **OBLIGATORIO:** Generar bitácora de matrices en iteraciones 1, 50 y 100.
- **OBLIGATORIO:** El lenguaje es Python.
- **OBLIGATORIO:** Entregar: código fuente, cálculo manual, bitácora, manual técnico, manual de usuario.

---

## 4. Diseño Técnico por Módulo

---

### 🧠 Módulo 1 — Motor de Red Neuronal

#### Componentes Internos

```
neural_network/
├── network.py         # Clase MLP: inicialización, forward, backward, train
├── activations.py     # relu(), relu_derivative(), sigmoid(), sigmoid_derivative(), softmax()
├── loss.py            # cross_entropy_loss(), accuracy()
├── initializer.py     # he_init(), xavier_init() con semilla fija
├── logger.py          # Bitácora de matrices en iter 1, 50, 100
└── debug.py           # Modo debug: impresión detallada de una iteración
```

#### Flujo de Datos — Forward Propagation

```
X (784,)
  │
  ▼  Z1 = W1 @ X + b1         [W1: (64,784), b1: (64,)]  → Z1: (64,)
  ▼  A1 = ReLU(Z1)            → A1: (64,)
  ▼  Z2 = W2 @ A1 + b2        [W2: (10,64), b2: (10,)]   → Z2: (10,)
  ▼  A2 = Softmax(Z2)         → A2: (10,)  [suma = 1.0]
  ▼
  Loss = -sum(y_onehot * log(A2))   [Cross-Entropy]
```

#### Flujo de Datos — Backpropagation

```
dL/dZ2 = A2 - y_onehot                          (10,)
dL/dW2 = dL/dZ2 ⊗ A1ᵀ                          (10,64)
dL/db2 = dL/dZ2                                  (10,)

dL/dA1 = W2ᵀ @ dL/dZ2                           (64,)
dL/dZ1 = dL/dA1 * ReLU'(Z1)                     (64,)
dL/dW1 = dL/dZ1 ⊗ Xᵀ                            (64,784)
dL/db1 = dL/dZ1                                  (64,)

Actualización:
  W1 -= lr * dL/dW1
  b1 -= lr * dL/db1
  W2 -= lr * dL/dW2
  b2 -= lr * dL/db2
```

#### Fórmulas Matemáticas Clave

**ReLU:**
```
ReLU(z) = max(0, z)
ReLU'(z) = 1 si z > 0, 0 si z ≤ 0
```

**Softmax:**
```
softmax(z_i) = exp(z_i) / sum(exp(z_j))  para j = 0..9
```

**Cross-Entropy Loss:**
```
L = -sum(y_k * log(ŷ_k))  para k = 0..9
```

**Gradiente Softmax + Cross-Entropy (combinado):**
```
dL/dZ2 = ŷ - y   (simplificación algebraica del gradiente combinado)
```

#### Tecnologías

- Python 3.10+
- NumPy (operaciones matriciales)
- `math` (funciones escalares si aplica)

---

### 🖼️ Módulo 2 — Pipeline de Procesamiento de Imagen

#### Componentes Internos

```
image_processing/
├── capture.py         # Interfaz con cámara: captura frame desde stream
├── preprocessor.py    # Conversión grises, resize 28x28, flatten, normalize
├── roi_detector.py    # Detección de región del dígito (bounding box)
└── visualizer.py      # Retorna thumbnail 28x28 para mostrar en UI
```

#### Flujo de Datos

```
[Cámara Web] ──JPEG──▶ capture.py
                              │
                              ▼
                    roi_detector.py
                    - Convertir a grises (OpenCV o PIL)
                    - Umbralización (threshold) para detectar el dígito
                    - Encontrar contornos y bounding box
                              │
                              ▼
                    preprocessor.py
                    - Recortar región del dígito
                    - Redimensionar a 28×28 (interpolación bilineal)
                    - Aplanar: (28,28) → (784,)
                    - Normalizar: pixel / 255.0 → [0.0, 1.0]
                              │
                              ▼
                    Vector X (784,)  ──▶  Motor Neuronal
```

#### Consideraciones de Preprocesamiento

- El fondo debe ser blanco y el dígito oscuro (como MNIST). Si la cámara produce lo contrario, invertir: `1.0 - pixel`.
- Aplicar un pequeño blur gaussiano para reducir ruido antes de umbralizar.
- Centrar el dígito dentro del recuadro 28×28 (igual que MNIST).

#### Tecnologías

- `OpenCV` (`cv2`) para captura y procesamiento de imagen.
- `Pillow` (PIL) como alternativa para resize.
- La captura del stream de cámara se puede manejar desde el frontend (JS `getUserMedia`) y enviar el frame al backend como base64 o multipart.

---

### 🌐 Módulo 3 — Aplicación Web Interactiva

#### Componentes Internos

```
web_app/
├── app.py                    # Servidor Flask/FastAPI, rutas y WebSocket
├── routes/
│   ├── training.py           # Endpoints: start, stop, step, status
│   ├── prediction.py         # Endpoint: capturar frame y predecir
│   └── reports.py            # Endpoint: exportar gráfica de pérdida
├── templates/
│   └── index.html            # Página principal (Jinja2)
├── static/
│   ├── css/style.css         # Estilos
│   ├── js/
│   │   ├── network_viz.js    # Visualización de la red (Canvas o SVG)
│   │   ├── camera.js         # Acceso a cámara via getUserMedia
│   │   ├── charts.js         # Gráfica de pérdida en tiempo real
│   │   └── dashboard.js      # Panel epoch/loss/accuracy
│   └── img/
└── sockets/
    └── events.py             # Eventos WebSocket para streaming de datos
```

#### Flujo de Datos de la UI

```
Usuario ──clic "Entrenar"──▶ Frontend ──POST /train/start──▶ Backend
                                                                  │
                                              Motor neuronal itera│
                                                                  ▼
                                              WebSocket emite cada N iteraciones:
                                              { epoch, loss, accuracy, weights_sample }
                                                                  │
Frontend recibe ──────────────────────────────────────────────────▶
  - Actualiza dashboard (epoch, loss, accuracy)
  - Actualiza gráfica de pérdida (Chart.js o D3.js)
  - Actualiza visualización de la red (grosor/color de conexiones)
```

```
Usuario ──clic "Capturar"──▶ camera.js captura frame
                                  │
                        Envía frame (base64) ──POST /predict──▶ Backend
                                                                     │
                                                          Fase 2 procesa│
                                                          Fase 1 infiere │
                                                                     ▼
                                              { digit: 7, probabilities: [0.01,...,0.95,...] }
                                                                     │
Frontend muestra ─────────────────────────────────────────────────────▶
  - Dígito predicho resaltado
  - Barras de probabilidad para cada clase 0–9
  - Thumbnail del dígito procesado 28×28
```

#### Visualización de la Red (network_viz.js)

- Representar las capas como columnas de nodos circulares.
- Capa de entrada: mostrar solo una muestra (ej. 28 nodos representativos de los 784).
- Capa oculta: 64 nodos.
- Capa de salida: 10 nodos (etiquetados 0–9).
- Conexiones: líneas cuyo **grosor** y **color** (rojo = negativo, azul = positivo, intensidad = magnitud) representan el valor del peso.
- En modo educativo: animar el flujo de activación nodo a nodo con delay configurable.

#### Tecnologías Sugeridas

| Componente | Tecnología |
|---|---|
| Servidor web | Flask (simple) o FastAPI (async) |
| Comunicación tiempo real | Flask-SocketIO o WebSockets nativos |
| Visualización red | Canvas HTML5 (JavaScript puro) |
| Gráfica de pérdida | Chart.js (CDN, sin instalación) |
| Captura de cámara | `navigator.mediaDevices.getUserMedia` (JS nativo) |
| Procesamiento imagen | OpenCV (`cv2`) en backend |
| Templates | Jinja2 (incluido en Flask) |

---

## 5. Planificación por Etapas (Roadmap)

### Etapa 0 — Preparación y Cálculo Manual (Días 1–2)

**Objetivo:** Comprender la matemática antes de programar. Generar el documento de cálculo manual.

**Tareas técnicas:**
1. Definir semilla aleatoria fija (`seed = 42`).
2. Generar pesos iniciales W1 `(64,784)` y W2 `(10,64)` con distribución normal pequeña.
3. Tomar **una sola imagen** de MNIST (ej. el dígito "5" del índice 0).
4. Calcular a mano (o en Excel):
   - Producto punto `Z1 = W1 @ X + b1` para los primeros 3 nodos ocultos.
   - `A1 = ReLU(Z1)` para esos 3 nodos.
   - `Z2 = W2 @ A1 + b2` para los 10 nodos de salida.
   - `A2 = Softmax(Z2)`.
   - `Loss = CrossEntropy(A2, y)`.
   - Gradiente `dZ2 = A2 - y_onehot`.
   - `dW2 = dZ2 @ A1ᵀ` para al menos 2 filas.
   - Un paso de actualización de peso: `W2_new = W2 - lr * dW2`.
5. Documentar todo en PDF/Excel con fórmulas visibles.

**Entregables:** Documento de cálculo manual (PDF o Excel) con ≥4 decimales.

**Dependencias:** Ninguna.

---

### Etapa 1 — Motor Neuronal Core (Días 3–5)

**Objetivo:** Implementar el motor neuronal completo con Forward, Backward, debug y bitácora.

**Tareas técnicas:**
1. Crear `initializer.py`: función `initialize_weights(seed=42)` que retorna `W1, b1, W2, b2`.
2. Crear `activations.py`:
   - `relu(Z)` y `relu_derivative(Z)` vectorizados con NumPy.
   - `softmax(Z)` con estabilidad numérica: `exp(Z - max(Z)) / sum(...)`.
3. Crear `loss.py`:
   - `cross_entropy(A2, y_onehot)` con clip para evitar `log(0)`.
   - `accuracy(A2, y)`.
4. Crear `network.py` — clase `MLP`:
   - `forward(X)` → retorna `Z1, A1, Z2, A2`.
   - `backward(X, y_onehot, Z1, A1, A2)` → retorna `dW1, db1, dW2, db2`.
   - `update(dW1, db1, dW2, db2)` → actualiza pesos.
   - `train(X_train, y_train, epochs, lr)` → itera y registra métricas.
5. Crear `debug.py`: función `debug_single_sample(X, y, network)` que imprime cada variable intermedia con 4 decimales y los compara contra el cálculo manual.
6. Crear `logger.py`: durante `train()`, en iteraciones 1, 50 y 100 guardar en `.log` una sección `W1[:5, :5]`.
7. Ejecutar debug y verificar que coincide con el cálculo manual.

**Entregables:** Módulo `neural_network/` funcional. Modo debug verificado. Bitácora generada.

**Dependencias:** Etapa 0 (semilla y valores manuales para verificación).

---

### Etapa 2 — Carga y Preprocesamiento de MNIST (Días 5–6)

**Objetivo:** Cargar el dataset MNIST y tenerlo listo para entrenar.

**Tareas técnicas:**
1. Crear `data/mnist_loader.py`:
   - Descargar MNIST desde `keras.datasets` (**solo** para la descarga, no para el modelo) o directamente desde el repositorio de Yann LeCun usando `urllib`.
   - Alternativa sin Keras: usar el paquete `python-mnist` o descargar los archivos `.gz` manualmente.
2. Normalizar imágenes: `X / 255.0`.
3. Aplanar imágenes: `(N, 28, 28)` → `(N, 784)`.
4. Convertir etiquetas a one-hot: `y_onehot` de forma `(N, 10)`.
5. Separar en train (60,000) y test (10,000).
6. Implementar mini-batch shuffle: `np.random.permutation`.

**Entregables:** Módulo `data/mnist_loader.py`. Script de prueba que carga y muestra una imagen.

**Dependencias:** Etapa 1.

---

### Etapa 3 — Entrenamiento Completo y Validación (Días 7–8)

**Objetivo:** Entrenar la red con MNIST y alcanzar accuracy aceptable.

**Tareas técnicas:**
1. Conectar `mnist_loader.py` con `network.py`.
2. Configurar hiperparámetros iniciales: `lr=0.01`, `epochs=20`, `batch_size=32`.
3. Ejecutar entrenamiento completo y monitorear loss/accuracy por epoch.
4. Ajustar `lr` y `epochs` hasta lograr accuracy ≥ 85% en test.
5. Verificar que la bitácora `.log` se generó correctamente en iteraciones 1, 50 y 100.
6. Guardar historial de pérdida como lista para graficar.

**Entregables:** Red entrenada. Historial de métricas. Bitácora de matrices validada.

**Dependencias:** Etapas 1 y 2.

---

### Etapa 4 — Pipeline de Procesamiento de Imagen (Días 8–9)

**Objetivo:** Capturar imagen de cámara y convertirla al vector de entrada de la red.

**Tareas técnicas:**
1. Implementar `capture.py`: recibir imagen base64 desde el frontend y decodificarla con OpenCV.
2. Implementar `roi_detector.py`:
   - Convertir a escala de grises (`cv2.cvtColor`).
   - Aplicar blur gaussiano (`cv2.GaussianBlur`).
   - Umbralizar (`cv2.threshold` con Otsu).
   - Encontrar contornos (`cv2.findContours`).
   - Extraer bounding box del contorno más grande.
3. Implementar `preprocessor.py`:
   - Recortar ROI.
   - Resize a 28×28 (`cv2.resize` con `INTER_AREA`).
   - Invertir si el dígito es claro sobre fondo oscuro.
   - Aplanar y normalizar.
4. Probar con imágenes de prueba estáticas antes de integrar con cámara.

**Entregables:** Módulo `image_processing/` funcional. Script de prueba con imagen de test.

**Dependencias:** Etapa 1 (para inferencia inmediata).

---

### Etapa 5 — Servidor Web y API (Días 10–11)

**Objetivo:** Crear el servidor Flask con todos los endpoints necesarios.

**Tareas técnicas:**
1. Crear `app.py` con Flask + Flask-SocketIO.
2. Instanciar la red neuronal al iniciar el servidor.
3. Implementar endpoints:
   - `POST /train/start` → inicia entrenamiento en hilo separado (`threading.Thread`).
   - `POST /train/stop` → detiene el entrenamiento.
   - `POST /train/step` → ejecuta exactamente un paso (modo educativo).
   - `GET /train/status` → retorna estado actual (epoch, loss, accuracy).
   - `POST /predict` → recibe imagen base64, procesa y retorna predicción.
   - `GET /reports/loss_chart` → retorna datos de la curva de pérdida como JSON.
4. Configurar WebSocket: emitir evento `training_update` cada N iteraciones con `{ epoch, loss, accuracy, weights_sample }`.
5. Configurar CORS si es necesario.

**Entregables:** Servidor Flask funcional. Endpoints probados con `curl` o Postman.

**Dependencias:** Etapas 1, 3 y 4.

---

### Etapa 6 — Frontend: Dashboard y Visualizaciones (Días 12–14)

**Objetivo:** Construir la interfaz web completa con todas las visualizaciones requeridas.

**Tareas técnicas:**
1. Diseñar layout HTML en `index.html`:
   - Sección izquierda: visualización de la red neuronal.
   - Sección central: controles (botones Entrenar/Pausar/Paso/Capturar, slider velocidad).
   - Sección derecha: dashboard (epoch, loss, accuracy) + gráfica de pérdida.
   - Sección inferior: preview de cámara + resultado de predicción.
2. Implementar `network_viz.js` con Canvas API:
   - Dibujar nodos como círculos por capa.
   - Dibujar conexiones como líneas coloreadas (rojo negativo, azul positivo).
   - Actualizar colores/grosores cuando llega `weights_sample` por WebSocket.
3. Implementar `camera.js`:
   - Activar cámara con `navigator.mediaDevices.getUserMedia`.
   - Mostrar stream en `<video>`.
   - Al capturar: extraer frame a `<canvas>`, convertir a base64, enviar a `/predict`.
4. Implementar `charts.js` con Chart.js:
   - Gráfica de línea para curva de pérdida.
   - Actualizar en tiempo real con datos del WebSocket.
   - Botón "Exportar PNG".
5. Implementar `dashboard.js`:
   - Escuchar eventos WebSocket y actualizar contadores epoch, loss, accuracy.
   - Mostrar barras de probabilidad Softmax al recibir predicción.
6. Implementar modo educativo:
   - Al activar "Modo Lento": cada clic en "Siguiente Paso" llama a `POST /train/step`.
   - Animar activación de nodos en `network_viz.js` resaltando el nodo activo.

**Entregables:** Frontend completo y funcional. Todas las visualizaciones operativas.

**Dependencias:** Etapa 5.

---

### Etapa 7 — Integración, Pruebas y Generación de Reportes (Días 15–16)

**Objetivo:** Integrar todos los módulos, probar el flujo completo y generar todos los entregables.

**Tareas técnicas:**
1. Prueba end-to-end: abrir navegador → entrenar → capturar dígito → ver predicción.
2. Verificar que la bitácora `.log` contiene las matrices en iteraciones 1, 50 y 100.
3. Verificar que la gráfica de pérdida se exporta correctamente.
4. Ejecutar modo debug y confirmar que coincide con el cálculo manual.
5. Probar modo educativo paso a paso.
6. Revisar que las conexiones de la red cambian visualmente durante el entrenamiento.
7. Escribir Manual Técnico: documentar las derivadas parciales y el álgebra paso a paso.
8. Escribir Manual de Usuario: instrucciones para iniciar la app y usar la cámara.

**Entregables:** Sistema completo integrado. Manuales. Todos los archivos de reporte.

**Dependencias:** Todas las etapas anteriores.

---

## 6. Plan de Implementación Paso a Paso (Checklist)

### Fase Matemática (Pre-código)

- [ ] Fijar semilla aleatoria: `np.random.seed(42)`
- [ ] Generar y guardar pesos iniciales W1`(64,784)`, b1`(64,)`, W2`(10,64)`, b2`(10,)`
- [ ] Seleccionar imagen X de índice 0 de MNIST y su etiqueta y
- [ ] Calcular a mano Z1 para los primeros 3 nodos ocultos
- [ ] Calcular A1 = ReLU(Z1) para esos 3 nodos
- [ ] Calcular Z2 y A2=Softmax(Z2) completos (10 nodos)
- [ ] Calcular Loss = CrossEntropy(A2, y_onehot)
- [ ] Calcular gradiente dZ2 = A2 - y_onehot
- [ ] Calcular dW2 (primeras 2 filas) y db2
- [ ] Calcular dA1, dZ1, dW1 (primeras 2 filas)
- [ ] Calcular nuevo W2[0,0] después de un update
- [ ] Documentar todo en Excel/PDF con ≥4 decimales

### Motor Neuronal

- [ ] Crear estructura de carpetas del proyecto
- [ ] Implementar `initializer.py` con semilla fija
- [ ] Implementar `relu()`, `relu_derivative()` vectorizados
- [ ] Implementar `softmax()` con estabilidad numérica
- [ ] Implementar `cross_entropy_loss()` con clipping
- [ ] Implementar `accuracy()`
- [ ] Implementar clase `MLP.forward(X)`
- [ ] Implementar clase `MLP.backward(X, y, Z1, A1, A2)`
- [ ] Implementar clase `MLP.update(gradients, lr)`
- [ ] Implementar clase `MLP.train(X, y, epochs, lr)`
- [ ] Implementar `debug.py`: imprimir cada variable intermedia a 4 decimales
- [ ] Verificar que debug output == cálculo manual
- [ ] Implementar `logger.py`: guardar W1[:5,:5] en iteraciones 1, 50, 100

### Dataset

- [ ] Implementar descarga de MNIST (sin Keras si es posible)
- [ ] Normalizar píxeles a [0.0, 1.0]
- [ ] Aplanar imágenes a (N, 784)
- [ ] Convertir etiquetas a one-hot (N, 10)
- [ ] Implementar mini-batch con shuffle
- [ ] Probar carga y visualizar una muestra

### Entrenamiento

- [ ] Conectar dataset con el motor neuronal
- [ ] Ejecutar entrenamiento con lr=0.01, epochs=20
- [ ] Monitorear loss y accuracy por epoch en consola
- [ ] Ajustar hiperparámetros hasta accuracy ≥ 85%
- [ ] Verificar bitácora de matrices generada
- [ ] Guardar historial de pérdida como lista JSON

### Procesamiento de Imagen

- [ ] Implementar recepción de imagen base64 en backend
- [ ] Implementar conversión a escala de grises
- [ ] Implementar blur gaussiano
- [ ] Implementar umbralización Otsu
- [ ] Implementar detección de contornos y bounding box
- [ ] Implementar resize a 28×28
- [ ] Implementar inversión de colores si necesario
- [ ] Implementar aplanado y normalización
- [ ] Probar con imagen estática y verificar vector resultante

### Servidor Web

- [ ] Inicializar proyecto Flask con estructura de carpetas
- [ ] Configurar Flask-SocketIO
- [ ] Implementar `POST /train/start` con hilo separado
- [ ] Implementar `POST /train/stop`
- [ ] Implementar `POST /train/step` (un paso)
- [ ] Implementar `GET /train/status`
- [ ] Implementar `POST /predict`
- [ ] Implementar `GET /reports/loss_chart`
- [ ] Configurar evento WebSocket `training_update`
- [ ] Probar todos los endpoints

### Frontend

- [ ] Crear `index.html` con layout de 4 secciones
- [ ] Implementar visualización de red con Canvas (3 capas, nodos, conexiones)
- [ ] Implementar coloreado de conexiones según valor del peso
- [ ] Implementar acceso a cámara con `getUserMedia`
- [ ] Implementar captura de frame y envío como base64
- [ ] Implementar gráfica de pérdida con Chart.js
- [ ] Implementar actualización en tiempo real vía WebSocket
- [ ] Implementar panel de estadísticas (epoch, loss, accuracy)
- [ ] Implementar barras de probabilidad Softmax
- [ ] Implementar modo educativo con botón "Siguiente Paso"
- [ ] Implementar animación de activación en modo educativo
- [ ] Implementar exportación de gráfica como PNG
- [ ] Ajustar estilos CSS

### Documentación y Entregables

- [ ] Manual Técnico: derivadas parciales, álgebra de forward y backward
- [ ] Manual de Usuario: instrucciones de instalación y uso de cámara
- [ ] Verificar bitácora `.log` con matrices en iter 1, 50 y 100
- [ ] Exportar gráfica de pérdida final
- [ ] Organizar repositorio Git con estructura clara
- [ ] Subir a Classroom/Git

---

## 7. Estructura de Carpetas del Proyecto

```
proyecto2_ann/
├── app.py                          # Punto de entrada del servidor Flask
├── requirements.txt                # numpy, flask, flask-socketio, opencv-python
├── neural_network/
│   ├── __init__.py
│   ├── network.py                  # Clase MLP
│   ├── activations.py              # relu, softmax, sigmoid
│   ├── loss.py                     # cross_entropy, accuracy
│   ├── initializer.py              # Inicialización de pesos con semilla
│   ├── logger.py                   # Bitácora de matrices
│   └── debug.py                    # Modo debug iteración única
├── image_processing/
│   ├── __init__.py
│   ├── capture.py                  # Decodificación de frame base64
│   ├── roi_detector.py             # Detección de región del dígito
│   └── preprocessor.py            # Resize, flatten, normalize
├── data/
│   ├── __init__.py
│   ├── mnist_loader.py             # Descarga y preparación de MNIST
│   └── raw/                        # Archivos .gz de MNIST descargados
├── web_app/
│   ├── routes/
│   │   ├── training.py
│   │   ├── prediction.py
│   │   └── reports.py
│   ├── sockets/
│   │   └── events.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/
│           ├── network_viz.js
│           ├── camera.js
│           ├── charts.js
│           └── dashboard.js
├── reports/
│   ├── weight_matrix.log           # Bitácora generada automáticamente
│   └── loss_chart.png              # Gráfica exportada
├── docs/
│   ├── calculo_manual.pdf          # Documento de cálculo manual
│   ├── manual_tecnico.pdf
│   └── manual_usuario.pdf
└── tests/
    ├── test_network.py             # Prueba forward/backward con valores conocidos
    └── test_preprocessor.py       # Prueba pipeline de imagen
```

---

## 8. Tabla Resumen de Fases y Dependencias

| Etapa | Días | Entregable Principal | Depende de |
|---|---|---|---|
| 0 — Cálculo Manual | 1–2 | PDF/Excel cálculo manual | — |
| 1 — Motor Neuronal | 3–5 | `neural_network/` + debug verificado | Etapa 0 |
| 2 — Dataset MNIST | 5–6 | `data/mnist_loader.py` | Etapa 1 |
| 3 — Entrenamiento | 7–8 | Red entrenada ≥85% accuracy | Etapas 1, 2 |
| 4 — Pipeline Imagen | 8–9 | `image_processing/` | Etapa 1 |
| 5 — Servidor Web | 10–11 | API Flask + WebSocket | Etapas 1, 3, 4 |
| 6 — Frontend | 12–14 | UI completa con visualizaciones | Etapa 5 |
| 7 — Integración | 15–16 | Sistema completo + manuales | Todas |

---

## 9. Notas Críticas para la Calificación

> **PUNTO CRÍTICO:** El cálculo manual debe coincidir exactamente con la primera iteración del programa. Para garantizarlo:
> 1. Usar `np.random.seed(42)` antes de generar cualquier peso.
> 2. El cálculo manual debe usar los **mismos valores iniciales** de pesos que el programa.
> 3. El modo debug debe imprimir `Z1[0]`, `A1[0]`, `Z2`, `A2`, `Loss`, `dZ2`, `dW2[0,0]`, `W2_new[0,0]` con `:.4f`.
> 4. Verificar también `dW1` y la actualización de W1 para al menos 2 pesos.

> **ENTREGA:** El repositorio Git debe contener exactamente los 5 entregables obligatorios más el código fuente organizado. Verificar antes de subir que la bitácora `.log` fue generada por una ejecución real del programa.
