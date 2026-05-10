# Estructura Completa del Proyecto
## ANN desde Cero — Proyecto 2 | Inteligencia Artificial 1 | USAC CUNOC

---

## Árbol de carpetas y archivos

```
proyecto2_ann/
│
├── run.py                          ← Punto de entrada: python run.py
├── app.py                          ← Fábrica Flask + SocketIO + configuración global
├── trainer.py                      ← Bucle de entrenamiento y modo paso a paso
├── requirements.txt                ← Dependencias del proyecto
├── .gitignore                      ← Excluye venv, pycache, MNIST .gz, reportes
│
├── neural_network/                 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│   │                               FASE 1 — Motor neuronal desde cero
│   ├── __init__.py                 Exporta MLP, funciones y clases del módulo
│   ├── initializer.py              Inicialización He (W1) y Xavier (W2), seed=42
│   ├── activations.py              relu, relu_derivative, softmax, sigmoid
│   ├── loss.py                     one_hot, cross_entropy_loss, accuracy
│   ├── network.py                  Clase MLP: forward, backward, update, predict
│   ├── logger.py                   Bitácora automática W1 en iteraciones 1, 50, 100
│   └── debug.py                    Modo debug: impresión con 4 decimales por iteración
│
├── image_processing/               ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│   │                               FASE 2 — Pipeline cámara → vector 784
│   ├── __init__.py                 Exporta decode_frame, detect_digit_region, preprocess_image
│   ├── capture.py                  base64 del navegador → imagen BGR (OpenCV)
│   ├── roi_detector.py             grises → blur → Otsu → contornos → bounding box
│   └── preprocessor.py            cuadrado → 28×28 → flatten → normalizar [0,1]
│
├── web_app/                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│   │                               FASE 3 — Aplicación web interactiva
│   ├── __init__.py
│   │
│   ├── routes/                     Endpoints REST
│   │   ├── __init__.py
│   │   ├── training.py             /train/start  /train/stop  /train/step  /train/status
│   │   ├── prediction.py           /predict  (recibe frame base64 → retorna dígito)
│   │   └── reports.py              /reports/loss_chart  /reports/weights_log
│   │
│   ├── sockets/                    Comunicación en tiempo real
│   │   ├── __init__.py
│   │   └── events.py               Eventos WebSocket: connect, disconnect, training_update
│   │
│   ├── templates/                  HTML (renderizado por Jinja2)
│   │   └── index.html              Página única: visualización + cámara + dashboard
│   │
│   └── static/                     Archivos estáticos servidos por Flask
│       ├── css/
│       │   └── style.css           Tema oscuro completo (layout, paneles, botones)
│       └── js/
│           ├── network_viz.js      Canvas API: nodos y conexiones coloreadas por peso
│           ├── camera.js           getUserMedia: acceso a cámara y captura de frames
│           ├── charts.js           Chart.js: curva de pérdida en tiempo real + exportar PNG
│           └── dashboard.js        WebSocket + orquestación de todos los módulos JS
│
├── data/                           ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│   │                               Dataset MNIST
│   ├── __init__.py
│   ├── mnist_loader.py             Descarga y carga MNIST sin Keras (urllib + struct)
│   └── raw/                        Archivos .gz descargados (ignorados por .gitignore)
│       ├── train-images-idx3-ubyte.gz   60,000 imágenes de entrenamiento
│       ├── train-labels-idx1-ubyte.gz   60,000 etiquetas de entrenamiento
│       ├── t10k-images-idx3-ubyte.gz    10,000 imágenes de prueba
│       └── t10k-labels-idx1-ubyte.gz    10,000 etiquetas de prueba
│
├── reports/                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│   │                               Archivos generados automáticamente en ejecución
│   ├── weight_matrix.log           Bitácora de pesos W1 en iteraciones 1, 50 y 100
│   └── loss_chart.png              Gráfica de pérdida exportada desde la UI
│
├── tests/                          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│   │                               Suite de pruebas automatizadas
│   ├── __init__.py
│   ├── test_network.py             Tests: shapes, activaciones, gradientes, backprop
│   └── test_preprocessor.py        Tests: pipeline de imagen, vector 784, normalización
│
└── docs/                           ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │                               Documentación técnica del proyecto
    ├── ESTRUCTURA_PROYECTO.md      Este archivo
    ├── DOCUMENTACION_FASE1.md      Motor neuronal: matemáticas, código, resultados
    └── FASE2_PROCESAMIENTO_IMAGEN.md  Pipeline de imagen: 8 pasos documentados
```

---

## Responsabilidad de cada archivo

### Raíz del proyecto

| Archivo | Responsabilidad |
|---|---|
| `run.py` | Punto de entrada. Ejecuta `python run.py`. Acepta `--prod` para desactivar debug. |
| `app.py` | Crea la aplicación Flask, registra blueprints, instancia MLP y WeightLogger, carga MNIST al inicio. |
| `trainer.py` | Bucle de entrenamiento con mini-batches, shuffle y callback para WebSocket. También `train_single_step()` para el modo educativo. |
| `requirements.txt` | `numpy`, `flask`, `flask-socketio`, `opencv-python`, `Pillow`, `gevent`, `gevent-websocket`. |
| `.gitignore` | Excluye: `venv/`, `__pycache__/`, archivos `.gz` de MNIST, reportes generados. |

---

### `neural_network/` — Fase 1

| Archivo | Funciones / Clases | Descripción |
|---|---|---|
| `initializer.py` | `initialize_weights(seed)` | He init para W1 `(64,784)`, Xavier para W2 `(10,64)`, sesgos en cero. |
| `activations.py` | `relu(Z)` `relu_derivative(Z)` `softmax(Z)` `sigmoid(Z)` `sigmoid_derivative(Z)` | Funciones de activación vectorizadas con NumPy. Softmax con estabilidad numérica. |
| `loss.py` | `one_hot(y)` `cross_entropy_loss(A2, y_oh)` `accuracy(A2, y)` | Pérdida con clipping `1e-15`. Accuracy por muestra (0 o 1). |
| `network.py` | Clase `MLP` con `forward` `backward` `update` `predict` `predict_proba` `get_weights_snapshot` | Motor completo. Mantiene pesos y historial de métricas. |
| `logger.py` | Clase `WeightLogger` con `log_weights(iter, W1)` | Escribe `W1[0:5, 350:355]` en `.log` solo en iteraciones `{1, 50, 100}`. |
| `debug.py` | `debug_single_sample(network, X, y)` | Imprime todos los valores intermedios con 4 decimales. No modifica pesos. |

---

### `image_processing/` — Fase 2

| Archivo | Función | Entrada → Salida |
|---|---|---|
| `capture.py` | `decode_frame(b64)` | `str` base64 → `ndarray (H,W,3)` BGR |
| `roi_detector.py` | `detect_digit_region(bgr)` | `(H,W,3)` → `(h,w)` binaria (grises→blur→Otsu→contornos→recorte) |
| `preprocessor.py` | `preprocess_image(b64)` | `str` → `( ndarray(784,) float64, ndarray(28,28) uint8 )` |

---

### `web_app/` — Fase 3

| Archivo | Endpoints / Eventos | Descripción |
|---|---|---|
| `routes/training.py` | `POST /train/start` `POST /train/stop` `POST /train/step` `GET /train/status` | Controla el ciclo de entrenamiento. Lanza hilo separado para modo rápido. |
| `routes/prediction.py` | `POST /predict` | Recibe frame base64, ejecuta pipeline completo, retorna dígito + probabilidades + thumbnail. |
| `routes/reports.py` | `GET /reports/loss_chart` `GET /reports/weights_log` | Retorna historial de métricas y contenido del archivo `.log`. |
| `sockets/events.py` | `connect` `disconnect` `request_status` | Eventos WebSocket. El entrenamiento emite `training_update` cada epoch. |
| `templates/index.html` | — | Página principal con 4 secciones: red neuronal, control, gráfica, cámara. |
| `static/js/network_viz.js` | — | Canvas API: dibuja nodos y conexiones. Color azul/rojo según signo del peso. |
| `static/js/camera.js` | — | `getUserMedia`, captura frame, envía base64 a `/predict`. |
| `static/js/charts.js` | — | Gráfica de pérdida con Chart.js. Actualización en tiempo real. Exporta PNG. |
| `static/js/dashboard.js` | — | Conecta WebSocket, orquesta todos los módulos JS, maneja botones de la UI. |
| `static/css/style.css` | — | Tema oscuro completo: layout grid, paneles, botones, barras de probabilidad. |

---

### `data/`

| Archivo | Descripción |
|---|---|
| `mnist_loader.py` | Descarga MNIST desde `ossci-datasets.s3.amazonaws.com` si no existe localmente. Lee archivos `.gz` con `struct` + `gzip`. Normaliza y aplana. Retorna `X_train(60000,784)`, `y_train(60000,)`, `X_test(10000,784)`, `y_test(10000,)`. |
| `raw/*.gz` | Archivos binarios de MNIST. Ignorados por `.gitignore`. Se descargan automáticamente al primer arranque. |

---

### `reports/` — Archivos generados automáticamente

| Archivo | Cuándo se genera | Contenido |
|---|---|---|
| `weight_matrix.log` | Durante el entrenamiento | Sección `W1[0:5, 350:355]` en iteraciones 1, 50 y 100. Con timestamp. |
| `loss_chart.png` | Al pulsar "Exportar Gráfica" en la UI | Imagen PNG de la curva de pérdida y accuracy por epoch. |

---

### `tests/`

| Archivo | Clases de test | Qué verifica |
|---|---|---|
| `test_network.py` | `TestActivations` `TestLoss` `TestForwardShapes` `TestBackwardShapes` `TestGradienteNumerica` | ReLU, Softmax suma=1, Cross-Entropy, shapes de todos los tensores, gradiente `dW2 = outer(A2-y, A1)` |
| `test_preprocessor.py` | `TestPreprocessor` | ROI no vacío, vector shape `(784,)`, valores en `[0,1]`, thumbnail `(28,28)` |

Ejecutar con:
```bash
venv/bin/python -m pytest tests/ -v
```

---

### `docs/`

| Archivo | Contenido |
|---|---|
| `ESTRUCTURA_PROYECTO.md` | Este documento — mapa completo del proyecto |
| `DOCUMENTACION_FASE1.md` | Motor neuronal: matemática de forward/backward, decisiones de diseño, resultados obtenidos (96.37% acc en 2 epochs) |
| `FASE2_PROCESAMIENTO_IMAGEN.md` | Pipeline de imagen: 8 pasos con pseudocódigo, fórmulas matemáticas y diagramas |

---

## Flujo de ejecución al arrancar

```
python run.py
  │
  ▼ app.py → create_app()
  │
  ├─ load_mnist()              ← descarga si no existe, carga 60k imágenes
  ├─ MLP(lr=0.01, seed=42)     ← inicializa pesos He + Xavier
  ├─ WeightLogger(...)         ← crea reports/weight_matrix.log vacío
  │
  ├─ register_blueprint(training_bp)   → /train/*
  ├─ register_blueprint(prediction_bp) → /predict
  ├─ register_blueprint(reports_bp)    → /reports/*
  ├─ register_events(socketio)         → WebSocket
  │
  └─ socketio.run(app, port=5000)
          │
          ▼
   http://localhost:5000   ← index.html se sirve aquí
```

---

## Dependencias entre módulos

```
run.py
  └── app.py
        ├── neural_network.network (MLP)
        ├── neural_network.logger  (WeightLogger)
        ├── data.mnist_loader      (load_mnist)
        ├── web_app.routes.training   → trainer.train / train_single_step
        │                                └── neural_network.network
        │                                └── neural_network.loss
        │                                └── neural_network.logger
        ├── web_app.routes.prediction → image_processing.preprocessor
        │                                └── image_processing.capture
        │                                └── image_processing.roi_detector
        └── web_app.sockets.events
```

---

## Cómo correr el proyecto

```bash
# 1. Crear entorno virtual (solo la primera vez)
python -m venv venv

# 2. Activar entorno
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Arrancar el servidor
python run.py

# 5. Abrir en el navegador
# http://localhost:5000

# 6. Ejecutar tests
python -m pytest tests/ -v
```
