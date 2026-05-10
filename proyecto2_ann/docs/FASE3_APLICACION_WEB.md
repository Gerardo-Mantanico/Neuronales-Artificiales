# Fase 3 — Aplicación Web Interactiva
## Desglose Técnico Completo: Visualización + Control + Tiempo Real

**Proyecto 2 — Inteligencia Artificial 1 | USAC CUNOC**

---

## Índice

1. [Qué problema resuelve esta fase](#1-qué-problema-resuelve-esta-fase)
2. [Arquitectura general del sistema web](#2-arquitectura-general-del-sistema-web)
3. [Mapa de archivos y responsabilidades](#3-mapa-de-archivos-y-responsabilidades)
4. [Módulo 1 — Servidor Flask (`app.py`)](#4-módulo-1--servidor-flask-apppy)
5. [Módulo 2 — Ruta de entrenamiento (`routes/training.py`)](#5-módulo-2--ruta-de-entrenamiento-routestrainingpy)
6. [Módulo 3 — Ruta de predicción (`routes/prediction.py`)](#6-módulo-3--ruta-de-predicción-routespredictionpy)
7. [Módulo 4 — Ruta de reportes (`routes/reports.py`)](#7-módulo-4--ruta-de-reportes-routesreportspy)
8. [Módulo 5 — WebSocket (`sockets/events.py`)](#8-módulo-5--websocket-socketsevents-py)
9. [Módulo 6 — Visualización de la red (`network_viz.js`)](#9-módulo-6--visualización-de-la-red-network_vizjs)
10. [Módulo 7 — Cámara web (`camera.js`)](#10-módulo-7--cámara-web-camerajs)
11. [Módulo 8 — Gráfica de pérdida (`charts.js`)](#11-módulo-8--gráfica-de-pérdida-chartsjs)
12. [Módulo 9 — Orquestador (`dashboard.js`)](#12-módulo-9--orquestador-dashboardjs)
13. [Módulo 10 — Interfaz HTML (`index.html`)](#13-módulo-10--interfaz-html-indexhtml)
14. [Módulo 11 — Estilos (`style.css`)](#14-módulo-11--estilos-stylecss)
15. [Flujo completo: del clic al resultado](#15-flujo-completo-del-clic-al-resultado)
16. [Tabla de requerimientos vs implementación](#16-tabla-de-requerimientos-vs-implementación)

---

## 1. Qué Problema Resuelve Esta Fase

Las fases 1 y 2 son invisibles para el usuario: código Python que entrena pesos y procesa píxeles. La Fase 3 **expone ese trabajo** a través de una interfaz web que permite:

- **Controlar** el entrenamiento (iniciar, detener, ir paso a paso)
- **Observar** cómo cambian los pesos en tiempo real mediante gráficos
- **Interactuar** mostrando un dígito a la cámara y ver la predicción
- **Entender** el proceso de aprendizaje de la máquina paso a paso

Es el componente educativo que exige el enunciado: "herramienta educativa visual, mostrando en tiempo real los cálculos, la actualización de pesos y el flujo de los datos a través de los nodos de la red."

---

## 2. Arquitectura General del Sistema Web

```
NAVEGADOR (cliente)                    PYTHON / FLASK (servidor)
────────────────────────────────────   ──────────────────────────────────────
                                        app.py
  index.html                              ├── MLP (red neuronal)
    │                                     ├── WeightLogger
    ├── network_viz.js  ◄──WebSocket───   ├── X_train, y_train (MNIST)
    ├── charts.js       ◄──WebSocket───   └── socketio
    ├── dashboard.js    ──HTTP POST──►    routes/training.py
    ├── camera.js       ──HTTP POST──►    routes/prediction.py
    └── style.css                         routes/reports.py ◄──HTTP GET──
                                          sockets/events.py
                                            └── emite "training_update"
                                                       │
                                              trainer.py
                                                ├── Fase 1: MLP
                                                └── Fase 2: image_processing
```

### Tecnologías por capa

| Capa | Tecnología | Por qué |
|---|---|---|
| Servidor HTTP | Flask 3.x | Ligero, Python nativo, Jinja2 incluido |
| Tiempo real | Flask-SocketIO + gevent | WebSocket bidireccional sin polling |
| Frontend | HTML5 + CSS3 + JS vanilla | Sin frameworks pesados |
| Gráficas | Chart.js 4.x (CDN) | Solo para UI — no afecta la red neuronal |
| Dibujo de la red | Canvas API (navegador) | Control pixel-perfect, sin librerías extra |
| Cámara | `getUserMedia` API | Estándar del navegador, sin plugins |

---

## 3. Mapa de Archivos y Responsabilidades

```
web_app/
│
├── app.py  (raíz del proyecto)        ← Fábrica: crea Flask, registra todo
│
├── routes/
│   ├── training.py                    ← POST /train/start|stop|step  GET /train/status
│   ├── prediction.py                  ← POST /predict
│   └── reports.py                     ← GET /reports/loss_chart|weights_log
│
├── sockets/
│   └── events.py                      ← Eventos WebSocket del servidor
│
├── templates/
│   └── index.html                     ← Página única (SPA mínimo con Jinja2)
│
└── static/
    ├── css/style.css                  ← Tema oscuro, layout grid, componentes
    └── js/
        ├── network_viz.js             ← Canvas: topología de red + pesos coloreados
        ├── camera.js                  ← getUserMedia + captura + envío al backend
        ├── charts.js                  ← Gráfica loss/accuracy tiempo real + export PNG
        └── dashboard.js               ← WebSocket + botones + orquestación de módulos
```

---

## 4. Módulo 1 — Servidor Flask (`app.py`)

### Responsabilidad

Es la **fábrica de la aplicación**. Se ejecuta una sola vez al iniciar y deja todo listo: carga datos, instancia la red, conecta los módulos.

### Código

```python
def create_app() -> Flask:
    app = Flask(__name__,
                template_folder="web_app/templates",
                static_folder="web_app/static")
    app.config["SECRET_KEY"] = "ann-proyecto2-usac"

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

    # Cargar MNIST una sola vez al arrancar
    X_train, y_train, X_test, y_test = load_mnist()

    # Instanciar motor neuronal y bitácora
    network = MLP(lr=0.01, seed=42)
    logger  = WeightLogger(filepath="reports/weight_matrix.log")

    # Guardar en config para que las rutas accedan a ellos
    app.config["NETWORK"]  = network
    app.config["LOGGER"]   = logger
    app.config["SOCKETIO"] = socketio
    app.config["X_TRAIN"]  = X_train
    app.config["Y_TRAIN"]  = y_train

    # Registrar blueprints y eventos
    app.register_blueprint(training_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(reports_bp)
    register_events(socketio)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app, socketio
```

### Pseudocódigo

```
FUNCIÓN create_app():

    app = crear_aplicacion_flask(
        templates = "web_app/templates",
        static    = "web_app/static"
    )

    socketio = crear_websocket(app, modo_async="gevent")

    # Una sola carga al inicio — compartida por todas las rutas
    X_train, y_train, X_test, y_test = cargar_mnist()

    red     = MLP(lr=0.01, semilla=42)    ← pesos iniciales reproducibles
    bitácora = WeightLogger(...)           ← crea/limpia el archivo .log

    # Guardar estado global accesible desde cualquier ruta
    app.config["NETWORK"]  = red
    app.config["LOGGER"]   = bitácora
    app.config["SOCKETIO"] = socketio
    app.config["X_TRAIN"]  = X_train
    app.config["Y_TRAIN"]  = y_train

    registrar_rutas(training_bp, prediction_bp, reports_bp)
    registrar_eventos_websocket(socketio)

    ruta "/" → renderizar "index.html"

    RETORNAR app, socketio
```

### Decisión de diseño: `app.config` como estado global

Las rutas de Flask no pueden compartir variables directamente entre sí (son funciones independientes). `app.config` actúa como **registro central** donde se almacenan la red neuronal, el logger y los datos. Cada ruta recupera lo que necesita con `current_app.config["NETWORK"]`.

---

## 5. Módulo 2 — Ruta de Entrenamiento (`routes/training.py`)

### Endpoints implementados

| Método | URL | Función |
|---|---|---|
| `POST` | `/train/start` | Inicia entrenamiento en hilo separado |
| `POST` | `/train/stop` | Señaliza parada al hilo activo |
| `POST` | `/train/step` | Ejecuta exactamente un paso (modo educativo) |
| `GET` | `/train/status` | Retorna estado actual de la red |

---

### `POST /train/start` — Modo Rápido

```python
_training_thread: threading.Thread = None
_stop_flag = threading.Event()

@bp.route("/start", methods=["POST"])
def start_training():
    global _training_thread, _stop_flag

    # Evitar doble entrenamiento simultáneo
    if _training_thread and _training_thread.is_alive():
        return jsonify({"error": "Entrenamiento ya en curso"}), 409

    data   = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 20))
    lr     = float(data.get("lr", 0.01))
    _stop_flag.clear()

    app = current_app._get_current_object()

    def run():
        network  = app.config["NETWORK"]
        socketio = app.config["SOCKETIO"]
        # ...
        def emit_progress(epoch, loss, acc):
            if _stop_flag.is_set(): return
            socketio.emit("training_update", {
                "epoch": epoch, "loss": round(loss, 6),
                "accuracy": round(acc, 6),
                "weights_sample": {"W2": network.W2[0:5, 0:5].tolist()},
            })

        train(network, X_train, y_train,
              epochs=epochs, logger=logger, callback=emit_progress)

        socketio.emit("training_done", {...})

    _training_thread = threading.Thread(target=run, daemon=True)
    _training_thread.start()
    return jsonify({"status": "started"})
```

#### Pseudocódigo

```
ENDPOINT POST /train/start:

    SI hilo de entrenamiento ya está corriendo:
        RETORNAR error 409 "ya en curso"

    epochs = parámetro del body JSON (default 20)
    lr     = parámetro del body JSON (default 0.01)

    limpiar_bandera_de_parada()

    DEFINIR función run():
        red.lr = lr
        limpiar historial de métricas

        DEFINIR callback emit_progress(epoch, loss, acc):
            SI bandera_de_parada activa: salir
            enviar por WebSocket "training_update" con:
              { epoch, loss, accuracy, W2[0:5, 0:5] }

        entrenar(red, X_train, y_train, epochs, callback=emit_progress)

        enviar por WebSocket "training_done" con métricas finales

    lanzar hilo(run, daemon=True)   ← no bloquea el servidor
    RETORNAR { "status": "started" }
```

#### Por qué un hilo separado (`threading.Thread`)

El entrenamiento de 20 epochs sobre 60,000 imágenes tarda varios minutos. Si se ejecutara directamente en el endpoint, el servidor HTTP quedaría bloqueado y no podría responder otras peticiones (cámara, paso a paso, etc.). El hilo `daemon=True` se termina automáticamente si el proceso principal muere.

---

### `POST /train/stop`

```python
@bp.route("/stop", methods=["POST"])
def stop_training():
    _stop_flag.set()
    return jsonify({"status": "stopping"})
```

#### Pseudocódigo

```
ENDPOINT POST /train/stop:

    activar_bandera_de_parada()
    # El hilo de entrenamiento la revisa en cada callback
    # y deja de emitir eventos / se detiene en la siguiente iteración

    RETORNAR { "status": "stopping" }
```

---

### `POST /train/step` — Modo Educativo

```python
@bp.route("/step", methods=["POST"])
def step():
    idx = network.iteration % len(X_train)   # siguiente muestra en secuencia
    X_i = X_train[idx]
    y_i = int(y_train[idx])
    resultado = train_single_step(network, X_i, y_i, logger)
    return jsonify(resultado)
```

#### Pseudocódigo

```
ENDPOINT POST /train/step:

    idx = iteracion_actual MOD total_muestras
    # usa el índice circular para recorrer el dataset secuencialmente

    X_i = imagen en posición idx
    y_i = etiqueta en posición idx

    resultado = ejecutar_un_paso(red, X_i, y_i, logger)
    # Internamente: forward → loss → backward → update → log si iter in {1,50,100}

    RETORNAR resultado como JSON:
    {
      "iteration":     número de paso actual,
      "loss":          pérdida de esta muestra,
      "accuracy":      1 o 0,
      "prediction":    dígito predicho (0–9),
      "probabilities": [p0, p1, ..., p9],    ← Softmax completo
      "weights_sample": {
          "W1_sample": W1[0:5, 0:5],
          "W2_sample": W2[0:5, 0:5]
      },
      "activations": {
          "A1": activaciones capa oculta (64 valores),
          "A2": probabilidades finales  (10 valores)
      }
    }
```

#### Para qué sirven `activations` en la respuesta

El frontend (`network_viz.js`) usa `A1` y `A2` para **iluminar** los nodos cuya activación es mayor, mostrando visualmente el flujo de la señal a través de la red en ese paso concreto.

---

### `GET /train/status`

```python
@bp.route("/status", methods=["GET"])
def status():
    network = current_app.config["NETWORK"]
    running = _training_thread is not None and _training_thread.is_alive()
    return jsonify({
        "running":          running,
        "iteration":        network.iteration,
        "loss_history":     network.loss_history,
        "accuracy_history": network.accuracy_history,
    })
```

Permite al frontend **recuperar el estado** si recarga la página o se reconecta el WebSocket.

---

## 6. Módulo 3 — Ruta de Predicción (`routes/prediction.py`)

### Endpoint

| Método | URL | Función |
|---|---|---|
| `POST` | `/predict` | Recibe frame base64, lo procesa con Fase 2, predice con Fase 1 |

### Código

```python
@bp.route("", methods=["POST"])
def predict():
    data = request.get_json()
    network = current_app.config["NETWORK"]

    # Fase 2: base64 → vector (784,)
    vector, thumbnail = preprocess_image(data["image"])

    # Fase 1: forward propagation → probabilidades
    probabilities = network.predict_proba(vector)
    digit = int(np.argmax(probabilities))

    # Codificar thumbnail para mostrar en la UI
    _, buffer = cv2.imencode(".png", thumbnail)
    thumb_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "digit":         digit,
        "probabilities": [round(float(p), 6) for p in probabilities],
        "thumbnail_b64": f"data:image/png;base64,{thumb_b64}",
    })
```

### Pseudocódigo

```
ENDPOINT POST /predict:

    Body JSON esperado: { "image": "<base64 del frame>" }

    SI no hay campo "image":
        RETORNAR error 400

    red = obtener_red_de_config()

    INTENTAR:
        # ── Fase 2 ─────────────────────────────────────────────
        vector, thumbnail = preprocess_image(data["image"])
        # vector:    (784,) float64  [0.0–1.0]
        # thumbnail: (28,28) uint8   [0–255]
    EXCEPTO error:
        RETORNAR error 500 con mensaje

    # ── Fase 1 ─────────────────────────────────────────────────
    probabilidades = red.predict_proba(vector)
    # Internamente: forward(X) → A2 (Softmax, 10 valores)

    digito = argmax(probabilidades)   ← clase con mayor probabilidad

    # ── Thumbnail para la UI ────────────────────────────────────
    png_bytes = codificar_imagen_png(thumbnail)
    thumb_b64 = base64_encode(png_bytes)

    RETORNAR JSON:
    {
      "digit":         digito (0–9),
      "probabilities": [p0, p1, ..., p9],   ← 6 decimales
      "thumbnail_b64": "data:image/png;base64,..."
    }
```

### Punto de integración Fase 1 + Fase 2

Este endpoint es exactamente donde las dos fases se unen:

```
POST /predict
  │
  ├── preprocess_image(base64)      ← Fase 2 completa
  │     └── retorna vector (784,)
  │
  └── network.predict_proba(vector) ← Fase 1: solo forward, sin entrenar
        └── retorna A2 (10,)
```

---

## 7. Módulo 4 — Ruta de Reportes (`routes/reports.py`)

### Endpoints

| Método | URL | Retorna |
|---|---|---|
| `GET` | `/reports/loss_chart` | JSON con listas `epochs`, `loss`, `accuracy` |
| `GET` | `/reports/weights_log` | JSON con contenido del archivo `weight_matrix.log` |

### Pseudocódigo

```
ENDPOINT GET /reports/loss_chart:

    red = obtener_red_de_config()
    RETORNAR JSON:
    {
      "epochs":   [1, 2, 3, ...],
      "loss":     [0.2435, 0.1175, ...],
      "accuracy": [0.9263, 0.9637, ...]
    }
    # Usado por charts.js para reconstruir la gráfica si el usuario recarga

ENDPOINT GET /reports/weights_log:

    INTENTAR leer "reports/weight_matrix.log"
    RETORNAR { "log": contenido_del_archivo }
    SI no existe: RETORNAR error 404 "Entrena primero"
```

---

## 8. Módulo 5 — WebSocket (`sockets/events.py`)

### Por qué WebSocket y no polling HTTP

Con polling el frontend haría `GET /train/status` cada segundo, lo que genera latencia y carga innecesaria. WebSocket mantiene **una conexión persistente**: el servidor empuja datos al cliente exactamente cuando hay algo nuevo (al finalizar cada epoch), sin esperar peticiones.

### Eventos implementados

```python
def register_events(socketio):

    @socketio.on("connect")
    def on_connect():
        print("Cliente conectado")

    @socketio.on("disconnect")
    def on_disconnect():
        print("Cliente desconectado")

    @socketio.on("request_status")
    def on_request_status():
        # El cliente pide el estado al reconectarse
        socketio.emit("training_update", {
            "epoch":    len(network.loss_history),
            "loss":     network.loss_history[-1] if network.loss_history else 0,
            "accuracy": network.accuracy_history[-1] if network.accuracy_history else 0,
        })
```

### Pseudocódigo del ciclo WebSocket completo

```
[Servidor] ── al terminar cada epoch en trainer.py:
    callback(epoch, loss, acc)
        └── socketio.emit("training_update", {epoch, loss, acc, W2_sample})

[Cliente] ── dashboard.js escucha:
    socket.on("training_update", (data) => {
        actualizar_dashboard(data.epoch, data.loss, data.accuracy)
        Charts.update(data.epoch, data.loss, data.accuracy)
        NetViz.updateWeights({ W2_sample: data.weights_sample.W2 })
    })

[Servidor] ── al terminar el entrenamiento completo:
    socketio.emit("training_done", {epochs, final_loss, final_accuracy})

[Cliente] ── dashboard.js escucha:
    socket.on("training_done", (data) => {
        habilitar_boton_entrenar()
        mostrar_toast("Entrenamiento completado")
    })
```

### Eventos que circulan por el WebSocket

| Dirección | Evento | Payload | Cuándo |
|---|---|---|---|
| Servidor → Cliente | `training_update` | `{epoch, loss, accuracy, weights_sample}` | Al finalizar cada epoch |
| Servidor → Cliente | `training_done` | `{epochs, final_loss, final_accuracy}` | Al terminar entrenamiento |
| Cliente → Servidor | `request_status` | ninguno | Al reconectar / cargar página |

---

## 9. Módulo 6 — Visualización de la Red (`network_viz.js`)

### Responsabilidad

Dibuja en un `<canvas>` la topología de la red neuronal: 3 columnas de nodos (entrada, oculta, salida) con líneas que representan los pesos. El color y grosor de cada línea cambia en tiempo real.

### Estructura del módulo (IIFE pattern)

```javascript
const NetViz = (() => {
    // Estado privado
    let canvas, ctx
    let nodesInput = [], nodesHidden = [], nodesOutput = []

    // API pública
    return { init, updateWeights }
})()
```

El patrón IIFE (Immediately Invoked Function Expression) encapsula el estado interno — `canvas`, `ctx` y las posiciones de los nodos no son accesibles desde fuera del módulo.

### Función `init(canvasId)`

```javascript
function init(canvasId) {
    canvas = document.getElementById(canvasId)
    ctx    = canvas.getContext("2d")
    _calcPositions()   // calcula coordenadas (x,y) de cada nodo
    draw({})           // dibuja estado inicial vacío
}
```

#### Pseudocódigo de `_calcPositions()`

```
FUNCIÓN _calcPositions():

    W = ancho del canvas   (480px)
    H = alto del canvas    (420px)

    columnas = [20 nodos, 64 nodos, 10 nodos]
    posiciones_x = [W*0.15, W*0.50, W*0.85]
       # capa entrada: x=72px, oculta: x=240px, salida: x=408px

    PARA cada capa (ci) y número de nodos (n):
        espaciado = H / (n + 1)
        PARA i = 0..n-1:
            agregar { x: posiciones_x[ci], y: espaciado*(i+1) }

    # Resultado: nodesInput(20), nodesHidden(64), nodesOutput(10)
    # con coordenadas uniformemente distribuidas en vertical
```

### Función `draw(data)`

```javascript
function draw(data) {
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // 1. Conexiones (detrás de los nodos)
    _drawConnections(nodesInput, nodesHidden.slice(0, 10), data.W1_sample, 0.3)
    _drawConnections(nodesHidden.slice(0, 10), nodesOutput, data.W2_sample, 0.5)

    // 2. Nodos (encima de las conexiones)
    _drawLayer(nodesInput,  "#64b5f6", data.input_activations,  6)
    _drawLayer(nodesHidden, "#81c784", data.hidden_activations, 5)
    _drawLayer(nodesOutput, "#e57373", data.output_activations, 8)

    // 3. Etiquetas de capa
    _drawLabels()
}
```

### Función `_drawConnections` — color por peso

```javascript
function _drawConnections(from, to, weightMatrix, alpha) {
    from.forEach((fn, fi) => {
        to.forEach((tn, ti) => {
            let weight = weightMatrix?.[ti]?.[fi] ?? 0
            const intensidad = Math.min(Math.abs(weight) * 5, 1)
            const grosor     = 0.5 + intensidad * 2   // 0.5px → 2.5px

            ctx.beginPath()
            ctx.moveTo(fn.x, fn.y)
            ctx.lineTo(tn.x, tn.y)
            ctx.lineWidth = grosor

            // Azul: peso positivo | Rojo: peso negativo
            if (weight >= 0)
                ctx.strokeStyle = `rgba(33, 150, 243, ${alpha * intensidad + 0.05})`
            else
                ctx.strokeStyle = `rgba(244, 67, 54,  ${alpha * intensidad + 0.05})`

            ctx.stroke()
        })
    })
}
```

#### Pseudocódigo

```
FUNCIÓN _drawConnections(desde, hacia, pesos, opacidad_base):

    PARA cada nodo_origen (fi) en desde:
        PARA cada nodo_destino (ti) en hacia:

            peso      = pesos[ti][fi]  (o 0 si no hay datos aún)
            intensidad = min(|peso| * 5, 1.0)
            grosor     = 0.5 + intensidad * 2.0

            trazar_línea(nodo_origen.x, nodo_origen.y,
                         nodo_destino.x, nodo_destino.y)
            ancho_línea = grosor

            SI peso >= 0:
                color = azul con opacidad = opacidad_base * intensidad
            SINO:
                color = rojo con opacidad = opacidad_base * intensidad

            dibujar_línea()
```

### Función `_drawLayer` — brillo por activación

```javascript
function _drawLayer(nodes, colorBase, activations, radius) {
    nodes.forEach((n, i) => {
        const act = activations ? Math.min(activations[i] || 0, 1) : 0

        ctx.beginPath()
        ctx.arc(n.x, n.y, radius, 0, Math.PI * 2)
        ctx.fillStyle = _interpolateColor("#1a1a2e", colorBase, act)
        // act=0 → color oscuro (#1a1a2e)
        // act=1 → color pleno del layer
        ctx.fill()
    })
}
```

#### Pseudocódigo de interpolación de color

```
FUNCIÓN _interpolateColor(color_oscuro, color_activo, t):
    # t: activación en [0.0, 1.0]

    r = redondear(r_oscuro + (r_activo - r_oscuro) * t)
    g = redondear(g_oscuro + (g_activo - g_oscuro) * t)
    b = redondear(b_oscuro + (b_activo - b_oscuro) * t)

    RETORNAR "rgb(r, g, b)"
    # t=0.0 → nodo casi negro (inactivo)
    # t=1.0 → nodo en color pleno (muy activo)
```

---

## 10. Módulo 7 — Cámara Web (`camera.js`)

### Responsabilidad

Gestiona el ciclo completo de la cámara: activar, capturar un frame, convertirlo a base64 y enviarlo al backend para predicción.

### Estructura del módulo

```javascript
const Camera = (() => {
    let videoEl, stream

    return { start, stop, capture, captureAndPredict }
})()
```

### Función `start(videoElementId)`

```javascript
async function start(videoElementId) {
    videoEl = document.getElementById(videoElementId)
    stream  = await navigator.mediaDevices.getUserMedia({ video: true })
    videoEl.srcObject = stream
    await videoEl.play()
}
```

#### Pseudocódigo

```
FUNCIÓN ASÍNCRONA start(id_elemento_video):

    videoEl = obtener_elemento_html(id_elemento_video)

    INTENTAR:
        stream = ESPERAR pedir_acceso_camara({ video: true })
        # El navegador muestra diálogo de permiso al usuario
        # Si el usuario acepta: stream contiene el flujo de video
        # Si rechaza: lanza NotAllowedError

        videoEl.srcObject = stream   ← conectar stream al <video>
        ESPERAR videoEl.play()       ← iniciar reproducción

    EXCEPTO error:
        mostrar_alerta("No se pudo acceder a la cámara: " + error)
```

### Función `capture()`

```javascript
async function capture() {
    const tmpCanvas = document.createElement("canvas")
    tmpCanvas.width  = videoEl.videoWidth
    tmpCanvas.height = videoEl.videoHeight
    tmpCanvas.getContext("2d").drawImage(videoEl, 0, 0)
    return tmpCanvas.toDataURL("image/jpeg", 0.85)
}
```

#### Pseudocódigo

```
FUNCIÓN ASÍNCRONA capture():

    SI cámara no está activa:
        mostrar_alerta("Activa la cámara primero")
        RETORNAR null

    # Crear canvas temporal del mismo tamaño que el video
    canvas_temp = crear_canvas(ancho=video.ancho, alto=video.alto)

    # Copiar el frame actual del <video> al canvas
    canvas_temp.contexto2D.dibujarImagen(videoEl, x=0, y=0)

    # Convertir canvas a base64 JPEG (calidad 85%)
    base64_jpeg = canvas_temp.toDataURL("image/jpeg", 0.85)
    # Resultado: "data:image/jpeg;base64,/9j/4AAQ..."

    RETORNAR base64_jpeg
```

### Función `captureAndPredict(onResult)`

```javascript
async function captureAndPredict(onResult) {
    const base64 = await capture()

    const resp = await fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ image: base64 }),
    })

    const data = await resp.json()
    onResult(data)   // callback con { digit, probabilities, thumbnail_b64 }
}
```

#### Pseudocódigo

```
FUNCIÓN ASÍNCRONA captureAndPredict(callback_resultado):

    base64 = ESPERAR capture()
    SI base64 es null: salir

    respuesta = ESPERAR enviar_POST("/predict", { "image": base64 })
    # El backend ejecuta: Fase2(base64) → vector → Fase1.forward() → probabilidades

    SI respuesta tiene error:
        mostrar_en_consola("Error en predicción")
        salir

    datos = ESPERAR parsear_json(respuesta)
    # datos = { digit, probabilities, thumbnail_b64 }

    callback_resultado(datos)
    # dashboard.js actualiza: stat-pred, barras Softmax, thumbnail
```

---

## 11. Módulo 8 — Gráfica de Pérdida (`charts.js`)

### Responsabilidad

Mantiene una gráfica de línea con **dos ejes Y**: pérdida (izquierda, rojo) y accuracy (derecha, azul). Se actualiza en tiempo real desde el WebSocket. Puede exportarse como PNG.

### Configuración inicial (`init`)

```javascript
lossChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],        // números de epoch [1, 2, 3, ...]
        datasets: [
            { label: "Loss",     yAxisID: "yLoss", ... },
            { label: "Accuracy", yAxisID: "yAcc",  ... }
        ]
    },
    options: {
        animation: false,   // sin animación para máxima fluidez
        scales: {
            yLoss: { position: "left",  ... },
            yAcc:  { position: "right", min: 0, max: 1, ... }
        }
    }
})
```

### Función `update(epoch, loss, accuracy)`

```javascript
function update(epoch, loss, accuracy) {
    lossChart.data.labels.push(epoch)
    lossChart.data.datasets[0].data.push(loss)
    lossChart.data.datasets[1].data.push(accuracy)
    lossChart.update("none")   // "none" = sin animación de transición
}
```

#### Pseudocódigo

```
FUNCIÓN update(epoch, loss, accuracy):

    agregar epoch    al eje X de la gráfica
    agregar loss     al dataset "Loss"
    agregar accuracy al dataset "Accuracy"

    redibujar_grafica(sin_animacion=true)
    # "none" evita que la gráfica "salte" al agregar puntos rápido
```

### Función `exportPNG()`

```javascript
function exportPNG() {
    const url = lossChart.toBase64Image("image/png", 1.0)
    const a   = document.createElement("a")
    a.href     = url
    a.download = "loss_chart.png"
    a.click()           // dispara descarga automática en el navegador
}
```

#### Pseudocódigo

```
FUNCIÓN exportPNG():

    imagen_base64 = grafica.convertir_a_png(calidad=1.0)
    # Chart.js renderiza el canvas a PNG

    enlace = crear_elemento_a()
    enlace.href     = imagen_base64
    enlace.download = "loss_chart.png"
    simular_clic(enlace)
    # El navegador inicia la descarga del archivo
```

---

## 12. Módulo 9 — Orquestador (`dashboard.js`)

### Responsabilidad

Es el **punto de entrada del frontend**. Se ejecuta cuando carga la página, inicializa todos los módulos JS, conecta el WebSocket y registra los listeners de todos los botones.

### Flujo al cargar la página

```javascript
document.addEventListener("DOMContentLoaded", () => {

    // 1. Inicializar módulos de visualización
    NetViz.init("canvas-network")
    Charts.init("canvas-loss")

    // 2. Conectar WebSocket
    const socket = io()

    // 3. Escuchar eventos del servidor
    socket.on("training_update", handler)
    socket.on("training_done",   handler)

    // 4. Registrar botones
    btn-train   → fetch POST /train/start
    btn-stop    → fetch POST /train/stop
    btn-step    → fetch POST /train/step → actualizar UI
    btn-camera  → Camera.start()
    btn-capture → Camera.captureAndPredict() → actualizar UI
    btn-export  → Charts.exportPNG()
})
```

### Pseudocódigo de cada acción

```
── BOTÓN "Entrenar (Rápido)" ───────────────────────────────
AL HACER CLIC en btn-train:
    epochs = valor del input-epochs
    lr     = valor del input-lr
    Charts.reset()                  ← limpiar gráfica anterior
    deshabilitar btn-train
    habilitar   btn-stop
    ENVIAR POST /train/start { epochs, lr }
    # El servidor arranca el hilo y emitirá events por WebSocket

── BOTÓN "Detener" ─────────────────────────────────────────
AL HACER CLIC en btn-stop:
    ENVIAR POST /train/stop
    deshabilitar btn-stop
    habilitar   btn-train

── BOTÓN "Paso a Paso" ─────────────────────────────────────
AL HACER CLIC en btn-step:
    respuesta = ESPERAR POST /train/step
    datos = parsear_json(respuesta)

    actualizar stat-epoch    = datos.iteration
    actualizar stat-loss     = datos.loss
    actualizar stat-acc      = datos.accuracy
    actualizar stat-pred     = datos.prediction

    NetViz.updateWeights({
        W2_sample:         datos.weights_sample.W2_sample,
        hidden_activations: datos.activations.A1,
        output_activations: datos.activations.A2
    })
    actualizar_barras_softmax(datos.probabilities)

── BOTÓN "Activar Cámara" ──────────────────────────────────
AL HACER CLIC en btn-camera:
    Camera.start("video-feed")
    deshabilitar btn-camera
    habilitar   btn-capture

── BOTÓN "Capturar y Predecir" ─────────────────────────────
AL HACER CLIC en btn-capture:
    Camera.captureAndPredict( (datos) => {
        actualizar stat-pred         = datos.digit
        actualizar img-thumbnail.src = datos.thumbnail_b64
        actualizar_barras_softmax(datos.probabilities)
    })

── EVENTO WebSocket "training_update" ──────────────────────
AL RECIBIR datos del servidor:
    actualizar stat-epoch = datos.epoch
    actualizar stat-loss  = datos.loss
    actualizar stat-acc   = datos.accuracy * 100 + "%"
    Charts.update(datos.epoch, datos.loss, datos.accuracy)
    NetViz.updateWeights({ W2_sample: datos.weights_sample.W2 })

── EVENTO WebSocket "training_done" ────────────────────────
AL RECIBIR datos del servidor:
    habilitar   btn-train
    deshabilitar btn-stop
    mostrar_toast("Entrenamiento completado. Loss: " + datos.final_loss)
```

### Función `_updateProbBars(probs)`

```javascript
function _updateProbBars(probs) {
    probs.forEach((p, i) => {
        document.getElementById(`prob-bar-${i}`).style.width =
            (p * 100).toFixed(1) + "%"
        document.getElementById(`prob-val-${i}`).textContent =
            (p * 100).toFixed(1) + "%"
    })
}
```

Actualiza las 10 barras de probabilidad Softmax (una por cada dígito 0–9) con animación CSS `transition: width 0.3s ease`.

---

## 13. Módulo 10 — Interfaz HTML (`index.html`)

### Layout de la página

```
┌─────────────────────────────────────────────────────────┐
│ HEADER: "Red Neuronal desde Cero — 784 → 64 → 10"       │
├──────────────────┬───────────────┬──────────────────────┤
│                  │               │                      │
│  Topología       │   Control     │  Curva de Pérdida    │
│  de la Red       │   - Épocas    │  y Accuracy          │
│  (Canvas)        │   - LR        │  (Chart.js)          │
│                  │   - Botones   │                      │
│                  │   - Stats     │                      │
├──────────────────┴───────────────┴──────────────────────┤
│  Cámara Web     │  Imagen 28×28  │  Probabilidades      │
│  (video)        │  (thumbnail)   │  Softmax (barras)    │
└─────────────────┴────────────────┴──────────────────────┘
```

### Estructura HTML relevante

```html
<!-- Grid principal: 3 columnas -->
<div class="main-grid">

  <!-- Columna 1: Canvas de la red -->
  <div class="panel">
    <canvas id="canvas-network" width="480" height="420"></canvas>
  </div>

  <!-- Columna 2: Panel de control -->
  <div class="panel control-panel">
    <input id="input-epochs" type="number" value="20" />
    <input id="input-lr"     type="number" value="0.01" />
    <button id="btn-train">▶ Entrenar (Rápido)</button>
    <button id="btn-stop"  disabled>■ Detener</button>
    <button id="btn-step">↳ Paso a Paso</button>
    <button id="btn-camera">📷 Activar Cámara</button>
    <button id="btn-capture" disabled>🔍 Capturar y Predecir</button>
    <button id="btn-export">↓ Exportar Gráfica</button>
    <!-- Stats: stat-epoch, stat-loss, stat-acc, stat-pred -->
  </div>

  <!-- Columna 3: Gráfica de pérdida -->
  <div class="panel">
    <canvas id="canvas-loss"></canvas>
  </div>

  <!-- Fila 2: Cámara + Thumbnail + Barras Softmax -->
  <div class="camera-section">
    <video id="video-feed"></video>
    <img   id="img-thumbnail" />
    <div class="prob-bars">
      <!-- 10 barras generadas con Jinja2 -->
      {% for i in range(10) %}
        <div id="prob-bar-{{ i }}"></div>
        <span id="prob-val-{{ i }}">0.0%</span>
      {% endfor %}
    </div>
  </div>

</div>

<!-- Carga de JS en orden de dependencias -->
<script src=".../network_viz.js"></script>   <!-- sin dependencias -->
<script src=".../camera.js"></script>         <!-- sin dependencias -->
<script src=".../charts.js"></script>         <!-- necesita Chart.js CDN -->
<script src=".../dashboard.js"></script>      <!-- necesita todos los anteriores -->
```

### Por qué Jinja2 para las barras Softmax

```html
{% for i in range(10) %}
  <div class="prob-row">
    <span class="prob-digit">{{ i }}</span>
    <div class="prob-track">
      <div class="prob-fill" id="prob-bar-{{ i }}"></div>
    </div>
    <span class="prob-val" id="prob-val-{{ i }}">0.0%</span>
  </div>
{% endfor %}
```

El bucle Jinja2 genera los 10 elementos HTML en el servidor, con IDs numerados `prob-bar-0` a `prob-bar-9`. JavaScript los actualiza por ID sin necesidad de crear elementos dinámicamente.

---

## 14. Módulo 11 — Estilos (`style.css`)

### Componentes CSS principales

| Selector | Rol |
|---|---|
| `.main-grid` | Grid 3 columnas + 2 filas para el layout principal |
| `.camera-section` | Grid 3 columnas que ocupa las 3 columnas del grid padre |
| `.panel` | Tarjeta con fondo oscuro `#161b22` y borde `#30363d` |
| `.btn`, `.btn-primary/danger/info/secondary` | Sistema de botones con 4 variantes de color |
| `.stats-grid` | Grid 2×2 para las 4 estadísticas (epoch, loss, acc, predicción) |
| `.stat-card` | Tarjeta individual de estadística con label pequeño y valor grande |
| `.prob-bars / .prob-track / .prob-fill` | Sistema de barras Softmax con `transition: width 0.3s` |
| `.prediction-display` | Dígito predicho en fuente 5rem color verde `#3fb950` |
| `#toast` | Notificación emergente en esquina inferior derecha |
| `#img-thumbnail` | `image-rendering: pixelated` para ver la imagen 28×28 ampliada sin blur |

### Decisión: `image-rendering: pixelated`

```css
#img-thumbnail {
    width: 112px;
    height: 112px;
    image-rendering: pixelated;
}
```

La imagen es de 28×28 píxeles mostrada en 112×112 (4× zoom). Sin `pixelated`, el navegador aplica interpolación bilineal que difumina los píxeles. Con `pixelated`, cada píxel se muestra como un cuadrado sólido — esto permite ver exactamente qué ve la red neuronal.

---

## 15. Flujo Completo: del Clic al Resultado

### Flujo A — Entrenamiento rápido

```
Usuario pulsa "▶ Entrenar"
  │
  ▼ dashboard.js
  fetch POST /train/start { epochs:20, lr:0.01 }
  │
  ▼ routes/training.py
  Thread(target=run).start()   ← no bloquea el servidor
  return { "status": "started" }
  │
  ▼ (en paralelo, en el hilo de entrenamiento)
  trainer.train(network, X_train, y_train, callback=emit_progress)
    │
    │  por cada epoch:
    │    forward → loss → backward → update × 60,000
    │    callback(epoch, loss, acc)
    │      └── socketio.emit("training_update", {..., W2_sample})
    │
    ▼ (en el navegador, por WebSocket)
    socket.on("training_update") → dashboard.js
      ├── actualizar stat-epoch, stat-loss, stat-acc
      ├── Charts.update(epoch, loss, acc)          → gráfica se extiende
      └── NetViz.updateWeights({ W2_sample })      → conexiones cambian color
  │
  ▼ (al terminar todas las epochs)
  socketio.emit("training_done")
  dashboard.js → habilitar btn-train, mostrar toast
```

### Flujo B — Predicción con cámara

```
Usuario pulsa "📷 Activar Cámara"
  └── Camera.start() → getUserMedia → <video> muestra stream en vivo

Usuario pulsa "🔍 Capturar y Predecir"
  │
  ▼ camera.js
  Canvas temporal copia frame del <video>
  base64 = canvas.toDataURL("image/jpeg", 0.85)
  │
  ▼ fetch POST /predict { image: base64 }
  │
  ▼ routes/prediction.py
  vector, thumbnail = preprocess_image(base64)   ← Fase 2
    │  grises → blur → Otsu → contornos → 28×28 → flatten → /255
    │
  probabilities = network.predict_proba(vector)  ← Fase 1
    │  forward: Z1→A1(ReLU)→Z2→A2(Softmax)
    │
  return { digit, probabilities[10], thumbnail_b64 }
  │
  ▼ dashboard.js recibe respuesta
  stat-pred.textContent    = datos.digit          → muestra "7"
  img-thumbnail.src        = datos.thumbnail_b64  → muestra imagen 28×28
  _updateProbBars(datos.probabilities)            → barras se animan
```

### Flujo C — Modo educativo (paso a paso)

```
Usuario pulsa "↳ Paso a Paso"
  │
  ▼ fetch POST /train/step
  │
  ▼ routes/training.py
  idx = network.iteration % len(X_train)
  resultado = train_single_step(network, X_i, y_i)
    │  UN forward + backward + update
    │
  return { iteration, loss, accuracy, prediction,
           probabilities, weights_sample, activations }
  │
  ▼ dashboard.js
  actualizar todos los stats
  NetViz.updateWeights({
      W2_sample:          pesos actualizados,
      hidden_activations: A1 (64 valores),  → nodos ocultos se iluminan
      output_activations: A2 (10 valores)   → nodos de salida se iluminan
  })
  _updateProbBars(probabilities)
```

---

## 16. Tabla de Requerimientos vs Implementación

| Requerimiento del enunciado | Archivo | Estado |
|---|---|---|
| Representación gráfica de la red (nodos + conexiones) | `network_viz.js` | Implementado |
| Conexiones cambian de color/grosor según peso | `network_viz.js → _drawConnections` | Implementado |
| Modo Ejecución Rápida (máxima velocidad) | `routes/training.py → start_training` | Implementado |
| Modo Cámara Lenta / paso a paso con botón "siguiente" | `routes/training.py → step` + `dashboard.js` | Implementado |
| Dashboard: Epoch actual e iteración | `dashboard.js → stat-epoch` | Implementado |
| Dashboard: Valor del Error (Loss) | `dashboard.js → stat-loss` | Implementado |
| Dashboard: Precisión (Accuracy) | `dashboard.js → stat-acc` | Implementado |
| Gráfica de función de costo en tiempo real | `charts.js → update()` vía WebSocket | Implementado |
| Gráfica exportable | `charts.js → exportPNG()` | Implementado |
| Activar cámara del dispositivo | `camera.js → start()` | Implementado |
| Capturar imagen y predecir dígito | `camera.js → captureAndPredict()` | Implementado |
| Mostrar imagen procesada 28×28 | `index.html → img-thumbnail` | Implementado |
| Mostrar probabilidades Softmax | `dashboard.js → _updateProbBars()` | Implementado |
| Comunicación en tiempo real | `sockets/events.py` + `dashboard.js → io()` | Implementado |
| Aplicación en Python | Flask (backend 100% Python) | Implementado |
