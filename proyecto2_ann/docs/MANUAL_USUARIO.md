# Manual de Usuario
## ANN desde Cero — Proyecto 2 | Inteligencia Artificial 1 | USAC CUNOC
**Ing. Daniel González | Autor: Gerardo Mantanico**

---

## 1. Requisitos del Sistema

### Software necesario
- **Python 3.10 o superior** — [python.org](https://python.org)
- **Navegador moderno** — Chrome 90+, Firefox 88+, Edge 90+ (con soporte a getUserMedia)
- **Cámara web** — integrada en laptop o externa por USB

### Hardware recomendado
- RAM: mínimo 4 GB (MNIST carga ~180 MB en memoria)
- CPU: cualquier procesador moderno (el entrenamiento tarda ~2 min/época)
- Cámara: cualquier resolución, mínimo 480p

---

## 2. Instalación

Abrir una terminal en la carpeta del proyecto y ejecutar:

```bash
# Paso 1: Crear el entorno virtual (solo la primera vez)
python -m venv venv

# Paso 2: Activar el entorno
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# Paso 3: Instalar dependencias
pip install -r requirements.txt

# Paso 4: Iniciar el servidor
python run.py
```

La primera vez que se ejecute, el sistema descarga automáticamente el dataset MNIST (~11 MB). Esto tarda unos 30 segundos dependiendo de la conexión.

Cuando vea este mensaje, el servidor está listo:

```
==================================================
  ANN desde Cero — Proyecto 2 USAC
  http://localhost:5000
  Modo: desarrollo
==================================================
```

---

## 3. Acceder a la Aplicación

Abrir el navegador y escribir:

```
http://localhost:5000
```

Se mostrará la interfaz principal con cuatro secciones:

```
┌──────────────────┬──────────────────┬──────────────────┐
│  Visualización   │     Control de   │   Gráfica de     │
│  Red Neuronal    │  Entrenamiento   │     Pérdida      │
│   (Canvas)       │   (Dashboard)    │   (Chart.js)     │
├──────────────────┴──────────────────┴──────────────────┤
│                Sección de Cámara                        │
│     [Video]  [Thumbnail 28×28]  [Probabilidades]       │
└────────────────────────────────────────────────────────┘
```

---

## 4. Panel de Control de Entrenamiento

### 4.1 Configurar parámetros

Antes de entrenar, puede ajustar:

| Campo | Descripción | Valor por defecto | Rango sugerido |
|---|---|---|---|
| Épocas | Vueltas completas sobre el dataset | 20 | 5 – 50 |
| Learning Rate | Velocidad de aprendizaje | 0.01 | 0.001 – 0.1 |

### 4.2 Modo Ejecución Rápida

**Botón: "Entrenar"**

Entrena la red a máxima velocidad. Ideal para obtener un modelo funcional rápidamente.

1. Configure el número de épocas deseado
2. Haga clic en **"Entrenar"**
3. Observe cómo la gráfica de pérdida desciende en tiempo real
4. El panel muestra la época, pérdida (Loss) y precisión (Accuracy) actuales
5. Use **"Detener"** en cualquier momento para pausar el entrenamiento

**Resultados esperados:**
- Época 1: ~92% de precisión
- Época 2: ~96% de precisión
- Época 5+: ~97–98% de precisión

### 4.3 Modo Educativo (Paso a Paso)

**Botón: "Siguiente Paso"**

Ejecuta exactamente **una iteración** de entrenamiento y anima la red.

1. Haga clic en **"Siguiente Paso"** repetidamente
2. En cada paso verá:
   - Los nodos de la red oculta iluminarse según sus activaciones
   - Las barras de probabilidad Softmax actualizarse
   - El dashboard mostrar la pérdida de esa muestra específica
3. Las conexiones cambian de **color** (azul = peso positivo, rojo = peso negativo) y **grosor** (proporcional al valor absoluto del peso)

Este modo es ideal para entender cómo cada imagen individual afecta la red.

---

## 5. Visualización de la Red Neuronal

La sección izquierda muestra la topología de la red:

```
Entrada (784)    Oculta (64)    Salida (10)
     ●               ●              ●  ← dígito 0
     ●               ●              ●  ← dígito 1
     ●               ●              ●  ← dígito 2
    ...             ...             ●  ← ...
     ●               ●              ●  ← dígito 9
```

**Interpretación de colores:**

| Elemento | Color / Brillo | Significado |
|---|---|---|
| Nodo de entrada (azul) | Más brillante | Mayor valor del píxel |
| Nodo oculto (verde) | Más brillante | Mayor activación ReLU |
| Nodo de salida (rojo) | Más brillante | Mayor probabilidad Softmax |
| Conexión azul | Más gruesa | Peso positivo más alto |
| Conexión roja | Más gruesa | Peso negativo más negativo |

> Nota: Se muestran 20 nodos de los 784 de entrada para no saturar el canvas.

---

## 6. Gráfica de Pérdida

La sección derecha muestra dos curvas en tiempo real:

- **Línea azul (eje izquierdo):** Función de pérdida (Loss) — debe descender suavemente
- **Línea verde (eje derecho):** Precisión (Accuracy) — debe ascender suavemente

**Botón: "Exportar PNG"** — guarda la gráfica actual en `reports/loss_chart.png`

---

## 7. Uso de la Cámara para Predicción

### 7.1 Preparación

Antes de usar la cámara:
1. Escriba un dígito (0–9) grande y claro en una hoja de papel blanca
2. Use marcador negro o plumón — trazos gruesos funcionan mejor
3. El dígito debe ocupar al menos el 50% del papel
4. Iluminar bien el área (evitar sombras sobre el papel)

### 7.2 Activar la cámara

1. Haga clic en **"Iniciar Cámara"**
2. El navegador solicitará permiso de acceso a la cámara — haga clic en **"Permitir"**
3. El video en vivo aparecerá en la sección inferior

> Si el navegador bloquea la cámara, verifique que está usando `http://localhost:5000` (no HTTPS en localhost no es necesario).

### 7.3 Capturar y predecir

1. Sostenga el papel con el dígito frente a la cámara
2. Centre el dígito en el encuadre
3. Haga clic en **"Capturar y Predecir"**
4. La aplicación mostrará:
   - **Thumbnail 28×28:** la imagen tal como la ve la red neuronal (escala aumentada)
   - **Dígito predicho:** el número reconocido (0–9)
   - **Barras de probabilidad:** la confianza de la red para cada dígito posible

### 7.4 Consejos para mejores resultados

| Situación | Solución |
|---|---|
| La red siempre predice el mismo dígito | La red no ha sido entrenada aún — ejecutar al menos 2 épocas |
| El thumbnail sale completamente negro | El dígito no tiene suficiente contraste con el fondo |
| La predicción es incorrecta | Escribir el dígito más grande y con trazos más gruesos |
| Error "No se pudo acceder a la cámara" | Revisar permisos del navegador en Configuración |
| La cámara no detecta el dígito | Mejorar la iluminación del papel |

### 7.5 Cómo funciona internamente

Cuando hace clic en "Capturar y Predecir", ocurre lo siguiente:

```
1. camera.js captura un frame del video como imagen base64
2. Se envía al servidor → POST /predict
3. El servidor ejecuta el pipeline:
   a. Decodifica base64 → imagen BGR (OpenCV)
   b. Convierte a escala de grises
   c. Aplica suavizado Gaussiano (reduce ruido)
   d. Umbralización Otsu + inversión (dígito blanco, fondo negro)
   e. Detecta el contorno más grande (el dígito)
   f. Recorta la región de interés con margen del 15%
   g. Redimensiona a 28×28 píxeles
   h. Aplana a vector de 784 valores en [0.0, 1.0]
   i. La red neuronal predice: argmax(Softmax(W2·ReLU(W1·x+b1)+b2))
4. El servidor devuelve: dígito predicho, probabilidades, thumbnail PNG
5. La interfaz actualiza la visualización
```

---

## 8. Bitácora de Matrices de Pesos

El sistema genera automáticamente el archivo `reports/weight_matrix.log` durante el entrenamiento. Este archivo muestra cómo evolucionan los pesos de la capa oculta en las iteraciones 1, 50 y 100.

Para ver el contenido del archivo durante o después del entrenamiento, puede:

**Opción A — Desde la terminal:**
```bash
cat reports/weight_matrix.log
```

**Opción B — Desde la API:**
```
GET http://localhost:5000/reports/weights_log
```

**Contenido del archivo:**
```
BITÁCORA DE MATRICES DE PESOS
Generada: 2026-05-03 10:00:00
════════════════════════════════════════

╔══════════════════════════════════════╗
║  ITERACIÓN 1                         ║
╚══════════════════════════════════════╝
Sección W1[0:5, 350:355] (zona central activa):
  [+0.02508...  -0.00698...  ...]
Timestamp: 2026-05-03 10:00:01

╔══════════════════════════════════════╗
║  ITERACIÓN 50                        ║
╚══════════════════════════════════════╝
...
```

Se notará que los valores en la iteración 100 son distintos a los de la iteración 1, evidenciando que los pesos están actualizándose correctamente.

---

## 9. Solución de Problemas Comunes

| Problema | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Entorno virtual no activado | Ejecutar `source venv/bin/activate` |
| Puerto 5000 en uso | Otro proceso usa el puerto | Cambiar a `socketio.run(app, port=5001)` en `run.py` |
| MNIST no descarga | Sin conexión a internet | Descargar manualmente desde el README |
| `ImportError: libGL.so` | OpenCV sin dependencias | `sudo apt install libgl1-mesa-glx` |
| La gráfica no se actualiza | WebSocket bloqueado | Revisar firewall o usar Chrome |

---

## 10. Modo Debug (verificación de cálculos)

Para verificar que el código produce los mismos resultados que el cálculo manual:

```bash
source venv/bin/activate
python -c "
import sys; sys.path.insert(0,'.')
from neural_network.network import MLP
from neural_network.debug import debug_single_sample
from data.mnist_loader import load_mnist

X_train, y_train, _, _ = load_mnist()
net = MLP(lr=0.01, seed=42)
debug_single_sample(net, X_train[0], int(y_train[0]))
"
```

La salida mostrará todos los valores intermedios (Z1, A1, Z2, A2, dZ2, dW2, dZ1, dW1) con 4 decimales, listos para comparar con el Documento de Cálculo Manual.

---

## 11. Detener el Servidor

Presionar **Ctrl+C** en la terminal donde corre `python run.py`.
