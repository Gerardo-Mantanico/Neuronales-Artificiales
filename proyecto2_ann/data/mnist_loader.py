import os
import gzip
import struct
import urllib.request
import numpy as np

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
ARCHIVOS = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def _descargar(nombre_archivo: str) -> None:
    ruta = os.path.join(RAW_DIR, nombre_archivo)
    if os.path.exists(ruta):
        return
    url = BASE_URL + nombre_archivo
    print(f"Descargando {nombre_archivo} ...")
    urllib.request.urlretrieve(url, ruta)
    print(f"  ✓ Guardado en {ruta}")


def _leer_imagenes(nombre_archivo: str) -> np.ndarray:
    ruta = os.path.join(RAW_DIR, nombre_archivo)
    with gzip.open(ruta, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)   # (N, 784)


def _leer_etiquetas(nombre_archivo: str) -> np.ndarray:
    ruta = os.path.join(RAW_DIR, nombre_archivo)
    with gzip.open(ruta, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data   # (N,)


def load_mnist() -> tuple:
    """
    Descarga (si no existe) y carga MNIST desde archivos .gz.

    Retorna:
      X_train: (60000, 784) float64 normalizado [0.0, 1.0]
      y_train: (60000,)     int64   etiquetas 0–9
      X_test:  (10000, 784) float64 normalizado
      y_test:  (10000,)     int64
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    for nombre in ARCHIVOS.values():
        _descargar(nombre)

    X_train = _leer_imagenes(ARCHIVOS["train_images"]).astype(np.float64) / 255.0
    y_train = _leer_etiquetas(ARCHIVOS["train_labels"]).astype(np.int64)
    X_test  = _leer_imagenes(ARCHIVOS["test_images"]).astype(np.float64) / 255.0
    y_test  = _leer_etiquetas(ARCHIVOS["test_labels"]).astype(np.int64)

    print(f"MNIST cargado: train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test
