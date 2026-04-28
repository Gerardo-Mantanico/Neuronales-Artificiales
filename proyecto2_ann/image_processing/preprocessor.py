import numpy as np
import cv2
from .capture import decode_frame
from .roi_detector import detect_digit_region


def preprocess_image(base64_string: str) -> tuple:
    """
    Pipeline completo: base64 → vector de 784 valores normalizados.

    Pasos:
      1. Decodificar frame base64 → imagen BGR
      2. Detectar región del dígito (ROI)
      3. Redimensionar a 28×28
      4. Aplanar (28,28) → (784,)
      5. Normalizar → [0.0, 1.0]

    Retorna:
      vector:    np.ndarray (784,) float64 — entrada para la red
      thumbnail: np.ndarray (28,28) uint8  — para mostrar en la UI
    """
    imagen_bgr = decode_frame(base64_string)

    roi = detect_digit_region(imagen_bgr)

    # Asegurar que sea cuadrado antes de redimensionar
    h, w = roi.shape[:2]
    lado = max(h, w)
    cuadrado = np.zeros((lado, lado), dtype=roi.dtype)
    y_off = (lado - h) // 2
    x_off = (lado - w) // 2
    cuadrado[y_off:y_off + h, x_off:x_off + w] = roi

    thumbnail = cv2.resize(cuadrado, (28, 28), interpolation=cv2.INTER_AREA)

    vector = thumbnail.flatten().astype(np.float64) / 255.0

    return vector, thumbnail
