import numpy as np
import cv2


def detect_digit_region(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Detecta y recorta la región donde está escrito el dígito.

    Pipeline:
      1. Convertir a escala de grises
      2. Blur gaussiano para reducir ruido
      3. Umbralización Otsu (adaptativa)
      4. Encontrar contornos
      5. Seleccionar el bounding box más grande
      6. Retornar recorte cuadrado con margen

    Parámetros:
      imagen_bgr: frame BGR capturado de la cámara

    Retorna:
      recorte en escala de grises (H, W) — puede variar en tamaño
      Si no detecta ningún dígito, retorna la imagen gris completa.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # Otsu determina el umbral automáticamente
    _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return gris

    # Tomar el contorno de mayor área
    contorno_mayor = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_mayor)

    # Margen proporcional para no cortar el dígito
    margen = int(max(w, h) * 0.15)
    x1 = max(0, x - margen)
    y1 = max(0, y - margen)
    x2 = min(gris.shape[1], x + w + margen)
    y2 = min(gris.shape[0], y + h + margen)

    recorte = binaria[y1:y2, x1:x2]
    return recorte
