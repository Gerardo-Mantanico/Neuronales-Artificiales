import numpy as np
import cv2


def detect_digit_region(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Detecta y recorta la región donde está escrito el dígito.

    Pipeline:
      1. Convertir a escala de grises
      2. Blur gaussiano para reducir ruido
      3. Umbralización Otsu (adaptativa)
      4. Limpieza morfológica de ruido (apertura/cierre)
      5. Encontrar contornos anidados (RETR_LIST)
      6. Filtrar por tamaño (evitar pared gigante y ruido diminuto)
      7. Retornar recorte cuadrado con margen

    Parámetros:
      imagen_bgr: frame BGR capturado de la cámara

    Retorna:
      recorte en escala de grises/binario (H, W)
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # Otsu determina el umbral automáticamente
    _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Limpieza morfológica para reducir sensibilidad a mala iluminación (ruido y cortes de trazo)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    # Encontrar todos los contornos (incluyendo anidados dentro de la hoja blanca)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return binaria

    h_img, w_img = gris.shape
    total_area = h_img * w_img

    # Filtrar contornos para descartar el fondo/pared/hoja gigante y el ruido mínimo
    min_area = 120
    max_area = total_area * 0.25

    validos = [c for c in contornos if min_area < cv2.contourArea(c) < max_area]

    if validos:
        contorno_mayor = max(validos, key=cv2.contourArea)
    else:
        # Fallback: si no hay en el rango ideal, ignorar lo que ocupe casi toda la pantalla
        validos_fallback = [c for c in contornos if cv2.contourArea(c) < total_area * 0.85]
        if validos_fallback:
            contorno_mayor = max(validos_fallback, key=cv2.contourArea)
        else:
            contorno_mayor = max(contornos, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contorno_mayor)

    # Margen proporcional más amplio (25%) para dar suficiente padding negro alrededor del número (como en MNIST)
    margen = int(max(w, h) * 0.25)
    x1 = max(0, x - margen)
    y1 = max(0, y - margen)
    x2 = min(gris.shape[1], x + w + margen)
    y2 = min(gris.shape[0], y + h + margen)

    recorte = binaria[y1:y2, x1:x2]

    # Dilatar el recorte proporcionalmente para engrosar las líneas
    # Esto asegura que al reducir a 28x28, el trazo mantenga un grosor similar a MNIST (2-3 píxeles)
    alto_rec = y2 - y1
    ancho_rec = x2 - x1
    dim = max(alto_rec, ancho_rec)
    k_size = max(3, int(dim * 0.04))
    if k_size % 2 == 0:
        k_size += 1
    
    kernel_dilatar = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    recorte = cv2.dilate(recorte, kernel_dilatar, iterations=1)

    return recorte
