import base64
import numpy as np
import cv2


def decode_frame(base64_string: str) -> np.ndarray:
    """
    Decodifica un frame enviado desde el navegador como base64.

    El frontend captura el frame del <video> en un <canvas> y lo convierte
    con canvas.toDataURL('image/jpeg') antes de enviarlo.

    Parámetros:
      base64_string: puede incluir prefijo 'data:image/jpeg;base64,' o no

    Retorna:
      imagen BGR de OpenCV, forma (H, W, 3)
    """
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    datos = base64.b64decode(base64_string)
    buffer = np.frombuffer(datos, dtype=np.uint8)
    imagen = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return imagen
