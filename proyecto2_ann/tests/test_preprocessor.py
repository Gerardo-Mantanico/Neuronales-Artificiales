"""
Tests para el pipeline de procesamiento de imagen.
Ejecutar con: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

try:
    import cv2
    CV2_DISPONIBLE = True
except ImportError:
    CV2_DISPONIBLE = False


@pytest.mark.skipif(not CV2_DISPONIBLE, reason="OpenCV no instalado")
class TestPreprocessor:

    def _imagen_dummy(self):
        """Crea una imagen BGR de prueba con un rectángulo negro (simula dígito)."""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (70, 60), (130, 140), (0, 0, 0), -1)
        return img

    def test_roi_detector_retorna_array(self):
        from image_processing.roi_detector import detect_digit_region
        img = self._imagen_dummy()
        roi = detect_digit_region(img)
        assert isinstance(roi, np.ndarray)

    def test_roi_detector_no_vacio(self):
        from image_processing.roi_detector import detect_digit_region
        img = self._imagen_dummy()
        roi = detect_digit_region(img)
        assert roi.size > 0

    def test_vector_784(self):
        """El preprocesador debe retornar exactamente 784 valores."""
        import base64
        from image_processing.preprocessor import preprocess_image

        img = self._imagen_dummy()
        _, buffer = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buffer).decode("utf-8")
        b64_with_header = f"data:image/jpeg;base64,{b64}"

        vector, thumbnail = preprocess_image(b64_with_header)

        assert vector.shape == (784,)

    def test_vector_normalizado(self):
        """Todos los valores deben estar en [0.0, 1.0]."""
        import base64
        from image_processing.preprocessor import preprocess_image

        img = self._imagen_dummy()
        _, buffer = cv2.imencode(".jpg", img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        vector, _ = preprocess_image(b64)

        assert np.all(vector >= 0.0)
        assert np.all(vector <= 1.0)

    def test_thumbnail_28x28(self):
        """El thumbnail debe ser 28×28 píxeles."""
        import base64
        from image_processing.preprocessor import preprocess_image

        img = self._imagen_dummy()
        _, buffer = cv2.imencode(".jpg", img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        _, thumbnail = preprocess_image(b64)

        assert thumbnail.shape == (28, 28)
