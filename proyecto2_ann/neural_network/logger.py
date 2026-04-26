import os
import numpy as np
from datetime import datetime

ITERACIONES_OBJETIVO = {1, 50, 100}


class WeightLogger:
    """
    Guarda secciones de la matriz W1 en las iteraciones 1, 50 y 100.
    El archivo se crea limpio al instanciar y se hace append en cada registro.
    """

    def __init__(self, filepath: str = "reports/weight_matrix.log") -> None:
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Limpiar archivo al inicio de cada entrenamiento
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(f"BITÁCORA DE MATRICES DE PESOS\n")
            f.write(f"Generada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

    def log_weights(self, iteration: int, W1: np.ndarray) -> None:
        """
        Registra W1[0:5, 0:5] si la iteración está en {1, 50, 100}.
        No hace nada en iteraciones que no corresponden.
        """
        if iteration not in ITERACIONES_OBJETIVO:
            return

        # Columnas 350-355: zona central de la imagen 28x28 (píxeles activos en MNIST)
        seccion = W1[0:5, 350:355]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        bloque = (
            f"╔══════════════════════════════════════════════════════╗\n"
            f"║  ITERACIÓN {iteration:<44}║\n"
            f"╚══════════════════════════════════════════════════════╝\n"
            f"Sección W1[0:5, 350:355]  (zona central activa):\n"
        )

        # Formatear cada fila con 8 decimales
        for fila in seccion:
            bloque += "  [" + "  ".join(f"{v:+.8f}" for v in fila) + "]\n"

        bloque += f"Timestamp: {timestamp}\n\n"

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(bloque)
