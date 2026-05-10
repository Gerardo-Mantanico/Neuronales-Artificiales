import os
import numpy as np
from datetime import datetime

ITERACIONES_OBJETIVO = {1, 50, 100}


class WeightLogger:
    # Bitacora para guardar los pesos W1 en las iteraciones 1, 50 y 100

    def __init__(self, filepath="reports/weight_matrix.log"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Limpiar archivo al iniciar el entrenamiento
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("BITACORA DE MATRICES DE PESOS\n")
            f.write(f"Generada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

    def log_weights(self, iteration, W1):
        if iteration not in ITERACIONES_OBJETIVO:
            return

        # Columnas 350-355: zona central de la imagen 28x28 (pixeles activos en MNIST)
        seccion = W1[0:5, 350:355]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        bloque = (
            f"--------------------------------------------------\n"
            f"  ITERACION {iteration}\n"
            f"--------------------------------------------------\n"
            f"Seccion W1[0:5, 350:355] (zona activa central):\n"
        )

        for fila in seccion:
            bloque += "  [" + "  ".join(f"{v:+.8f}" for v in fila) + "]\n"

        bloque += f"Timestamp: {timestamp}\n\n"

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(bloque)
