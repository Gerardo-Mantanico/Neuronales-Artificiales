"""
Punto de entrada para desarrollo local.

Uso:
  python run.py            ← modo desarrollo (debug=True)
  python run.py --prod     ← modo producción (debug=False)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app import create_app

if __name__ == "__main__":
    debug = "--prod" not in sys.argv
    app, socketio = create_app()

    print("\n" + "=" * 50)
    print("  ANN desde Cero — Proyecto 2 USAC")
    print("  http://localhost:5000")
    print("  Modo:", "desarrollo" if debug else "producción")
    print("=" * 50 + "\n")

    socketio.run(app, host="0.0.0.0", port=5000, debug=debug)
