import os
import sys

# Permite importar módulos desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template
from flask_socketio import SocketIO

from neural_network.network import MLP
from neural_network.logger import WeightLogger
from data.mnist_loader import load_mnist
from web_app.routes.training import bp as training_bp
from web_app.routes.prediction import bp as prediction_bp
from web_app.routes.reports import bp as reports_bp
from web_app.sockets.events import register_events


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="web_app/templates",
        static_folder="web_app/static",
    )
    app.config["SECRET_KEY"] = "ann-proyecto2-usac"

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

    # Cargar MNIST una sola vez al inicio
    print("Cargando dataset MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Instanciar red y logger
    network = MLP(lr=0.01, seed=42)
    logger  = WeightLogger(filepath="reports/weight_matrix.log")

    # Guardar en config para acceso desde las rutas
    app.config["NETWORK"]  = network
    app.config["LOGGER"]   = logger
    app.config["SOCKETIO"] = socketio
    app.config["X_TRAIN"]  = X_train
    app.config["Y_TRAIN"]  = y_train
    app.config["X_TEST"]   = X_test
    app.config["Y_TEST"]   = y_test

    # Registrar blueprints
    app.register_blueprint(training_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(reports_bp)

    # Registrar eventos WebSocket
    register_events(socketio)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app, socketio


if __name__ == "__main__":
    app, socketio = create_app()
    print("\n Servidor iniciado en http://localhost:5000\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
