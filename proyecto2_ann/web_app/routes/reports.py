import json
from flask import Blueprint, jsonify, current_app, send_file

bp = Blueprint("reports", __name__, url_prefix="/reports")


@bp.route("/loss_chart", methods=["GET"])
def loss_chart():
    """Retorna el historial de pérdida y accuracy para graficar."""
    network = current_app.config["NETWORK"]
    return jsonify({
        "epochs": list(range(1, len(network.loss_history) + 1)),
        "loss":   network.loss_history,
        "accuracy": network.accuracy_history,
    })


@bp.route("/weights_log", methods=["GET"])
def weights_log():
    """Retorna el contenido del archivo de bitácora de pesos."""
    try:
        with open("reports/weight_matrix.log", "r", encoding="utf-8") as f:
            contenido = f.read()
        return jsonify({"log": contenido})
    except FileNotFoundError:
        return jsonify({"error": "Bitácora no encontrada. Entrena primero."}), 404
