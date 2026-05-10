from flask import Blueprint, jsonify, request, current_app
from image_processing.preprocessor import preprocess_image
import base64
import numpy as np

bp = Blueprint("prediction", __name__, url_prefix="/predict")


@bp.route("", methods=["POST"])
def predict():
    # Recibe la imagen de la camara en base64, predice y regresa el resultado
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Falta campo 'image'"}), 400

    network = current_app.config["NETWORK"]

    try:
        vector, thumbnail = preprocess_image(data["image"])
    except Exception as e:
        return jsonify({"error": f"Error en procesamiento: {str(e)}"}), 500

    probabilities = network.predict_proba(vector)
    digit = int(np.argmax(probabilities))

    # Codificar la miniatura para pintarla en el HTML
    import cv2
    _, buffer = cv2.imencode(".png", thumbnail)
    thumb_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "digit": digit,
        "probabilities": [round(float(p), 6) for p in probabilities],
        "thumbnail_b64": f"data:image/png;base64,{thumb_b64}",
    })
