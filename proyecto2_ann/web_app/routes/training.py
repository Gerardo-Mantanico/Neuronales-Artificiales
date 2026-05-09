import threading
from flask import Blueprint, jsonify, request, current_app

bp = Blueprint("training", __name__, url_prefix="/train")

_training_thread: threading.Thread = None
_stop_flag = threading.Event()


@bp.route("/start", methods=["POST"])
def start_training():
    """Inicia el entrenamiento en un hilo separado."""
    global _training_thread, _stop_flag

    if _training_thread and _training_thread.is_alive():
        return jsonify({"error": "Entrenamiento ya en curso"}), 409

    data = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 20))
    lr = float(data.get("lr", 0.01))

    _stop_flag.clear()

    app = current_app._get_current_object()

    def run():
        from trainer import train
        network = app.config["NETWORK"]
        X_train = app.config["X_TRAIN"]
        y_train = app.config["Y_TRAIN"]
        logger  = app.config["LOGGER"]
        socketio = app.config["SOCKETIO"]

        network.lr = lr
        network.loss_history.clear()
        network.accuracy_history.clear()

        def emit_progress(epoch, loss, acc):
            if _stop_flag.is_set():
                return
            from web_app.sockets.events import any_connected
            if not any_connected():
                _stop_flag.set()
                return
            socketio.emit("training_update", {
                "epoch": epoch,
                "loss": round(loss, 6),
                "accuracy": round(acc, 6),
                "weights_sample": {
                    "W2": network.W2[0:5, 0:5].tolist(),
                },
            })

        train(network, X_train, y_train,
              epochs=epochs, logger=logger, callback=emit_progress)

        socketio.emit("training_done", {
            "epochs": epochs,
            "final_loss": network.loss_history[-1],
            "final_accuracy": network.accuracy_history[-1],
        })

    _training_thread = threading.Thread(target=run, daemon=True)
    _training_thread.start()

    return jsonify({"status": "started", "epochs": epochs, "lr": lr})


@bp.route("/stop", methods=["POST"])
def stop_training():
    """Detiene el entrenamiento al finalizar la iteración actual."""
    _stop_flag.set()
    return jsonify({"status": "stopping"})


@bp.route("/status", methods=["GET"])
def status():
    """Retorna el estado actual de la red."""
    network = current_app.config["NETWORK"]
    running = _training_thread is not None and _training_thread.is_alive()
    return jsonify({
        "running": running,
        "iteration": network.iteration,
        "loss_history": network.loss_history,
        "accuracy_history": network.accuracy_history,
    })


@bp.route("/step", methods=["POST"])
def step():
    """Ejecuta un solo paso de entrenamiento (modo educativo)."""
    from trainer import train_single_step

    app = current_app._get_current_object()
    network = app.config["NETWORK"]
    logger  = app.config["LOGGER"]
    X_train = app.config["X_TRAIN"]
    y_train = app.config["Y_TRAIN"]

    # Selecciona la siguiente muestra en secuencia
    idx = network.iteration % len(X_train)
    X_i = X_train[idx]
    y_i = int(y_train[idx])

    resultado = train_single_step(network, X_i, y_i, logger)
    return jsonify(resultado)
