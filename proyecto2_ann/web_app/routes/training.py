import threading
import os
from flask import Blueprint, jsonify, request, current_app

bp = Blueprint("training", __name__, url_prefix="/train")

_training_thread = None
_stop_flag = threading.Event()


@bp.route("/start", methods=["POST"])
def start_training():
    # Inicia el hilo de entrenamiento
    global _training_thread, _stop_flag

    if _training_thread and _training_thread.is_alive():
        return jsonify({"error": "Entrenamiento ya en curso"}), 409

    data = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 20))
    lr = float(data.get("lr", 0.01))

    _stop_flag.clear()

    app = current_app._get_current_object()

    def run():
        global _training_thread
        try:
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

                # Muestra aleatoria para la animacion de la red
                import random
                idx_random = random.randint(0, len(X_train) - 1)
                X_sample = X_train[idx_random]
                Z1, A1, Z2, A2 = network.forward(X_sample)

                # Emitir en segundo plano para no trabar el hilo principal
                socketio.start_background_task(socketio.emit, "training_update", {
                    "epoch": epoch,
                    "loss": round(loss, 6),
                    "accuracy": round(acc, 6),
                    "weights_sample": {
                        "W1": network.W1[0:10, 0:20].tolist(),
                        "W2": network.W2[0:10, 0:10].tolist(),
                    },
                    "activations": {
                        "X": X_sample[0:20].tolist(),
                        "A1": A1.tolist(),
                        "A2": A2.tolist(),
                    }
                })

            train(network, X_train, y_train,
                  epochs=epochs, logger=logger, callback=emit_progress,
                  stop_event=_stop_flag)

            # Notificar que termino el entrenamiento
            socketio.start_background_task(socketio.emit, "training_done", {
                "epochs": epochs,
                "final_loss": network.loss_history[-1] if network.loss_history else 0.0,
                "final_accuracy": network.accuracy_history[-1] if network.accuracy_history else 0.0,
            })

            # Guardar los pesos nuevos
            try:
                models_dir = "models"
                os.makedirs(models_dir, exist_ok=True)
                weights_path = os.path.join(models_dir, "mlp_weights.npz")
                network.save(weights_path)
                socketio.start_background_task(socketio.emit, "saved_weights", {"path": weights_path})
            except Exception as e:
                socketio.start_background_task(socketio.emit, "saved_weights", {"error": str(e)})
        finally:
            _training_thread = None

    _training_thread = threading.Thread(target=run, daemon=True)
    _training_thread.start()

    return jsonify({"status": "started", "epochs": epochs, "lr": lr})


@bp.route("/stop", methods=["POST"])
def stop_training():
    # Detiene el entrenamiento levantando la bandera
    _stop_flag.set()
    return jsonify({"status": "stopping"})


@bp.route("/status", methods=["GET"])
def status():
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
    # Corre una sola iteracion (modo educativo paso a paso)
    from trainer import train_single_step

    app = current_app._get_current_object()
    network = app.config["NETWORK"]
    logger  = app.config["LOGGER"]
    X_train = app.config["X_TRAIN"]
    y_train = app.config["Y_TRAIN"]

    # Siguiente muestra secuencial
    idx = network.iteration % len(X_train)
    X_i = X_train[idx]
    y_i = int(y_train[idx])

    resultado = train_single_step(network, X_i, y_i, logger)
    return jsonify(resultado)
