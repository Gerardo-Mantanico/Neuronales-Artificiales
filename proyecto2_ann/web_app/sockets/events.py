import threading

_connected_sids = set()
_lock = threading.Lock()


def register_events(socketio):
    # Eventos de websocket para la conexion con la UI

    @socketio.on("connect")
    def on_connect():
        from flask import request
        with _lock:
            _connected_sids.add(request.sid)

    @socketio.on("disconnect")
    def on_disconnect():
        from flask import request
        with _lock:
            _connected_sids.discard(request.sid)
        # Si no queda nadie conectado, paramos el hilo de entrenamiento
        if not _connected_sids:
            from web_app.routes.training import _stop_flag
            _stop_flag.set()

    @socketio.on("request_status")
    def on_request_status():
        # El cliente pide las metricas actuales
        from flask import current_app
        network = current_app.config["NETWORK"]
        socketio.emit("training_update", {
            "epoch": len(network.loss_history),
            "loss":  network.loss_history[-1] if network.loss_history else 0,
            "accuracy": network.accuracy_history[-1] if network.accuracy_history else 0,
        })


def any_connected() -> bool:
    with _lock:
        return bool(_connected_sids)
