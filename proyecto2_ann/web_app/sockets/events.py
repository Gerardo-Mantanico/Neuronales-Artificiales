def register_events(socketio):
    """Registra los eventos WebSocket del servidor."""

    @socketio.on("connect")
    def on_connect():
        print("Cliente conectado")

    @socketio.on("disconnect")
    def on_disconnect():
        print("Cliente desconectado")

    @socketio.on("request_status")
    def on_request_status():
        """El cliente solicita el estado actual de la red."""
        from flask import current_app
        network = current_app.config["NETWORK"]
        socketio.emit("training_update", {
            "epoch": len(network.loss_history),
            "loss":  network.loss_history[-1] if network.loss_history else 0,
            "accuracy": network.accuracy_history[-1] if network.accuracy_history else 0,
        })
