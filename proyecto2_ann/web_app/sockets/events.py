def register_events(socketio):
    """Registra los eventos WebSocket del servidor."""

    @socketio.on("connect")
    def on_connect():
        print("Cliente conectado")

    @socketio.on("disconnect")
    def on_disconnect():
        print("Cliente desconectado")
