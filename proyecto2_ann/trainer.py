import numpy as np
from neural_network.network import MLP
from neural_network.loss import one_hot, cross_entropy_loss, accuracy
from neural_network.logger import WeightLogger


def train(
    network,
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    logger=None,
    callback=None,
    stop_event=None,
):
    # Metodo para entrenar la red usando mini-batches de forma secuencial
    n_muestras = len(X_train)
    network.iteration = 0

    for epoch in range(1, epochs + 1):
        if stop_event is not None and stop_event.is_set():
            print("Entrenamiento cancelado por el usuario.")
            break

        # Barajar los datos antes de cada epoca
        indices = np.random.permutation(n_muestras)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        loss_epoch = 0.0
        correctas_epoch = 0

        for inicio in range(0, n_muestras, batch_size):
            if stop_event is not None and stop_event.is_set():
                break

            fin = min(inicio + batch_size, n_muestras)
            X_batch = X_shuffled[inicio:fin]
            y_batch = y_shuffled[inicio:fin]

            for X_i, y_i in zip(X_batch, y_batch):
                network.iteration += 1

                # Forward pass
                Z1, A1, Z2, A2 = network.forward(X_i)

                # Calcular perdida y aciertos de la epoca
                y_oh = one_hot(int(y_i))
                loss_epoch += cross_entropy_loss(A2, y_oh)
                correctas_epoch += accuracy(A2, int(y_i))

                # Backward pass
                dW1, db1, dW2, db2 = network.backward(X_i, y_oh, Z1, A1, A2)

                # Modificar pesos
                network.update(dW1, db1, dW2, db2)

                # Registrar pesos en la bitacora si toca
                if logger is not None:
                    logger.log_weights(network.iteration, network.W1)

            # Ceder control a gevent cada 50 batches para que no se congele el SocketIO
            batch_idx = inicio // batch_size
            if batch_idx % 50 == 0:
                import gevent
                gevent.sleep(0)

        if stop_event is not None and stop_event.is_set():
            break

        # Sacar promedios de la epoca
        loss_avg = loss_epoch / n_muestras
        acc_avg = correctas_epoch / n_muestras

        network.loss_history.append(loss_avg)
        network.accuracy_history.append(acc_avg)

        print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss_avg:.4f} | Acc: {acc_avg:.4f}")

        if callback is not None:
            callback(epoch, loss_avg, acc_avg)

    return {
        "loss_history": network.loss_history,
        "accuracy_history": network.accuracy_history,
    }


def train_single_step(network, X_i, y_i, logger=None):
    # Corre una sola iteracion de entrenamiento (para el modo paso a paso de la UI)
    network.iteration += 1

    Z1, A1, Z2, A2 = network.forward(X_i)
    y_oh = one_hot(y_i)
    loss = cross_entropy_loss(A2, y_oh)
    acc = accuracy(A2, y_i)

    dW1, db1, dW2, db2 = network.backward(X_i, y_oh, Z1, A1, A2)
    network.update(dW1, db1, dW2, db2)

    if logger is not None:
        logger.log_weights(network.iteration, network.W1)

    return {
        "iteration": network.iteration,
        "loss": round(loss, 6),
        "accuracy": acc,
        "prediction": int(np.argmax(A2)),
        "probabilities": A2.tolist(),
        "weights_sample": {
            "W1_sample": network.W1[0:5, 0:5].tolist(),
            "W2_sample": network.W2[0:5, 0:5].tolist(),
        },
        "activations": {
            "A1": A1.tolist(),
            "A2": A2.tolist(),
        },
    }
