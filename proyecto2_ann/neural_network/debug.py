import numpy as np
from .loss import one_hot, cross_entropy_loss

def format_list(lst):
    # Imprime los floats con 4 decimales separados por espacios
    return "[" + "  ".join(f"{val:.4f}" for val in lst) + "]"

def debug_single_sample(network, X, y):
    print("============================================================")
    print("  MODO DEBUG — Una iteracion (sin modificar pesos)")
    print("============================================================")
    
    print("\n[ENTRADA]")
    print(f"  X[:10]     = {format_list(X[:10])}")
    print(f"  y_real     = {y}")
    y_oh = one_hot(y)
    print(f"  y_onehot   = {y_oh.tolist()}")
    
    # Forward Capa Oculta
    Z1 = network.W1 @ X + network.b1
    from .activations import relu
    A1 = relu(Z1)
    print("\n[FORWARD — CAPA OCULTA]")
    print(f"  W1[0,:5]   = {format_list(network.W1[0,:5])}")
    print(f"  b1[0]      = {network.b1[0]:.4f}")
    print(f"  Z1[:5]     = {format_list(Z1[:5])}")
    print(f"  A1[:5]     = {format_list(A1[:5])}")
    print(f"  Nodos activos (Z1>0): {np.sum(Z1 > 0)} / 64")
    
    # Forward Capa Salida
    Z2 = network.W2 @ A1 + network.b2
    from .activations import softmax
    A2 = softmax(Z2)
    print("\n[FORWARD — CAPA DE SALIDA]")
    print(f"  W2[0,:5]   = {format_list(network.W2[0,:5])}")
    print(f"  b2[0]      = {network.b2[0]:.4f}")
    print(f"  Z2         = {format_list(Z2)}")
    print(f"  A2         = {format_list(A2)}")
    print(f"  sum(A2)    = {np.sum(A2):.10f}  <- debe ser 1.0")
    
    # Pérdida y predicción
    loss = cross_entropy_loss(A2, y_oh)
    pred = np.argmax(A2)
    print("\n[PÉRDIDA]")
    print(f"  Loss       = {loss:.4f}")
    print(f"  Predicción = {pred}  (correcto: {y})")
    
    # Backward
    dW1, db1, dW2, db2 = network.backward(X, y_oh, Z1, A1, A2)
    print("\n[BACKWARD — CAPA DE SALIDA]")
    dZ2 = A2 - y_oh
    print(f"  dZ2        = {format_list(dZ2)}")
    print(f"  dW2[0,:3]  = {format_list(dW2[0,:3])}")
    print(f"  db2[0]     = {db2[0]:.4f}")
    
    # Backward Capa Oculta
    dA1 = network.W2.T @ dZ2
    from .activations import relu_derivative
    dZ1 = dA1 * relu_derivative(Z1)
    print("\n[BACKWARD — CAPA OCULTA]")
    print(f"  dA1[:5]    = {format_list(dA1[:5])}")
    print(f"  dZ1[:5]    = {format_list(dZ1[:5])}")
    print(f"  dW1[0,:3]  = {format_list(dW1[0,:3])}")
    print(f"  db1[0]     = {db1[0]:.4f}")
    
    # Actualización (simulada con lr = 0.01)
    lr = 0.01
    W2_nuevo = network.W2[0,0] - lr * dW2[0,0]
    W1_nuevo = network.W1[0,0] - lr * dW1[0,0]
    print("\n[ACTUALIZACIÓN (lr=0.01)]")
    print(f"  W2[0,0]     antes  = {network.W2[0,0]:.4f}")
    print(f"  dW2[0,0]           = {dW2[0,0]:.4f}")
    print(f"  W2[0,0]     despues = {W2_nuevo:.4f}")
    print("")
    print(f"  W1[0,0]     antes  = {network.W1[0,0]:.4f}")
    print(f"  dW1[0,0]           = {dW1[0,0]:.4f}")
    print(f"  W1[0,0]     despues = {W1_nuevo:.4f}")
