import numpy as np
from .network import MLP
from .loss import one_hot, cross_entropy_loss
from .activations import relu_derivative


def debug_single_sample(
    network: MLP,
    X: np.ndarray,
    y: int,
) -> None:
    """
    Imprime cada variable intermedia de UNA iteración con 4 decimales.
    NO modifica los pesos de la red.
    Usar para verificar que el código coincide con el cálculo manual.
    """
    SEP = "═" * 60

    print(f"\n{SEP}")
    print("  MODO DEBUG — Una iteración (sin modificar pesos)")
    print(f"{SEP}\n")

    y_oh = one_hot(y)

    # ── Entrada ────────────────────────────────────────────────
    print("[ENTRADA]")
    print(f"  X[:10]     = {_fmt(X[:10])}")
    print(f"  y_real     = {y}")
    print(f"  y_onehot   = {y_oh.astype(int).tolist()}\n")

    # ── Forward capa oculta ────────────────────────────────────
    print("[FORWARD — CAPA OCULTA]")
    Z1 = network.W1 @ X + network.b1
    A1 = np.maximum(0.0, Z1)

    print(f"  W1[0,:5]   = {_fmt(network.W1[0, :5])}")
    print(f"  b1[0]      = {network.b1[0]:.4f}")
    print(f"  Z1[:5]     = {_fmt(Z1[:5])}")
    print(f"  A1[:5]     = {_fmt(A1[:5])}")
    print(f"  Nodos activos (Z1>0): {int(np.sum(Z1 > 0))} / 64\n")

    # ── Forward capa de salida ─────────────────────────────────
    print("[FORWARD — CAPA DE SALIDA]")
    Z2 = network.W2 @ A1 + network.b2
    Z2_estable = Z2 - np.max(Z2)
    exp_Z2 = np.exp(Z2_estable)
    A2 = exp_Z2 / np.sum(exp_Z2)

    print(f"  W2[0,:5]   = {_fmt(network.W2[0, :5])}")
    print(f"  b2[0]      = {network.b2[0]:.4f}")
    print(f"  Z2         = {_fmt(Z2)}")
    print(f"  A2         = {_fmt(A2)}")
    print(f"  sum(A2)    = {np.sum(A2):.10f}  ← debe ser 1.0\n")

    # ── Pérdida ────────────────────────────────────────────────
    print("[PÉRDIDA]")
    loss = cross_entropy_loss(A2, y_oh)
    print(f"  Loss       = {loss:.4f}")
    print(f"  Predicción = {int(np.argmax(A2))}  (correcto: {y})\n")

    # ── Backward capa de salida ────────────────────────────────
    print("[BACKWARD — CAPA DE SALIDA]")
    dZ2 = A2 - y_oh
    dW2 = np.outer(dZ2, A1)
    db2 = dZ2

    print(f"  dZ2        = {_fmt(dZ2)}")
    print(f"  dW2[0,:3]  = {_fmt(dW2[0, :3])}")
    print(f"  db2[0]     = {db2[0]:.4f}\n")

    # ── Backward capa oculta ───────────────────────────────────
    print("[BACKWARD — CAPA OCULTA]")
    dA1 = network.W2.T @ dZ2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.outer(dZ1, X)
    db1 = dZ1

    print(f"  dA1[:5]    = {_fmt(dA1[:5])}")
    print(f"  dZ1[:5]    = {_fmt(dZ1[:5])}")
    print(f"  dW1[0,:3]  = {_fmt(dW1[0, :3])}")
    print(f"  db1[0]     = {db1[0]:.4f}\n")

    # ── Verificación de actualización ──────────────────────────
    print(f"[ACTUALIZACIÓN (lr={network.lr})]")
    print(f"  W2[0,0]     antes  = {network.W2[0, 0]:.4f}")
    print(f"  dW2[0,0]           = {dW2[0, 0]:.4f}")
    W2_nuevo_00 = network.W2[0, 0] - network.lr * dW2[0, 0]
    print(f"  W2[0,0]     después = {W2_nuevo_00:.4f}")
    print(f"\n  W1[0,0]     antes  = {network.W1[0, 0]:.4f}")
    print(f"  dW1[0,0]           = {dW1[0, 0]:.4f}")
    W1_nuevo_00 = network.W1[0, 0] - network.lr * dW1[0, 0]
    print(f"  W1[0,0]     después = {W1_nuevo_00:.4f}")

    print(f"\n{SEP}")
    print("  Comparar estos valores con el cálculo manual.")
    print(f"{SEP}\n")


def _fmt(arr: np.ndarray) -> str:
    """Formatea un array 1D con 4 decimales."""
    return "[" + "  ".join(f"{v:.4f}" for v in arr) + "]"
