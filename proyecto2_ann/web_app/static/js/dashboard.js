/**
 * dashboard.js — orquestación principal de la UI
 */

document.addEventListener("DOMContentLoaded", () => {
  NetViz.init("canvas-network");
  Charts.init("canvas-loss");

  let totalEpochs = 20;

  const socket = io();

  // ── WebSocket: actualización por época ────────────────────
  socket.on("training_update", (data) => {
    setStatus("training");

    document.getElementById("stat-epoch").textContent = data.epoch;
    document.getElementById("stat-loss").textContent  = data.loss.toFixed(4);
    document.getElementById("stat-acc").textContent   = (data.accuracy * 100).toFixed(2) + "%";

    setProgress(data.epoch, totalEpochs);
    Charts.update(data.epoch, data.loss, data.accuracy);

    if (data.weights_sample) {
      NetViz.updateWeights({
        W1_sample: data.weights_sample.W1,
        W2_sample: data.weights_sample.W2,
        input_activations: data.activations ? data.activations.X : null,
        hidden_activations: data.activations ? data.activations.A1 : null,
        output_activations: data.activations ? data.activations.A2 : null,
      });
    }
  });

  socket.on("training_done", (data) => {
    setStatus("ready");
    hideProgress();
    document.getElementById("btn-train").disabled = false;
    document.getElementById("btn-stop").disabled  = true;
    showToast(`Entrenamiento completado · Loss: ${data.final_loss.toFixed(4)}`, "success");
  });

  // ── Botón Entrenar ────────────────────────────────────────
  document.getElementById("btn-train").addEventListener("click", async () => {
    totalEpochs = parseInt(document.getElementById("input-epochs").value) || 20;
    const lr    = parseFloat(document.getElementById("input-lr").value)   || 0.01;

    Charts.reset();
    document.getElementById("btn-train").disabled = true;
    document.getElementById("btn-stop").disabled  = false;
    showProgress(totalEpochs);
    setStatus("training");

    const resp = await fetch("/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: totalEpochs, lr }),
    });

    if (!resp.ok) {
      showToast("Error al iniciar entrenamiento", "error");
      document.getElementById("btn-train").disabled = false;
      document.getElementById("btn-stop").disabled  = true;
      setStatus("ready");
    }
  });

  // ── Botón Detener ─────────────────────────────────────────
  document.getElementById("btn-stop").addEventListener("click", async () => {
    await fetch("/train/stop", { method: "POST" });
    document.getElementById("btn-stop").disabled  = true;
    document.getElementById("btn-train").disabled = false;
    hideProgress();
    setStatus("stopped");
    showToast("Entrenamiento detenido", "error");
  });

  // ── Botón Paso a Paso ─────────────────────────────────────
  document.getElementById("btn-step").addEventListener("click", async () => {
    const resp = await fetch("/train/step", { method: "POST" });
    if (!resp.ok) { showToast("Error en paso a paso", "error"); return; }

    const data = await resp.json();

    document.getElementById("stat-epoch").textContent = data.iteration;
    document.getElementById("stat-loss").textContent  = data.loss.toFixed(4);
    document.getElementById("stat-acc").textContent   = (data.accuracy * 100).toFixed(0) + "%";
    document.getElementById("stat-pred").textContent  = data.prediction;

    NetViz.updateWeights({
      W2_sample: data.weights_sample.W2_sample,
      hidden_activations: data.activations.A1,
      output_activations: data.activations.A2,
    });

    updateProbBars(data.probabilities);
  });

  // ── Botón Cámara ──────────────────────────────────────────
  document.getElementById("btn-camera").addEventListener("click", () => {
    Camera.start("video-feed");
    document.getElementById("btn-camera").disabled  = true;
    document.getElementById("btn-capture").disabled = false;
    const ph = document.getElementById("video-placeholder");
    if (ph) ph.classList.add("hidden");
  });

  document.getElementById("btn-capture").addEventListener("click", () => {
    Camera.captureAndPredict((data) => {
      document.getElementById("stat-pred").textContent = data.digit;

      const img = document.getElementById("img-thumbnail");
      img.src = data.thumbnail_b64;

      const empty = document.getElementById("thumb-empty");
      if (empty) empty.classList.add("hidden");

      updateProbBars(data.probabilities);
      showToast(`Predicción: dígito ${data.digit}`, "success");
    });
  });

  // ── Exportar ──────────────────────────────────────────────
  document.getElementById("btn-export").addEventListener("click", () => {
    Charts.exportPNG();
    showToast("Gráfica exportada", "info");
  });

  // ── Barras de probabilidad Softmax ────────────────────────
  function updateProbBars(probs) {
    const maxVal = Math.max(...probs);

    probs.forEach((p, i) => {
      const bar  = document.getElementById(`prob-bar-${i}`);
      const lbl  = document.getElementById(`prob-val-${i}`);
      const row  = document.getElementById(`prob-row-${i}`);

      if (bar) bar.style.width = (p * 100).toFixed(1) + "%";
      if (lbl) lbl.textContent = (p * 100).toFixed(1) + "%";

      if (row) {
        const isWinner = p === maxVal && p > 0;
        row.classList.toggle("winner", isWinner);
        row.classList.toggle("dimmed", p < maxVal * 0.4 && maxVal > 0.1);
      }
    });
  }

  // ── Utilidades de estado ──────────────────────────────────
  function setStatus(state) {
    const pill  = document.getElementById("status-pill");
    const label = document.getElementById("status-label");
    if (!pill || !label) return;

    pill.className = "status-pill";
    const map = {
      ready:    { cls: "",         text: "Listo" },
      training: { cls: "training", text: "Entrenando" },
      stopped:  { cls: "stopped",  text: "Detenido" },
    };
    const s = map[state] || map.ready;
    if (s.cls) pill.classList.add(s.cls);
    label.textContent = s.text;
  }

  function showProgress(total) {
    const wrap = document.getElementById("progress-wrap");
    if (wrap) wrap.classList.add("visible");
    setProgress(0, total);
  }

  function hideProgress() {
    const wrap = document.getElementById("progress-wrap");
    if (wrap) wrap.classList.remove("visible");
  }

  function setProgress(current, total) {
    const fill = document.getElementById("progress-fill");
    const text = document.getElementById("progress-text");
    if (!fill || !text) return;
    const pct = total > 0 ? (current / total) * 100 : 0;
    fill.style.width = pct.toFixed(1) + "%";
    text.textContent = `${current} / ${total}`;
  }

  // ── Toasts ────────────────────────────────────────────────
  function showToast(msg, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const icons = { success: "✓", error: "✕", info: "ℹ" };
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span class="toast-icon">${icons[type] || icons.info}</span><span>${msg}</span>`;

    container.appendChild(toast);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => toast.classList.add("show"));
    });

    setTimeout(() => {
      toast.classList.remove("show");
      toast.classList.add("hide");
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }
});
