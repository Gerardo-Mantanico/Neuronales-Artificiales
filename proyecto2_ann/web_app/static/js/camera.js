/**
 * camera.js — acceso a la cámara y captura de frames
 */

const Camera = (() => {
  let videoEl, stream;

  async function start(videoElementId) {
    videoEl = document.getElementById(videoElementId);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoEl.srcObject = stream;
      await videoEl.play();
    } catch (err) {
      const container = document.getElementById("toast-container");
      if (container) {
        const toast = document.createElement("div");
        toast.className = "toast toast-error";
        toast.innerHTML = `<span class="toast-icon">✕</span><span>Cámara: ${err.message}</span>`;
        container.appendChild(toast);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => toast.classList.add("show"));
        });
        setTimeout(() => { toast.classList.remove("show"); setTimeout(() => toast.remove(), 300); }, 3500);
      }
    }
  }

  function stop() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
  }

  async function capture() {
    if (!videoEl || !stream) return null;

    const tmp = document.createElement("canvas");
    tmp.width  = videoEl.videoWidth;
    tmp.height = videoEl.videoHeight;
    tmp.getContext("2d").drawImage(videoEl, 0, 0);
    return tmp.toDataURL("image/jpeg", 0.85);
  }

  async function captureAndPredict(onResult) {
    const base64 = await capture();
    if (!base64) return;

    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 }),
    });

    if (!resp.ok) { console.error("Error en predicción:", resp.status); return; }
    onResult(await resp.json());
  }

  return { start, stop, capture, captureAndPredict };
})();
