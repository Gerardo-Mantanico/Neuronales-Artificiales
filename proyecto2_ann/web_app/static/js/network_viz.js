/**
 * network_viz.js
 * Visualización de la red neuronal en un <canvas>.
 * Dibuja 3 capas: entrada (muestra), oculta (64), salida (10).
 * Las conexiones cambian de color y grosor según el valor del peso.
 */

const NetViz = (() => {
  let canvas, ctx;
  // Posiciones calculadas de cada nodo
  let nodesInput = [], nodesHidden = [], nodesOutput = [];

  // Muestra representativa de nodos de entrada a dibujar
  const N_INPUT_DISPLAY = 20;
  const N_HIDDEN = 64;
  const N_OUTPUT = 10;

  function init(canvasId) {
    canvas = document.getElementById(canvasId);
    ctx = canvas.getContext("2d");
    _calcPositions();
    draw({});
  }

  function _calcPositions() {
    const W = canvas.width;
    const H = canvas.height;
    const capas = [N_INPUT_DISPLAY, N_HIDDEN, N_OUTPUT];
    const xs = [W * 0.15, W * 0.5, W * 0.85];
    const results = [[], [], []];

    capas.forEach((n, ci) => {
      const espaciado = H / (n + 1);
      for (let i = 0; i < n; i++) {
        results[ci].push({ x: xs[ci], y: espaciado * (i + 1) });
      }
    });

    [nodesInput, nodesHidden, nodesOutput] = results;
  }

  function draw(data) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Conexiones entrada → oculta (muestra aleatoria para no saturar)
    _drawConnections(nodesInput, nodesHidden.slice(0, 10), data.W1_sample, 0.3);

    // Conexiones oculta → salida
    _drawConnections(nodesHidden.slice(0, 10), nodesOutput, data.W2_sample, 0.5);

    // Nodos
    _drawLayer(nodesInput,  "#64b5f6", data.input_activations,  6);
    _drawLayer(nodesHidden, "#81c784", data.hidden_activations, 5);
    _drawLayer(nodesOutput, "#e57373", data.output_activations, 8);

    // Etiquetas
    _drawLabels();
  }

  function _drawConnections(from, to, weightMatrix, alpha) {
    from.forEach((fn, fi) => {
      to.forEach((tn, ti) => {
        let weight = 0;
        if (weightMatrix && weightMatrix[ti] && weightMatrix[ti][fi] !== undefined) {
          weight = weightMatrix[ti][fi];
        }
        const intensidad = Math.min(Math.abs(weight) * 5, 1);
        const grosor = 0.5 + intensidad * 2;

        ctx.beginPath();
        ctx.moveTo(fn.x, fn.y);
        ctx.lineTo(tn.x, tn.y);
        ctx.lineWidth = grosor;

        if (weight >= 0) {
          ctx.strokeStyle = `rgba(33, 150, 243, ${alpha * intensidad + 0.05})`;
        } else {
          ctx.strokeStyle = `rgba(244, 67, 54, ${alpha * intensidad + 0.05})`;
        }
        ctx.stroke();
      });
    });
  }

  function _drawLayer(nodes, colorBase, activations, radius) {
    nodes.forEach((n, i) => {
      const act = activations ? Math.min(activations[i] || 0, 1) : 0;
      ctx.beginPath();
      ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = _interpolateColor("#1a1a2e", colorBase, act);
      ctx.fill();
      ctx.strokeStyle = colorBase;
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

  function _interpolateColor(hex1, hex2, t) {
    const r1 = parseInt(hex1.slice(1,3),16), g1 = parseInt(hex1.slice(3,5),16), b1 = parseInt(hex1.slice(5,7),16);
    const r2 = parseInt(hex2.slice(1,3),16), g2 = parseInt(hex2.slice(3,5),16), b2 = parseInt(hex2.slice(5,7),16);
    const r = Math.round(r1 + (r2-r1)*t);
    const g = Math.round(g1 + (g2-g1)*t);
    const b = Math.round(b1 + (b2-b1)*t);
    return `rgb(${r},${g},${b})`;
  }

  function _drawLabels() {
    ctx.fillStyle = "#ccc";
    ctx.font = "12px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Entrada (784)", nodesInput[0].x, 18);
    ctx.fillText("Oculta (64)",   nodesHidden[0].x, 18);
    ctx.fillText("Salida (10)",   nodesOutput[0].x, 18);
  }

  function updateWeights(weightsData) {
    draw(weightsData);
  }

  return { init, updateWeights };
})();
