/**
 * charts.js
 * Gráfica de pérdida y accuracy en tiempo real usando Chart.js.
 * La gráfica puede exportarse como PNG.
 */

const Charts = (() => {
  let lossChart;

  function init(canvasId) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    lossChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Loss",
            data: [],
            borderColor: "#ef5350",
            backgroundColor: "rgba(239,83,80,0.1)",
            tension: 0.3,
            fill: true,
            yAxisID: "yLoss",
          },
          {
            label: "Accuracy",
            data: [],
            borderColor: "#42a5f5",
            backgroundColor: "rgba(66,165,245,0.1)",
            tension: 0.3,
            fill: true,
            yAxisID: "yAcc",
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: {
            title: { display: true, text: "Epoch", color: "#ccc" },
            ticks: { color: "#ccc" },
            grid:  { color: "rgba(255,255,255,0.05)" },
          },
          yLoss: {
            type: "linear",
            position: "left",
            title: { display: true, text: "Loss", color: "#ef5350" },
            ticks: { color: "#ef5350" },
          },
          yAcc: {
            type: "linear",
            position: "right",
            min: 0, max: 1,
            title: { display: true, text: "Accuracy", color: "#42a5f5" },
            ticks: { color: "#42a5f5", callback: v => (v * 100).toFixed(0) + "%" },
            grid: { drawOnChartArea: false },
          },
        },
        plugins: {
          legend: { labels: { color: "#ccc" } },
        },
      },
    });
  }

  function update(epoch, loss, accuracy) {
    lossChart.data.labels.push(epoch);
    lossChart.data.datasets[0].data.push(loss);
    lossChart.data.datasets[1].data.push(accuracy);
    lossChart.update("none");   // sin animación para fluidez
  }

  function exportPNG() {
    const url = lossChart.toBase64Image("image/png", 1.0);
    const a = document.createElement("a");
    a.href = url;
    a.download = "loss_chart.png";
    a.click();
  }

  function reset() {
    lossChart.data.labels = [];
    lossChart.data.datasets.forEach(d => (d.data = []));
    lossChart.update("none");
  }

  return { init, update, exportPNG, reset };
})();
